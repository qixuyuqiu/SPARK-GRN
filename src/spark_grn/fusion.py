"""Uncertainty-aware weak-residual RNA-ATAC fusion."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class UncertaintyAwareResidualFusion(nn.Module):
    """Fuse aligned modalities with a bounded adaptive gate.

    The aligned RNA/ATAC mixture is the base representation. Transformer
    cross-modal messages enter through a learnable bounded residual, which
    reduces the risk that an unstable attention update overwrites both inputs.
    """

    def __init__(
        self,
        rna_dim: int,
        atac_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        ff_multiplier: int,
        dropout: float,
        gate_floor: float,
        initial_atac_weight: float,
        alpha_init: float,
    ) -> None:
        super().__init__()
        self.gate_floor = float(gate_floor)
        self.rna_projection = (
            nn.Linear(rna_dim, hidden_dim) if rna_dim != hidden_dim else nn.Identity()
        )
        self.atac_projection = (
            nn.Linear(atac_dim, hidden_dim) if atac_dim != hidden_dim else nn.Identity()
        )
        self.rna_norm = nn.LayerNorm(hidden_dim)
        self.atac_norm = nn.LayerNorm(hidden_dim)
        self.modality_embedding = nn.Embedding(2, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.uncertainty_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.gate_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate_output = nn.Linear(hidden_dim, 2)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.alpha_logit = nn.Parameter(
            torch.tensor(math.log(alpha_init / (1.0 - alpha_init)))
        )
        self.last_gate_mean: torch.Tensor | None = None
        self.last_gate_quantiles: torch.Tensor | None = None
        self._reset_parameters(initial_atac_weight)

    def _reset_parameters(self, initial_atac_weight: float) -> None:
        nn.init.normal_(self.modality_embedding.weight, mean=0.0, std=0.02)
        for block in (self.uncertainty_encoder, self.gate_hidden):
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate_output.weight)
        raw_atac = (initial_atac_weight - self.gate_floor) / (
            1.0 - 2.0 * self.gate_floor
        )
        raw_atac = min(max(raw_atac, 1e-6), 1.0 - 1e-6)
        with torch.no_grad():
            self.gate_output.bias.copy_(
                torch.tensor([math.log(1.0 - raw_atac), math.log(raw_atac)])
            )

    @property
    def residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        rna_features: torch.Tensor,
        atac_features: torch.Tensor,
        atac_logvar: torch.Tensor,
    ) -> torch.Tensor:
        if rna_features.shape[0] != atac_features.shape[0]:
            raise ValueError("RNA and ATAC must contain the same genes.")
        rna = self.rna_norm(self.rna_projection(rna_features))
        atac = self.atac_norm(self.atac_projection(atac_features))
        tokens = torch.stack((rna, atac), dim=1)
        token_ids = torch.arange(2, device=tokens.device)
        transformer_input = tokens + self.modality_embedding(token_ids).unsqueeze(0)
        transformed = self.transformer(transformer_input)
        uncertainty = self.uncertainty_encoder(atac_logvar)
        gate_features = torch.cat(
            (transformed[:, 0], transformed[:, 1], uncertainty),
            dim=-1,
        )
        raw_weights = torch.softmax(
            self.gate_output(self.gate_hidden(gate_features)),
            dim=-1,
        )
        weights = self.gate_floor + (
            1.0 - 2.0 * self.gate_floor
        ) * raw_weights
        update = transformed - transformer_input
        fused_base = weights[:, :1] * rna + weights[:, 1:] * atac
        fused_update = (
            weights[:, :1] * update[:, 0]
            + weights[:, 1:] * update[:, 1]
        )
        with torch.no_grad():
            self.last_gate_mean = weights.mean(dim=0).detach().cpu()
            self.last_gate_quantiles = torch.quantile(
                weights[:, 1].detach(),
                torch.tensor([0.1, 0.5, 0.9], device=weights.device),
            ).cpu()
        return self.output_norm(
            fused_base + self.residual_scale * fused_update
        )

    def diagnostics(self) -> dict[str, float | list[float] | None]:
        return {
            "fusion_residual_scale": float(self.residual_scale.detach().cpu()),
            "gate_mean": (
                self.last_gate_mean.tolist() if self.last_gate_mean is not None else None
            ),
            "atac_gate_quantiles": (
                self.last_gate_quantiles.tolist()
                if self.last_gate_quantiles is not None
                else None
            ),
        }

