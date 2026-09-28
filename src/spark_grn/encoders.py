"""Stable probabilistic ATAC encoder for gene-level accessibility."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StableVIBATACEncoder(nn.Module):
    """Bounded VIB encoder with configurable input bottleneck capacity.

    A null bottleneck dimension preserves the full input width. This keeps the
    component type fixed while allowing datasets with informative ATAC inputs
    to retain more capacity.
    """

    def __init__(
        self,
        n_atac_cells: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        bottleneck_dim: int | None = 256,
        logvar_min: float = -6.0,
        logvar_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(n_atac_cells)
        self.output_dim = int(output_dim)
        self.bottleneck_dim = (
            self.input_dim if bottleneck_dim is None else int(bottleneck_dim)
        )
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.input_projection = nn.Linear(self.input_dim, self.bottleneck_dim)
        self.input_norm = nn.LayerNorm(self.bottleneck_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Softplus(),
            nn.Dropout(dropout),
        )
        self.mu_projection = nn.Linear(hidden_dim, output_dim)
        self.logvar_projection = nn.Linear(hidden_dim, output_dim)
        self.self_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self, gene_activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gene_activity.ndim != 2 or gene_activity.shape[1] != self.input_dim:
            raise ValueError(
                "GeneActivity must have shape [n_genes, n_atac_cells]; "
                f"received {tuple(gene_activity.shape)}."
            )
        projected = F.softplus(
            self.input_norm(self.input_projection(gene_activity))
        )
        hidden = self.encoder(projected)
        mu = self.mu_projection(hidden)
        logvar = self.logvar_projection(hidden).clamp(
            min=self.logvar_min,
            max=self.logvar_max,
        )
        if self.training:
            standard_deviation = torch.exp(0.5 * logvar)
            latent = mu + torch.randn_like(standard_deviation) * standard_deviation
            kl_loss = -0.5 * (
                1.0 + logvar - mu.square() - logvar.exp()
            ).mean()
        else:
            latent = mu
            kl_loss = gene_activity.new_zeros(())
        return latent * self.self_gate(mu), kl_loss, logvar

