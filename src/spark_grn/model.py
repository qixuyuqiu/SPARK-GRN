"""Unified production architecture for SPARK-GRN."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .encoders import StableVIBATACEncoder
from .encoders_direction_aware import DirectionAwareGATv2RNAEncoder
from .fusion import UncertaintyAwareResidualFusion
from .kan import ResidualDualPathKANPredictor


class SPARKGRN(nn.Module):
    """Direction-aware GATv2 + stable VIB + adaptive fusion + KANv3."""

    def __init__(
        self,
        config: ModelConfig,
        n_rna_cells: int,
        n_atac_cells: int | None = None,
    ) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.modality = config.modality
        self.rna_encoder = DirectionAwareGATv2RNAEncoder(
            n_cells=n_rna_cells,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.gnn_layers,
            heads=config.gat_heads,
            dropout=config.rna_dropout,
            input_edge_index_is_transposed=True,
        )
        if self.modality == "BOTH":
            if n_atac_cells is None or n_atac_cells < 1:
                raise ValueError("n_atac_cells is required for multimodal training.")
            self.atac_encoder: StableVIBATACEncoder | None = StableVIBATACEncoder(
                n_atac_cells=n_atac_cells,
                hidden_dim=config.hidden_dim * config.vib_hidden_multiplier,
                output_dim=config.hidden_dim,
                dropout=config.atac_dropout,
                bottleneck_dim=config.vib_bottleneck_dim,
                logvar_min=config.vib_logvar_min,
                logvar_max=config.vib_logvar_max,
            )
            self.fusion: UncertaintyAwareResidualFusion | None = (
                UncertaintyAwareResidualFusion(
                    rna_dim=config.hidden_dim,
                    atac_dim=config.hidden_dim,
                    hidden_dim=config.hidden_dim,
                    n_heads=config.fusion_heads,
                    n_layers=config.fusion_layers,
                    ff_multiplier=config.fusion_ff_multiplier,
                    dropout=config.fusion_dropout,
                    gate_floor=config.fusion_gate_floor,
                    initial_atac_weight=config.fusion_initial_atac_weight,
                    alpha_init=config.fusion_residual_alpha_init,
                )
            )
        else:
            self.atac_encoder = None
            self.fusion = None
        self.predictor = ResidualDualPathKANPredictor(
            input_dim=config.hidden_dim,
            hidden_dims=config.kan_hidden_dims,
            output_dim=1,
            grid_size=config.kan_grid_size,
            spline_order=config.kan_spline_order,
            alpha_init=config.kan_residual_alpha_init,
        )

    def encode_genes(
        self,
        expression: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gene_activity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rna_features = self.rna_encoder(expression, edge_index, edge_weight)
        if self.modality == "RNA_ONLY":
            return rna_features, expression.new_zeros(())
        if gene_activity is None or self.atac_encoder is None or self.fusion is None:
            raise ValueError("GeneActivity is required when modality='BOTH'.")
        atac_features, kl_loss, atac_logvar = self.atac_encoder(gene_activity)
        fused = self.fusion(rna_features, atac_features, atac_logvar)
        return fused, kl_loss

    def score_pairs(
        self,
        gene_features: torch.Tensor,
        query_pairs: torch.Tensor,
    ) -> torch.Tensor:
        if query_pairs.ndim != 2 or query_pairs.shape[1] != 2:
            raise ValueError("query_pairs must have shape [n_pairs, 2].")
        tf_features = gene_features[query_pairs[:, 0]]
        target_features = gene_features[query_pairs[:, 1]]
        return self.predictor(tf_features, target_features).squeeze(-1)

    def forward(
        self,
        expression: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        gene_activity: torch.Tensor | None,
        query_pairs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gene_features, kl_loss = self.encode_genes(
            expression,
            edge_index,
            edge_weight,
            gene_activity,
        )
        return self.score_pairs(gene_features, query_pairs), kl_loss

    def regularization_loss(self) -> torch.Tensor:
        return self.predictor.raw_spline_l1()

    def diagnostics(self) -> dict[str, object]:
        output: dict[str, object] = {
            "parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
            "kan_residual_alpha": float(
                self.predictor.residual_alpha().detach().cpu()
            ),
        }
        if self.atac_encoder is not None:
            output["vib_bottleneck_dim"] = self.atac_encoder.bottleneck_dim
        if self.fusion is not None:
            output.update(self.fusion.diagnostics())
        return output

