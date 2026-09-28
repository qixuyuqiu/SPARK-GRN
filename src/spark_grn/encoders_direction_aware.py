
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, LayerNorm
from torch_geometric.utils import remove_self_loops


class DirectionAwareGATv2RNAEncoder(nn.Module):
    """Encode expression with independent upstream and downstream branches.

    The forward branch propagates TF -> target, so targets aggregate upstream
    regulator information. The reverse branch propagates target -> TF, so TFs
    aggregate downstream target context. Branch parameters are independent.

    Parameters
    ----------
    input_edge_index_is_transposed
        ``True`` for the current SPARK-GRN data loader, whose historical
        normalized adjacency stores biological TF -> target edges as
        target -> TF in PyG ``edge_index`` convention.
    """

    def __init__(
        self,
        n_cells: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        *,
        input_edge_index_is_transposed: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        if num_layers < 1:
            raise ValueError("num_layers must be at least one.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.dropout = float(dropout)
        self.hidden_dim = int(hidden_dim)
        self.input_edge_index_is_transposed = bool(input_edge_index_is_transposed)
        self.input_projection = nn.Linear(n_cells, hidden_dim)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

        head_dim = hidden_dim // heads
        self.forward_layers = nn.ModuleList(
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=1,
            )
            for _ in range(num_layers)
        )
        self.reverse_layers = nn.ModuleList(
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=1,
            )
            for _ in range(num_layers)
        )
        self.fusion_layers = nn.ModuleList(
            nn.Linear(3 * hidden_dim, hidden_dim) for _ in range(num_layers)
        )
        self.norms = nn.ModuleList(
            LayerNorm(hidden_dim) for _ in range(num_layers)
        )
        self.output_projection = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for fusion in self.fusion_layers:
            nn.init.xavier_uniform_(fusion.weight)
            if fusion.bias is not None:
                nn.init.zeros_(fusion.bias)
        if isinstance(self.output_projection, nn.Linear):
            nn.init.xavier_uniform_(self.output_projection.weight)
            if self.output_projection.bias is not None:
                nn.init.zeros_(self.output_projection.bias)

    def biological_edge_indices(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return TF->target and target->TF edges with aligned edge features."""

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, n_edges]; "
                f"received {tuple(edge_index.shape)}."
            )
        if edge_weight is not None and edge_weight.shape[0] != edge_index.shape[1]:
            raise ValueError("edge_weight must align one-to-one with edge_index.")

        edge_attr = None
        if edge_weight is not None:
            edge_attr = edge_weight.unsqueeze(-1) if edge_weight.ndim == 1 else edge_weight

        stored_edges, edge_attr = remove_self_loops(edge_index, edge_attr)
        forward_edges = (
            stored_edges.flip(0)
            if self.input_edge_index_is_transposed
            else stored_edges
        )
        reverse_edges = forward_edges.flip(0)
        return forward_edges, reverse_edges, edge_attr

    def forward(
        self,
        expression: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if expression.ndim != 2:
            raise ValueError(
                "expression must have shape [n_genes, n_cells]; "
                f"received {tuple(expression.shape)}."
            )

        forward_edges, reverse_edges, edge_attr = self.biological_edge_indices(
            edge_index, edge_weight
        )
        values = F.leaky_relu(
            self.input_projection(expression), negative_slope=0.2
        )
        values = F.dropout(values, p=self.dropout, training=self.training)

        for forward_layer, reverse_layer, fusion, norm in zip(
            self.forward_layers,
            self.reverse_layers,
            self.fusion_layers,
            self.norms,
        ):
            residual = values
            forward_values = forward_layer(
                values, forward_edges, edge_attr=edge_attr
            )
            reverse_values = reverse_layer(
                values, reverse_edges, edge_attr=edge_attr
            )
            values = fusion(
                torch.cat([residual, forward_values, reverse_values], dim=-1)
            )
            values = norm(values)
            values = F.leaky_relu(values, negative_slope=0.2)
            values = F.dropout(values, p=self.dropout, training=self.training)
            values = values + residual

        return self.output_projection(values)
