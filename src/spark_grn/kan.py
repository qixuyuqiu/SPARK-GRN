

from __future__ import annotations

from collections.abc import Iterable, Iterator
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * step
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features * (grid_size + spline_order))
        )
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.scale_noise = float(scale_noise)
        self.scale_base = float(scale_base)
        self.scale_spline = float(scale_spline)
        self.base_activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            coefficients = self.curve_to_coefficients(
                self.grid.T[self.spline_order : -self.spline_order], noise
            )
            self.spline_weight.copy_(coefficients)
            nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )

    def b_splines(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.in_features:
            raise ValueError(
                f"KAN input must have shape [batch, {self.in_features}]."
            )
        expanded = values.unsqueeze(-1)
        bases = ((expanded >= self.grid[:, :-1]) & (expanded < self.grid[:, 1:])).to(
            values.dtype
        )
        for order in range(1, self.spline_order + 1):
            bases = (
                (expanded - self.grid[:, : -(order + 1)])
                / (self.grid[:, order:-1] - self.grid[:, : -(order + 1)])
                * bases[:, :, :-1]
                + (self.grid[:, order + 1 :] - expanded)
                / (self.grid[:, order + 1 :] - self.grid[:, 1:-order])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def curve_to_coefficients(
        self, x_values: torch.Tensor, y_values: torch.Tensor
    ) -> torch.Tensor:
        design = self.b_splines(x_values).transpose(0, 1)
        response = y_values.transpose(0, 1)
        solution = torch.linalg.lstsq(design, response).solution
        return solution.permute(2, 0, 1).contiguous().view(self.out_features, -1)

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        return (
            self.spline_weight.view(
                self.out_features,
                self.in_features,
                self.grid_size + self.spline_order,
            )
            * self.spline_scaler.unsqueeze(-1)
        ).view(self.out_features, -1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = F.linear(self.base_activation(values), self.base_weight)
        spline = F.linear(
            self.b_splines(values).view(values.shape[0], -1),
            self.scaled_spline_weight,
        )
        return base + spline


def _kan_layer(
    in_features: int, out_features: int, grid_size: int, spline_order: int
) -> KANLayer:
    return KANLayer(in_features, out_features, grid_size, spline_order)


class ResidualDualPathKANPredictor(nn.Module):
    """Direction-preserving KANv3 with a learnable bottleneck residual.

    The ordered ``[TF || target]`` path is initialized first and remains the
    primary scorer. A compressed pair representation is added as a bounded,
    learnable residual, preserving TF-target direction while allowing the
    model to recover complementary bottleneck interactions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        output_dim: int = 1,
        grid_size: int = 5,
        spline_order: int = 3,
        alpha_init: float = 0.8,
    ) -> None:
        super().__init__()
        widths = [int(width) for width in hidden_dims]
        if not widths:
            raise ValueError("Residual KANv3 requires at least one hidden layer.")
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must lie strictly between 0 and 1.")

        first_hidden = widths[0]
        self.tf_norm = nn.LayerNorm(input_dim)
        self.target_norm = nn.LayerNorm(input_dim)
        self.directional_first_layer = _kan_layer(
            input_dim * 2, first_hidden, grid_size, spline_order
        )

        self.shared_layers = nn.ModuleList()
        current_dim = first_hidden
        for width in widths[1:]:
            self.shared_layers.append(
                _kan_layer(current_dim, width, grid_size, spline_order)
            )
            current_dim = width
        self.output_layer = _kan_layer(
            current_dim, output_dim, grid_size, spline_order
        )

        self.feature_mixer = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.SiLU(),
        )
        self.bottleneck_first_layer = _kan_layer(
            input_dim, first_hidden, grid_size, spline_order
        )
        self.bottleneck_norm = nn.LayerNorm(first_hidden)
        initial_logit = math.log(alpha_init / (1.0 - alpha_init))
        self.residual_logit = nn.Parameter(torch.tensor(initial_logit))

    def residual_alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_logit)

    def iter_kan_layers(self) -> Iterator[KANLayer]:
        yield self.directional_first_layer
        yield from self.shared_layers
        yield self.output_layer
        yield self.bottleneck_first_layer

    def raw_spline_l1(self) -> torch.Tensor:
        return torch.stack(
            [layer.spline_weight.abs().mean() for layer in self.iter_kan_layers()]
        ).sum()

    def forward(
        self, tf_embedding: torch.Tensor, target_embedding: torch.Tensor
    ) -> torch.Tensor:
        if tf_embedding.shape != target_embedding.shape:
            raise ValueError("TF and target embeddings must have identical shapes.")
        normalized_pair = torch.cat(
            [self.tf_norm(tf_embedding), self.target_norm(target_embedding)],
            dim=-1,
        )
        directional = self.directional_first_layer(normalized_pair)
        raw_pair = torch.cat([tf_embedding, target_embedding], dim=-1)
        bottleneck = self.bottleneck_norm(
            self.bottleneck_first_layer(self.feature_mixer(raw_pair))
        )
        values = directional + self.residual_alpha() * bottleneck
        for layer in self.shared_layers:
            values = layer(values)
        return self.output_layer(values)
