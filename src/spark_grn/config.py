"""Typed configuration for one architecture with dataset-specific presets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal


Modality = Literal["BOTH", "RNA_ONLY"]


@dataclass(slots=True)
class ModelConfig:
    """Architecture parameters.

    Component types are fixed. Dataset presets may alter capacity,
    regularization, and initialization, but never swap the architecture.
    """

    hidden_dim: int = 64
    gnn_layers: int = 2
    gat_heads: int = 8
    rna_dropout: float = 0.2

    vib_bottleneck_dim: int | None = 256
    vib_hidden_multiplier: int = 2
    vib_logvar_min: float = -6.0
    vib_logvar_max: float = 2.0
    atac_dropout: float = 0.2

    fusion_heads: int = 4
    fusion_layers: int = 1
    fusion_ff_multiplier: int = 2
    fusion_dropout: float = 0.2
    fusion_gate_floor: float = 0.1
    fusion_initial_atac_weight: float = 0.5
    fusion_residual_alpha_init: float = 0.05

    kan_hidden_dims: tuple[int, ...] = (64, 16)
    kan_grid_size: int = 3
    kan_spline_order: int = 3
    kan_residual_alpha_init: float = 0.8
    modality: Modality = "BOTH"

    def validate(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if self.gnn_layers < 1:
            raise ValueError("gnn_layers must be positive.")
        if self.hidden_dim % self.gat_heads:
            raise ValueError("hidden_dim must be divisible by gat_heads.")
        if self.hidden_dim % self.fusion_heads:
            raise ValueError("hidden_dim must be divisible by fusion_heads.")
        if self.vib_bottleneck_dim is not None and self.vib_bottleneck_dim < 1:
            raise ValueError("vib_bottleneck_dim must be positive or null.")
        if self.vib_hidden_multiplier < 1:
            raise ValueError("vib_hidden_multiplier must be positive.")
        if self.vib_logvar_min >= self.vib_logvar_max:
            raise ValueError("vib_logvar_min must be less than vib_logvar_max.")
        if self.fusion_layers < 1 or self.fusion_ff_multiplier < 1:
            raise ValueError("Fusion depth and FFN multiplier must be positive.")
        if not 0.0 <= self.fusion_gate_floor < 0.5:
            raise ValueError("fusion_gate_floor must be in [0, 0.5).")
        if not self.fusion_gate_floor < self.fusion_initial_atac_weight < 1.0 - self.fusion_gate_floor:
            raise ValueError(
                "fusion_initial_atac_weight must lie inside the bounded gate interval."
            )
        for name in ("fusion_residual_alpha_init", "kan_residual_alpha_init"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in (0, 1).")
        for name in ("rna_dropout", "atac_dropout", "fusion_dropout"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if not self.kan_hidden_dims or min(self.kan_hidden_dims) < 1:
            raise ValueError("kan_hidden_dims must contain positive widths.")
        if self.modality not in {"BOTH", "RNA_ONLY"}:
            raise ValueError("modality must be BOTH or RNA_ONLY.")


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 1024
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20
    fixed_pos_weight: float | None = None
    kan_l1: float = 1e-4
    kl_beta: float = 1e-4
    kl_warmup_fraction: float = 0.25
    gradient_clip_norm: float = 1.0
    threshold: float = 0.5
    seed: int = 42

    def validate(self) -> None:
        if min(self.batch_size, self.epochs, self.patience) < 1:
            raise ValueError("batch_size, epochs, and patience must be positive.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid optimizer settings.")
        if self.fixed_pos_weight is not None and self.fixed_pos_weight <= 0:
            raise ValueError("fixed_pos_weight must be positive when specified.")
        if self.kan_l1 < 0 or self.kl_beta < 0:
            raise ValueError("Regularization coefficients cannot be negative.")
        if not 0.0 <= self.kl_warmup_fraction <= 1.0:
            raise ValueError("kl_warmup_fraction must be in [0, 1].")
        if self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")


@dataclass(slots=True)
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: str = "unknown"
    preset_status: str = "development"
    notes: str = ""

    def validate(self) -> None:
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a complete, explicit JSON configuration."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_values = dict(payload.get("model", {}))
    if "kan_hidden_dims" in model_values:
        model_values["kan_hidden_dims"] = tuple(model_values["kan_hidden_dims"])
    config = ExperimentConfig(
        model=ModelConfig(**model_values),
        training=TrainingConfig(**payload.get("training", {})),
        dataset=str(payload.get("dataset", "unknown")),
        preset_status=str(payload.get("preset_status", "development")),
        notes=str(payload.get("notes", "")),
    )
    config.validate()
    return config


def apply_override(config: ExperimentConfig, expression: str) -> None:
    """Apply one command-line override such as model.fusion_layers=2."""

    key, separator, raw_value = expression.partition("=")
    if not separator or "." not in key:
        raise ValueError("Overrides must use section.field=value syntax.")
    section_name, field_name = key.split(".", 1)
    if section_name not in {"model", "training"}:
        raise ValueError("Override section must be model or training.")
    section = getattr(config, section_name)
    if not hasattr(section, field_name):
        raise ValueError(f"Unknown override field: {key}")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    if field_name == "kan_hidden_dims":
        value = tuple(value)
    setattr(section, field_name, value)
    config.validate()

