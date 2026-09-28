"""Training, evaluation, checkpointing, and edge scoring."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .config import ExperimentConfig, ModelConfig
from .data import DatasetBundle
from .metrics import evaluate_predictions
from .model import SPARKGRN


@dataclass(slots=True)
class TrainingResult:
    checkpoint_path: Path
    history_path: Path
    metrics_path: Path
    best_epoch: int
    best_validation_aupr: float
    test_metrics: dict[str, Any]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _runtime_metadata(device: torch.device, deterministic: bool) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for distribution in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "tqdm",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "deterministic_algorithms_requested": bool(deterministic),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata["cuda_device_name"] = torch.cuda.get_device_name(index)
        metadata["cuda_runtime"] = torch.version.cuda
    return metadata


def _loader(split: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    pairs = torch.as_tensor(split[:, :2], dtype=torch.long)
    labels = torch.as_tensor(split[:, 2], dtype=torch.float32)
    return DataLoader(
        TensorDataset(pairs, labels), batch_size=batch_size, shuffle=shuffle
    )


def _json_ready(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "curve_data"}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    bundle: DatasetBundle,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    expression = bundle.expression.to(device)
    gene_activity = (
        bundle.gene_activity.to(device) if bundle.gene_activity is not None else None
    )
    edge_index = bundle.edge_index.to(device)
    edge_weight = bundle.edge_weight.to(device)
    # Evaluation is deterministic because VIB uses its posterior mean in eval mode.
    gene_features, _ = model.encode_genes(
        expression, edge_index, edge_weight, gene_activity
    )
    for pairs, batch_labels in loader:
        logits = model.score_pairs(gene_features, pairs.to(device))
        predictions.append(torch.sigmoid(logits).cpu())
        labels.append(batch_labels)
    return evaluate_predictions(
        torch.cat(labels).numpy(),
        torch.cat(predictions).numpy(),
        threshold=threshold,
    )


def train_model(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    output_dir: str | Path,
    device: str | torch.device | None = None,
    deterministic: bool = False,
    show_progress: bool = True,
    model_factory: Callable[[ModelConfig, int, int | None], nn.Module] | None = None,
) -> TrainingResult:
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device} was requested, but CUDA is not available."
        )
    seed_everything(config.training.seed, deterministic=deterministic)
    if config.model.modality == "BOTH" and bundle.gene_activity is None:
        raise ValueError("The selected complete multimodal model requires GeneActivity.")

    n_atac_cells = (
        bundle.gene_activity.shape[1]
        if bundle.gene_activity is not None
        else None
    )
    model = (
        model_factory(config.model, bundle.expression.shape[1], n_atac_cells)
        if model_factory is not None
        else SPARKGRN(
            config.model,
            n_rna_cells=bundle.expression.shape[1],
            n_atac_cells=n_atac_cells,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    train_labels = bundle.train[:, 2]
    if config.training.fixed_pos_weight is None:
        positive = int((train_labels == 1).sum())
        negative = int((train_labels == 0).sum())
        pos_weight = np.sqrt(negative / positive) if positive else 1.0
    else:
        pos_weight = config.training.fixed_pos_weight
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )
    train_loader = _loader(bundle.train, config.training.batch_size, shuffle=True)
    validation_loader = _loader(
        bundle.validation, config.training.batch_size, shuffle=False
    )
    test_loader = _loader(bundle.test, config.training.batch_size, shuffle=False)

    expression = bundle.expression.to(device)
    gene_activity = (
        bundle.gene_activity.to(device) if bundle.gene_activity is not None else None
    )
    edge_index = bundle.edge_index.to(device)
    edge_weight = bundle.edge_weight.to(device)
    checkpoint_path = output_dir / "best_model.pth"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "test_metrics.json"
    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": config.dataset,
                "seed": config.training.seed,
                "runtime": _runtime_metadata(device, deterministic),
                "output_directory": str(output_dir.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    best_aupr = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.time()

    for epoch in range(1, config.training.epochs + 1):
        model.train()
        running_loss = 0.0
        warmup_epochs = max(
            1,
            round(config.training.epochs * config.training.kl_warmup_fraction),
        )
        kl_scale = (
            min(1.0, epoch / warmup_epochs)
            if config.training.kl_warmup_fraction > 0
            else 1.0
        )
        iterator = tqdm(train_loader, leave=False, disable=not show_progress)
        for pairs, labels in iterator:
            pairs = pairs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, kl_loss = model(
                expression, edge_index, edge_weight, gene_activity, pairs
            )
            loss = criterion(logits, labels)
            loss = loss + config.training.kan_l1 * model.regularization_loss()
            loss = loss + config.training.kl_beta * kl_scale * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip_norm
            )
            optimizer.step()
            running_loss += float(loss.detach())
            iterator.set_description(f"epoch {epoch}")
            iterator.set_postfix(loss=f"{float(loss.detach()):.4f}")

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            bundle,
            device,
            config.training.threshold,
        )
        mean_loss = running_loss / max(1, len(train_loader))
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_loss,
                "validation_auc": validation_metrics["auc"],
                "validation_aupr": validation_metrics["aupr"],
            }
        )
        print(
            f"Epoch {epoch:03d} | loss={mean_loss:.5f} | "
            f"val_AUROC={validation_metrics['auc']:.5f} | "
            f"val_AUPRC={validation_metrics['aupr']:.5f}"
        )
        if validation_metrics["aupr"] > best_aupr:
            best_aupr = validation_metrics["aupr"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.training.patience:
            break

    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    test_metrics = evaluate_model(
        model, test_loader, bundle, device, config.training.threshold
    )
    test_metrics["best_epoch"] = best_epoch
    test_metrics["best_validation_aupr"] = best_aupr
    test_metrics["train_time_seconds"] = time.time() - started
    diagnostics = getattr(model, "diagnostics", None)
    if callable(diagnostics):
        test_metrics["model_diagnostics"] = diagnostics()
    metrics_path.write_text(
        json.dumps(_json_ready(test_metrics), indent=2), encoding="utf-8"
    )
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        metrics_path=metrics_path,
        best_epoch=best_epoch,
        best_validation_aupr=best_aupr,
        test_metrics=test_metrics,
    )


@torch.no_grad()
def score_candidate_edges(
    model: SPARKGRN,
    bundle: DatasetBundle,
    pairs: np.ndarray,
    device: str | torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    device = torch.device(device)
    expression = bundle.expression.to(device)
    gene_activity = (
        bundle.gene_activity.to(device) if bundle.gene_activity is not None else None
    )
    features, _ = model.encode_genes(
        expression,
        bundle.edge_index.to(device),
        bundle.edge_weight.to(device),
        gene_activity,
    )
    probabilities: list[torch.Tensor] = []
    pair_tensor = torch.as_tensor(pairs, dtype=torch.long)
    for start in range(0, pair_tensor.shape[0], batch_size):
        batch = pair_tensor[start : start + batch_size].to(device)
        probabilities.append(torch.sigmoid(model.score_pairs(features, batch)).cpu())
    return torch.cat(probabilities).numpy()
