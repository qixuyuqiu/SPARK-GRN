"""Input loading and training-prior graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass(slots=True)
class DatasetBundle:
    genes: np.ndarray
    expression: torch.Tensor
    gene_activity: torch.Tensor | None
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def _find_column(columns: list[str], candidates: set[str]) -> str | None:
    return next((column for column in columns if column.lower() in candidates), None)


def load_edge_split(
    path: str | Path,
    gene_to_index: dict[str, int],
) -> np.ndarray:
    """Load TF-target-label triples encoded as indices or gene symbols."""

    frame = pd.read_csv(path, sep=None, engine="python")
    columns = [str(column) for column in frame.columns]
    source = _find_column(columns, {"gene1", "tf", "source"})
    target = _find_column(columns, {"gene2", "tg", "target"})
    label = _find_column(columns, {"label", "score"})
    if source is None or target is None or label is None:
        if frame.shape[1] < 3:
            raise ValueError(f"{path} must contain at least three columns.")
        raw = frame.iloc[:, -3:].to_numpy()
    else:
        raw = frame[[source, target, label]].to_numpy()
    if raw.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    try:
        source_indices = raw[:, 0].astype(np.int64)
        target_indices = raw[:, 1].astype(np.int64)
        if np.all(source_indices >= 0) and np.all(target_indices >= 0):
            n_genes = len(gene_to_index)
            keep = (source_indices < n_genes) & (target_indices < n_genes)
            return np.column_stack(
                [source_indices[keep], target_indices[keep], raw[keep, 2].astype(float)]
            ).astype(np.float64)
    except (TypeError, ValueError):
        pass

    mapped: list[tuple[int, int, float]] = []
    for tf, tg, value in raw:
        tf_key = str(tf).strip().upper()
        tg_key = str(tg).strip().upper()
        if tf_key in gene_to_index and tg_key in gene_to_index:
            mapped.append((gene_to_index[tf_key], gene_to_index[tg_key], float(value)))
    return np.asarray(mapped, dtype=np.float64).reshape(-1, 3)


def build_training_prior(
    train_edges: np.ndarray,
    n_genes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the normalized directed prior graph from training positives only."""

    positives = train_edges[train_edges[:, 2] == 1, :2].astype(np.int64)
    if positives.shape[0] == 0:
        raise ValueError("Training data contains no positive TF-target edges.")
    adjacency = sp.coo_matrix(
        (np.ones(positives.shape[0], dtype=np.float32), (positives[:, 0], positives[:, 1])),
        shape=(n_genes, n_genes),
        dtype=np.float32,
    )
    # Matches the historical implementation: A+I followed by symmetric
    # degree normalization. GATv2 also adds internal computational self-loops.
    with_loops = adjacency + sp.eye(n_genes, dtype=np.float32)
    row_sum = np.asarray(with_loops.sum(1)).reshape(-1)
    inverse_sqrt = np.zeros_like(row_sum, dtype=np.float64)
    np.power(row_sum, -0.5, out=inverse_sqrt, where=row_sum > 0)
    degree = sp.diags(inverse_sqrt)
    normalized = with_loops.dot(degree).transpose().dot(degree).tocoo()
    edge_index = torch.from_numpy(
        np.vstack([normalized.row, normalized.col]).astype(np.int64)
    )
    edge_weight = torch.from_numpy(normalized.data.astype(np.float32))
    return edge_index, edge_weight


def load_gene_activity(path: str | Path, genes: np.ndarray) -> torch.Tensor:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, index_col=0)
        frame.index = frame.index.astype(str).str.upper()
        expected = pd.Index([str(gene).upper() for gene in genes])
        missing = expected.difference(frame.index)
        if len(missing):
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"GeneActivity is missing {len(missing)} RNA genes (e.g. {preview})."
            )
        values = frame.loc[expected].to_numpy(dtype=np.float32)
    else:
        values = np.load(path, allow_pickle=False).astype(np.float32)
        if values.shape[0] != len(genes):
            raise ValueError(
                "NumPy GeneActivity input has no gene labels and must match the RNA row order."
            )
    if values.ndim != 2:
        raise ValueError("GeneActivity input must be a two-dimensional matrix.")
    return torch.from_numpy(np.nan_to_num(values))


def load_dataset(
    expression_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    gene_activity_path: str | Path | None = None,
) -> DatasetBundle:
    expression_frame = pd.read_csv(expression_path, index_col=0)
    genes = expression_frame.index.astype(str).to_numpy()
    if pd.Index(genes.str.upper() if hasattr(genes, "str") else [g.upper() for g in genes]).duplicated().any():
        raise ValueError("Expression matrix contains duplicated gene identifiers.")
    gene_to_index = {str(gene).upper(): index for index, gene in enumerate(genes)}
    expression = torch.from_numpy(
        np.nan_to_num(expression_frame.to_numpy(dtype=np.float32))
    )
    train = load_edge_split(train_path, gene_to_index)
    validation = load_edge_split(validation_path, gene_to_index)
    test = load_edge_split(test_path, gene_to_index)
    for name, split in (("train", train), ("validation", validation), ("test", test)):
        if split.shape[0] == 0:
            raise ValueError(f"The {name} split is empty after gene mapping.")
    edge_index, edge_weight = build_training_prior(train, len(genes))
    gene_activity = (
        load_gene_activity(gene_activity_path, genes)
        if gene_activity_path is not None
        else None
    )
    return DatasetBundle(
        genes=genes,
        expression=expression,
        gene_activity=gene_activity,
        edge_index=edge_index,
        edge_weight=edge_weight,
        train=train,
        validation=validation,
        test=test,
    )
