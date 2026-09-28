"""Score directed TF-target candidates with a trained SPARK-GRN model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from spark_grn.config import load_config
from spark_grn.data import load_dataset
from spark_grn.model import SPARKGRN
from spark_grn.training import score_candidate_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--gene-activity", type=Path)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def _find_column(columns: list[str], candidates: set[str]) -> str | None:
    return next((column for column in columns if column.lower() in candidates), None)


def load_candidate_pairs(
    path: Path,
    genes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, sep=None, engine="python")
    columns = [str(column) for column in frame.columns]
    source = _find_column(columns, {"gene1", "tf", "source"})
    target = _find_column(columns, {"gene2", "tg", "target"})
    if source is None or target is None:
        if frame.shape[1] < 2:
            raise ValueError("Candidate input must contain at least two columns.")
        raw = frame.iloc[:, :2].to_numpy()
    else:
        raw = frame[[source, target]].to_numpy()
    if raw.shape[0] == 0:
        raise ValueError("Candidate input is empty.")

    gene_names = np.asarray([str(gene) for gene in genes])
    gene_to_index = {name.upper(): index for index, name in enumerate(gene_names)}
    pairs: list[tuple[int, int]] = []
    tf_names: list[str] = []
    tg_names: list[str] = []
    for tf, tg in raw:
        try:
            tf_index = int(tf)
            tg_index = int(tg)
            if not (0 <= tf_index < len(genes) and 0 <= tg_index < len(genes)):
                raise ValueError
        except (TypeError, ValueError):
            tf_key = str(tf).strip().upper()
            tg_key = str(tg).strip().upper()
            if tf_key not in gene_to_index or tg_key not in gene_to_index:
                raise ValueError(f"Unknown candidate pair: {tf!r} -> {tg!r}")
            tf_index = gene_to_index[tf_key]
            tg_index = gene_to_index[tg_key]
        pairs.append((tf_index, tg_index))
        tf_names.append(gene_names[tf_index])
        tg_names.append(gene_names[tg_index])
    return (
        np.asarray(pairs, dtype=np.int64),
        np.asarray(tf_names),
        np.asarray(tg_names),
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    config = load_config(args.config)
    if config.model.modality == "BOTH" and args.gene_activity is None:
        raise SystemExit("--gene-activity is required for a multimodal config.")
    bundle = load_dataset(
        args.expression,
        args.train,
        args.validation,
        args.test,
        args.gene_activity,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = SPARKGRN(
        config.model,
        n_rna_cells=bundle.expression.shape[1],
        n_atac_cells=(
            bundle.gene_activity.shape[1]
            if bundle.gene_activity is not None
            else None
        ),
    ).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    pairs, tf_names, tg_names = load_candidate_pairs(args.candidates, bundle.genes)
    scores = score_candidate_edges(
        model,
        bundle,
        pairs,
        device=device,
        batch_size=args.batch_size,
    )
    order = np.argsort(-scores, kind="stable")
    output = pd.DataFrame(
        {
            "TF": tf_names[order],
            "TG": tg_names[order],
            "TF_index": pairs[order, 0],
            "TG_index": pairs[order, 1],
            "score": scores[order],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output)} ranked candidates to {args.output}")


if __name__ == "__main__":
    main()
