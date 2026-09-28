from __future__ import annotations

import argparse
import json
from pathlib import Path

from spark_grn.config import apply_override, load_config
from spark_grn.data import load_dataset
from spark_grn.training import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unified SPARK-GRN model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expression", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--gene-activity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    for expression in args.overrides:
        apply_override(config, expression)
    if args.seed is not None:
        config.training.seed = args.seed
    if config.model.modality == "BOTH" and args.gene_activity is None:
        raise SystemExit("--gene-activity is required for a multimodal config.")
    bundle = load_dataset(
        args.expression,
        args.train,
        args.validation,
        args.test,
        args.gene_activity,
    )
    result = train_model(
        bundle,
        config,
        args.output,
        device=args.device,
        deterministic=args.deterministic,
        show_progress=not args.no_progress,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.test_metrics.items()
                if key != "curve_data"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

