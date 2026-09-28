"""Run repeated five-fold evaluation on paired RNA-ATAC datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pandas as pd


TEST_METRICS = ["auc", "aupr", "f1", "accuracy", "precision", "recall"]

TEST_COLUMN_MAP = {
    "auc": "AUC",
    "aupr": "AUPR",
    "f1": "F1",
    "accuracy": "ACCURACY",
    "precision": "PRECISION",
    "recall": "RECALL",
}

TRAINING_STAT_COLUMN_MAP = {
    "best_epoch": "BestEpoch",
    "best_validation_aupr": "BestValidationAUPR",
    "best_train_loss": "BestTrainLoss",
    "best_train_bce": "BestTrainBCE",
    "best_train_kl": "BestTrainKL",
    "best_train_weighted_kl": "BestTrainWeightedKL",
    "best_train_kan_raw": "BestTrainKANRaw",
    "best_train_weighted_kan": "BestTrainWeightedKAN",
    "best_kl_scale": "BestKLScale",
    "best_effective_kl_beta": "BestEffectiveKLBeta",
    "best_kl_bce_ratio": "BestKLBCE_Ratio",
    "kl_beta": "KLBeta",
    "kl_warmup_fraction": "KLWarmupFraction",
}

SUMMARY_COLUMNS = [
    "BestValidationAUPR",
    "AUC",
    "AUPR",
    "F1",
    "ACCURACY",
    "PRECISION",
    "RECALL",
    "BestEpoch",
    "BestTrainLoss",
    "BestTrainBCE",
    "BestTrainKL",
    "BestTrainWeightedKL",
    "BestTrainKANRaw",
    "BestTrainWeightedKAN",
    "BestKLScale",
    "BestEffectiveKLBeta",
    "BestKLBCE_Ratio",
    "KLBeta",
    "KLWarmupFraction",
    "Time(s)",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_hash(path: Path) -> str:
    """Hash JSON semantics rather than raw whitespace/formatting."""
    payload = load_json(path)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def stable_model_seed(base_seed: int, dataset: str, repeat: int, fold: int) -> int:
    """Stable paired seed independent of configuration and CLI dataset ordering."""
    identity = f"{dataset}|{int(repeat)}|{int(fold)}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % 2_000_000_000
    seed = (int(base_seed) + offset) % 2_147_483_647
    return seed if seed > 0 else 1


def split_seed(root: Path, dataset: str, repeat: int) -> int | None:
    manifest = (
        root
        / "Train_validation_test_repeated"
        / dataset
        / f"Repeat_{repeat:02d}"
        / "split_manifest.json"
    )
    if not manifest.is_file():
        return None
    payload = load_json(manifest)
    value = payload.get("split_seed")
    return None if value is None else int(value)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root (default: inferred from this script).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("Data/Multi-omics"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/multiomics"),
        help="Directory containing one <Dataset>.json config per dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/multiomics_repeated_cv/all_best"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["3T3", "A549", "GM12878", "K562"],
    )

    repeat_group = parser.add_mutually_exclusive_group()
    repeat_group.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Use Repeat_01 ... Repeat_N (default: 10).",
    )
    repeat_group.add_argument(
        "--repeat-ids",
        nargs="+",
        type=int,
        help="Explicit repeats, e.g. --repeat-ids 1 or --repeat-ids 1 2 3.",
    )

    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--gpu-ids", nargs="+", default=["0"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260723)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_under(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def requested_repeats(args: argparse.Namespace) -> list[int]:
    if args.repeat_ids is not None:
        repeats = sorted(set(int(x) for x in args.repeat_ids))
    else:
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        repeats = list(range(1, int(args.repeats) + 1))
    if not repeats or any(x < 1 for x in repeats):
        raise ValueError("Repeat IDs must be positive integers.")
    return repeats


def paths_for(root: Path, dataset: str, repeat: int, fold: int) -> dict[str, Path]:
    dataset_dir = root / "dataset" / dataset
    split_dir = (
        root
        / "Train_validation_test_repeated"
        / dataset
        / f"Repeat_{repeat:02d}"
        / f"Fold_{fold}"
    )
    return {
        "expression": dataset_dir / "ExpressionData.csv",
        "gene_activity": dataset_dir / "GeneScoreData.csv",
        "train": split_dir / "Train_set.csv",
        "validation": split_dir / "Validation_set.csv",
        "test": split_dir / "Test_set.csv",
    }


def run_task(
    task: dict[str, object],
    *,
    project_root: Path,
    data_root: Path,
    config_dir: Path,
    output: Path,
    python_exe: Path,
    deterministic: bool,
) -> dict[str, object]:
    started = time.time()

    dataset = str(task["Dataset"])
    repeat = int(task["Repeat"])
    fold = int(task["Fold"])
    gpu = str(task["GPU"])
    seed = int(task["ModelSeed"])
    cfg_hash = str(task["ConfigHash"])

    files = paths_for(data_root, dataset, repeat, fold)
    config_path = config_dir / f"{dataset}.json"

    missing = [
        str(path)
        for path in [*files.values(), config_path]
        if not path.is_file()
    ]
    if missing:
        return {
            **task,
            "Status": "Failed",
            "Time(s)": time.time() - started,
            "Error": "Missing: " + "; ".join(missing),
        }

    # ConfigHash is deliberately included so different Optuna configs cannot
    # overwrite logs/checkpoints/test_metrics.json from another configuration.
    run_dir = (
        output
        / "runs"
        / dataset
        / f"Config_{cfg_hash}"
        / f"Repeat_{repeat:02d}"
        / f"Fold_{fold}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    command = [
        str(python_exe),
        "-m",
        "spark_grn.cli.train",
        "--config",
        str(config_path),
        "--expression",
        str(files["expression"]),
        "--gene-activity",
        str(files["gene_activity"]),
        "--train",
        str(files["train"]),
        "--validation",
        str(files["validation"]),
        "--test",
        str(files["test"]),
        "--output",
        str(run_dir),
        "--device",
        "cpu" if gpu.lower() == "cpu" else f"cuda:{gpu}",
        "--seed",
        str(seed),
        "--no-progress",
    ]
    if deterministic:
        command.append("--deterministic")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(project_root / "src")
        + os.pathsep
        + environment.get("PYTHONPATH", "")
    )
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )

    elapsed = time.time() - started
    if process.returncode != 0:
        return {
            **task,
            "Status": "Failed",
            "Time(s)": elapsed,
            "Error": str(log_path),
        }

    metrics_path = run_dir / "test_metrics.json"
    if not metrics_path.is_file():
        return {
            **task,
            "Status": "Failed",
            "Time(s)": elapsed,
            "Error": f"Missing test_metrics.json: {metrics_path}",
        }

    try:
        metrics = load_json(metrics_path)
    except Exception as exc:
        return {
            **task,
            "Status": "Failed",
            "Time(s)": elapsed,
            "Error": f"Failed to parse {metrics_path}: {exc!r}",
        }

    required = [*TEST_METRICS, "best_validation_aupr"]
    missing_metrics = [key for key in required if key not in metrics]
    if missing_metrics:
        return {
            **task,
            "Status": "Failed",
            "Time(s)": elapsed,
            "Error": (
                "Missing required fields in test_metrics.json: "
                + ", ".join(missing_metrics)
            ),
        }

    row: dict[str, object] = {
        **task,
        "Status": "Success",
        "Time(s)": elapsed,
        "Error": "",
    }

    for json_name, csv_name in TEST_COLUMN_MAP.items():
        row[csv_name] = float(metrics[json_name])

    for json_name, csv_name in TRAINING_STAT_COLUMN_MAP.items():
        row[csv_name] = metrics.get(json_name, float("nan"))

    return row


def summarize(frame: pd.DataFrame, output: Path) -> None:
    successful = frame.loc[frame["Status"].astype(str).eq("Success")].copy()
    if successful.empty:
        print("No successful tasks; summary files were not created.")
        return

    numeric_columns = [column for column in SUMMARY_COLUMNS if column in successful.columns]
    for column in numeric_columns:
        successful[column] = pd.to_numeric(successful[column], errors="coerce")

    group_keys = ["Dataset", "ConfigHash", "Repeat"]

    repeat_means = (
        successful.groupby(group_keys, as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )
    fold_counts = (
        successful.groupby(group_keys, as_index=False)
        .size()
        .rename(columns={"size": "NFolds"})
    )
    repeat_frame = fold_counts.merge(repeat_means, on=group_keys, how="left")
    repeat_frame = repeat_frame.sort_values(group_keys).reset_index(drop=True)
    atomic_csv(repeat_frame, output / "Repeat_Mean_Results.csv")

    rows: list[dict[str, object]] = []
    for (dataset, cfg_hash), group in repeat_frame.groupby(
        ["Dataset", "ConfigHash"], sort=True
    ):
        row: dict[str, object] = {
            "Dataset": dataset,
            "ConfigHash": cfg_hash,
            "NRepeats": int(len(group)),
            "MeanNFolds": float(group["NFolds"].mean()),
        }
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_Mean"] = (
                float(values.mean()) if not values.empty else float("nan")
            )
            row[f"{column}_SD"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
        rows.append(row)

    atomic_csv(pd.DataFrame(rows), output / "Dataset_Repeat_Summary.csv")


def main() -> int:
    args = parse_args()

    project_root = (
        args.project_root.expanduser().resolve()
        if args.project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    data_root = resolve_under(project_root, args.data_root)
    config_dir = resolve_under(project_root, args.config_dir)
    output = resolve_under(project_root, args.output)
    python_exe = args.python.expanduser().resolve()

    if not project_root.is_dir():
        raise FileNotFoundError(f"Missing project root: {project_root}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Missing data root: {data_root}")
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Missing config dir: {config_dir}")
    if not python_exe.is_file():
        raise FileNotFoundError(f"Missing Python executable: {python_exe}")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if not args.gpu_ids:
        raise ValueError("--gpu-ids cannot be empty")
    if not args.folds or any(int(x) < 1 for x in args.folds):
        raise ValueError("--folds must contain positive integers")

    repeats = requested_repeats(args)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "Detailed_Fold_Results.csv"

    existing = pd.DataFrame()
    completed: set[tuple[str, str, int, int]] = set()

    if detail_path.is_file() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        required_identity = {"Dataset", "ConfigHash", "Repeat", "Fold", "Status"}
        if required_identity.issubset(existing.columns):
            successful = existing.loc[
                existing["Status"].astype(str).eq("Success")
            ]
            completed = {
                (
                    str(row.Dataset),
                    str(row.ConfigHash),
                    int(row.Repeat),
                    int(row.Fold),
                )
                for row in successful[
                    ["Dataset", "ConfigHash", "Repeat", "Fold"]
                ].itertuples(index=False)
            }

    tasks: list[dict[str, object]] = []

    for dataset in args.datasets:
        config_path = config_dir / f"{dataset}.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing dataset config: {config_path}")
        cfg_hash = config_hash(config_path)

        for repeat in repeats:
            current_split_seed = split_seed(data_root, dataset, repeat)
            for fold in args.folds:
                resume_identity = (dataset, cfg_hash, int(repeat), int(fold))
                if resume_identity in completed:
                    continue

                tasks.append(
                    {
                        "Dataset": dataset,
                        "Repeat": int(repeat),
                        "Fold": int(fold),
                        "ConfigHash": cfg_hash,
                        "SplitID": (
                            f"Repeated5Fold_Repeat{int(repeat):02d}_Fold{int(fold)}"
                        ),
                        "SplitSeed": current_split_seed,
                        "ModelSeed": stable_model_seed(
                            args.base_seed,
                            dataset,
                            int(repeat),
                            int(fold),
                        ),
                        "GPU": args.gpu_ids[len(tasks) % len(args.gpu_ids)],
                    }
                )

    print("=" * 110)
    print("SPARK-GRN matched multiomics repeated five-fold runner")
    print("=" * 110)
    print(f"Project root : {project_root}")
    print(f"Data root    : {data_root}")
    print(f"Config dir   : {config_dir}")
    print(f"Output       : {output}")
    print(f"Datasets     : {args.datasets}")
    print(f"Repeats      : {repeats}")
    print(f"Folds        : {args.folds}")
    print(f"GPU IDs      : {args.gpu_ids}")
    print(f"Max workers  : {args.max_workers}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Planned tasks: {len(tasks)}")
    print("=" * 110)

    if args.dry_run:
        for task in tasks[:20]:
            print(task)
        return 0

    rows = [] if existing.empty or args.no_resume else existing.to_dict("records")

    if tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_task,
                    task,
                    project_root=project_root,
                    data_root=data_root,
                    config_dir=config_dir,
                    output=output,
                    python_exe=python_exe,
                    deterministic=args.deterministic,
                ): task
                for task in tasks
            }

            completed_count = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        **task,
                        "Status": "Failed",
                        "Time(s)": float("nan"),
                        "Error": f"Unhandled runner exception: {exc!r}",
                    }

                rows.append(row)
                completed_count += 1

                interim = pd.DataFrame(rows)
                if {"Dataset", "ConfigHash", "Repeat", "Fold"}.issubset(
                    interim.columns
                ):
                    interim = interim.drop_duplicates(
                        ["Dataset", "ConfigHash", "Repeat", "Fold"],
                        keep="last",
                    )
                atomic_csv(interim, detail_path)

                print(
                    f"[{completed_count}/{len(tasks)}] {row['Dataset']} "
                    f"repeat={row['Repeat']} fold={row['Fold']} {row['Status']}"
                )
                if row["Status"] == "Failed":
                    print(f"  Error: {row.get('Error', '')}")

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates(
            ["Dataset", "ConfigHash", "Repeat", "Fold"], keep="last"
        )
        frame = frame.sort_values(
            ["Dataset", "ConfigHash", "Repeat", "Fold"]
        ).reset_index(drop=True)
        atomic_csv(frame, detail_path)
        summarize(frame, output)

    print(f"Detailed results: {detail_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
