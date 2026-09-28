
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

import pandas as pd

from _beeline_common import (
    discover_conditions,
    files_for,
    missing_files,
    required_config,
    select_conditions,
)


METRICS = ["auc", "aupr", "f1", "accuracy", "precision", "recall"]
METRIC_COLUMNS = ["AUC", "AUPR", "F1", "ACCURACY", "PRECISION", "RECALL"]
IDENTITY_COLUMNS = [
    "NetworkType",
    "Dataset",
    "GeneSetting",
    "ConfigHash",
    "Repeat",
    "Fold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--networks", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--gene-settings", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Model-initialization repeats on the same fixed BEELINE folds.",
    )
    parser.add_argument("--gpu-ids", nargs="+", default=["0"])
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260823)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def stable_model_seed(base_seed: int, parts: tuple[object, ...]) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31 - 1)


def device_name(gpu: str) -> str:
    return "cpu" if gpu.lower() == "cpu" else f"cuda:{gpu}"


def run_task(
    task: dict[str, object],
    project_root: Path,
    output: Path,
    python: Path,
    deterministic: bool,
) -> dict[str, object]:
    started = time.time()
    run_dir = (
        output
        / "runs"
        / str(task["NetworkType"])
        / str(task["Dataset"])
        / str(task["GeneSetting"])
        / f"Config_{task['ConfigHash']}"
        / f"Repeat_{int(task['Repeat']):02d}"
        / f"Fold_{int(task['Fold'])}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    command = [
        str(python),
        "-m",
        "spark_grn.cli.train",
        "--config",
        str(task["ConfigFile"]),
        "--expression",
        str(task["ExpressionFile"]),
        "--train",
        str(task["TrainFile"]),
        "--validation",
        str(task["ValidationFile"]),
        "--test",
        str(task["TestFile"]),
        "--output",
        str(run_dir),
        "--device",
        device_name(str(task["GPU"])),
        "--seed",
        str(task["ModelSeed"]),
        "--no-progress",
    ]
    if deterministic:
        command.append("--deterministic")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
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
            "Error": f"Missing metrics: {metrics_path}",
        }
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        **task,
        "Status": "Success",
        "Time(s)": elapsed,
        "Error": "",
        "BestEpoch": metrics.get("best_epoch"),
        "BestValidationAUPR": metrics.get("best_validation_aupr"),
        **{
            metric.upper() if metric != "aupr" else "AUPR": metrics.get(metric)
            for metric in METRICS
        },
    }


def summarize(frame: pd.DataFrame, output: Path) -> None:
    successful = frame.loc[frame["Status"].eq("Success")].copy()
    if successful.empty:
        return
    group_keys = ["NetworkType", "Dataset", "GeneSetting", "ConfigHash"]
    repeat_rows: list[dict[str, object]] = []
    for keys, group in successful.groupby(group_keys + ["Repeat"], dropna=False):
        row = dict(zip(group_keys + ["Repeat"], keys))
        row["NFolds"] = len(group)
        for metric in METRIC_COLUMNS:
            row[f"{metric}_Mean"] = group[metric].mean()
            row[f"{metric}_SD"] = group[metric].std(ddof=1)
        repeat_rows.append(row)
    repeat_frame = pd.DataFrame(repeat_rows).sort_values(group_keys + ["Repeat"])
    atomic_csv(repeat_frame, output / "Repeat_Mean_Results.csv")

    condition_rows: list[dict[str, object]] = []
    for keys, group in repeat_frame.groupby(group_keys, dropna=False):
        row = dict(zip(group_keys, keys))
        row["NRepeats"] = len(group)
        row["CompleteFiveFoldRepeats"] = int((group["NFolds"] == 5).sum())
        for metric in METRIC_COLUMNS:
            row[f"{metric}_Mean"] = group[f"{metric}_Mean"].mean()
            row[f"{metric}_SD"] = group[f"{metric}_Mean"].std(ddof=1)
        condition_rows.append(row)
    condition_frame = pd.DataFrame(condition_rows).sort_values(group_keys)
    atomic_csv(condition_frame, output / "Dataset_Repeat_Summary.csv")

    for metric in ("AUC", "AUPR"):
        wide = condition_frame.pivot_table(
            index="Dataset",
            columns=["NetworkType", "GeneSetting"],
            values=f"{metric}_Mean",
            aggfunc="first",
        )
        wide.to_csv(output / f"BEELINE_{metric}_Wide.csv", encoding="utf-8-sig")
    failed = frame.loc[frame["Status"].ne("Success")].copy()
    if not failed.empty:
        atomic_csv(failed, output / "Failed_Tasks.csv")


def main() -> None:
    args = parse_args()
    project_root = (args.project_root or Path(__file__).resolve().parents[1]).resolve()
    data_root = (args.data_root or project_root / "Data" / "BEELINE").resolve()
    config_dir = (args.config_dir or project_root / "configs" / "beeline").resolve()
    output = (args.output or project_root / "outputs" / "beeline_5fold"/"best").resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.repeats < 1 or args.max_workers < 1:
        raise ValueError("--repeats and --max-workers must be positive")
    if not args.gpu_ids:
        raise ValueError("At least one --gpu-ids value is required")

    conditions = select_conditions(
        discover_conditions(data_root),
        args.networks,
        args.datasets,
        args.gene_settings,
    )
    existing = pd.DataFrame()
    detail_path = output / "Detailed_Fold_Results.csv"
    completed: set[tuple[object, ...]] = set()
    if detail_path.is_file() and not args.no_resume:
        existing = pd.read_csv(detail_path)
        success = existing.loc[existing["Status"].eq("Success")]
        if set(IDENTITY_COLUMNS).issubset(success.columns):
            completed = set(map(tuple, success[IDENTITY_COLUMNS].itertuples(index=False, name=None)))

    tasks: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for condition in conditions:
        config_path = required_config(config_dir, condition.network_type)
        digest = config_hash(config_path)
        for repeat in range(1, args.repeats + 1):
            for fold in args.folds:
                files = files_for(data_root, condition, fold)
                missing = missing_files(files.values())
                if missing:
                    raise FileNotFoundError("Missing BEELINE inputs: " + "; ".join(missing))
                identity = (*condition.key, digest, repeat, fold)

                # IMPORTANT:
                # Resume identity may include ConfigHash so different configs
                # are treated as different completed tasks.
                # Model seed MUST NOT include ConfigHash, otherwise Optuna
                # changes the random initialization every trial.
                seed_identity = (
                    condition.network_type,
                    condition.dataset,
                    condition.gene_setting,
                    repeat,
                    fold,
                )
                task = {
                    "NetworkType": condition.network_type,
                    "Dataset": condition.dataset,
                    "GeneSetting": condition.gene_setting,
                    "ConfigHash": digest,
                    "Repeat": repeat,
                    "Fold": fold,
                    "SplitID": "Fixed_BEELINE_5Fold",
                    "ModelSeed": stable_model_seed(args.base_seed, seed_identity),
                    "GPU": args.gpu_ids[len(manifest_rows) % len(args.gpu_ids)],
                    "ConfigFile": str(config_path),
                    "ExpressionFile": str(files["expression"]),
                    "TrainFile": str(files["train"]),
                    "ValidationFile": str(files["validation"]),
                    "TestFile": str(files["test"]),
                }
                manifest_rows.append({**task, "ResumeSkipped": identity in completed})
                if identity not in completed:
                    tasks.append(task)

    atomic_csv(pd.DataFrame(manifest_rows), output / "Task_Manifest.csv")
    manifest = {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "config_dir": str(config_dir),
        "output": str(output),
        "conditions": len(conditions),
        "fixed_folds": args.folds,
        "initialization_repeats": args.repeats,
        "planned_tasks": len(tasks),
        "total_selected_tasks": len(manifest_rows),
        "note": "Repeat is a model-initialization repeat on the same fixed BEELINE folds, not a regenerated data split.",
    }
    (output / "Run_Manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if args.dry_run:
        for task in tasks[:12]:
            print(task)
        return

    rows = existing.to_dict("records") if not existing.empty else []
    if tasks:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    run_task,
                    task,
                    project_root,
                    output,
                    args.python.resolve(),
                    args.deterministic,
                ): task
                for task in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    row = future.result()
                except Exception as exc:
                    task = futures[future]
                    row = {**task, "Status": "Failed", "Error": str(exc)}
                rows.append(row)
                current = pd.DataFrame(rows).drop_duplicates(IDENTITY_COLUMNS, keep="last")
                atomic_csv(current, detail_path)
                print(
                    f"[{index}/{len(tasks)}] {row['NetworkType']} | {row['Dataset']} | "
                    f"{row['GeneSetting']} | repeat={row['Repeat']} | fold={row['Fold']} | {row['Status']}"
                )
    frame = pd.DataFrame(rows).drop_duplicates(IDENTITY_COLUMNS, keep="last")
    if not frame.empty:
        atomic_csv(frame, detail_path)
        summarize(frame, output)


if __name__ == "__main__":
    main()