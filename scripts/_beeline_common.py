from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NETWORK_ALIASES = {
    "specific": "Specific",
    "non-specific": "Non-Specific",
    "nonspecific": "Non-Specific",
    "non_specific": "Non-Specific",
    "string": "STRING",
    "lofgof": "Lofgof",
    "lof/gof": "Lofgof",
    "lof-gof": "Lofgof",
}


@dataclass(frozen=True, slots=True)
class BeelineCondition:
    network_type: str
    dataset: str
    gene_setting: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.network_type, self.dataset, self.gene_setting)


def canonical_network(value: str) -> str:
    stripped = value.strip()
    return NETWORK_ALIASES.get(stripped.lower(), stripped)


def gene_setting_sort_key(value: str) -> tuple[int, str]:
    digits = "".join(character for character in value if character.isdigit())
    return (int(digits) if digits else 0, value)


def discover_conditions(data_root: Path) -> list[BeelineCondition]:
    dataset_root = data_root / "dataset"
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"BEELINE dataset directory not found: {dataset_root}"
        )
    conditions: list[BeelineCondition] = []
    for network_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for dataset_dir in sorted(path for path in network_dir.iterdir() if path.is_dir()):
            settings = sorted(
                (path for path in dataset_dir.iterdir() if path.is_dir()),
                key=lambda path: gene_setting_sort_key(path.name),
            )
            for setting_dir in settings:
                if (setting_dir / "BL--ExpressionData.csv").is_file():
                    conditions.append(
                        BeelineCondition(
                            canonical_network(network_dir.name),
                            dataset_dir.name,
                            setting_dir.name,
                        )
                    )
    if not conditions:
        raise RuntimeError(f"No BEELINE conditions found under {dataset_root}")
    return conditions


def select_conditions(
    conditions: Sequence[BeelineCondition],
    networks: Sequence[str] | None,
    datasets: Sequence[str] | None,
    gene_settings: Sequence[str] | None,
) -> list[BeelineCondition]:
    network_filter = (
        {canonical_network(value) for value in networks} if networks else None
    )
    dataset_filter = set(datasets) if datasets else None
    setting_filter = set(gene_settings) if gene_settings else None
    selected = [
        condition
        for condition in conditions
        if (network_filter is None or condition.network_type in network_filter)
        and (dataset_filter is None or condition.dataset in dataset_filter)
        and (setting_filter is None or condition.gene_setting in setting_filter)
    ]
    if selected:
        return selected
    available = {
        "networks": sorted({condition.network_type for condition in conditions}),
        "datasets": sorted({condition.dataset for condition in conditions}),
        "gene_settings": sorted(
            {condition.gene_setting for condition in conditions},
            key=gene_setting_sort_key,
        ),
    }
    raise ValueError(f"The filters selected no BEELINE conditions: {available}")


def files_for(
    data_root: Path,
    condition: BeelineCondition,
    fold: int,
) -> dict[str, Path]:
    data_dir = (
        data_root
        / "dataset"
        / condition.network_type
        / condition.dataset
        / condition.gene_setting
    )
    split_dir = (
        data_root
        / "Train_validation_test"
        / condition.network_type
        / condition.dataset
        / condition.gene_setting
        / f"Fold_{fold}"
    )
    return {
        "expression": data_dir / "BL--ExpressionData.csv",
        "train": split_dir / "Train_set.csv",
        "validation": split_dir / "Validation_set.csv",
        "test": split_dir / "Test_set.csv",
    }


def required_config(config_dir: Path, network_type: str) -> Path:
    path = config_dir / f"{canonical_network(network_type)}.json"
    if not path.is_file():
        raise FileNotFoundError(f"BEELINE preset not found: {path}")
    return path


def missing_files(paths: Iterable[Path]) -> list[str]:
    return [str(path) for path in paths if not path.is_file()]
