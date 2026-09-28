"""Evaluation metrics for sparse TF-target link prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(int(k), scores.size)
    if k < 1:
        return float("nan")
    selected = np.argsort(scores)[-k:]
    return float(labels[selected].sum() / k)


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have identical shapes.")
    labels = (labels > 0).astype(np.int64)
    if np.unique(labels).size != 2:
        raise ValueError("AUROC and AUPRC require both positive and negative labels.")
    predicted = (probabilities >= threshold).astype(np.int64)
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probabilities)
    fpr, tpr, _ = roc_curve(labels, probabilities)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "aupr": float(average_precision_score(labels, probabilities)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "p@100": precision_at_k(labels, probabilities, 100),
        "p@500": precision_at_k(labels, probabilities, 500),
        "threshold": float(threshold),
        "curve_data": {
            "precision": precision_curve,
            "recall": recall_curve,
            "fpr": fpr,
            "tpr": tpr,
        },
    }

