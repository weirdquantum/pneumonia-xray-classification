"""Classification metrics, threshold selection and patient-level bootstrap CIs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def binary_metrics(y: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Metrics with PNEUMONIA (label 1) as the positive class."""
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def div(a: float, b: float) -> float:
        return a / b if b else float("nan")

    sensitivity = div(tp, tp + fn)
    precision = div(tp, tp + fp)
    return {
        "threshold": float(threshold),
        "accuracy": div(tp + tn, len(y)),
        "sensitivity": sensitivity,
        "specificity": div(tn, tn + fp),
        "precision": precision,
        "npv": div(tn, tn + fn),
        "f1": div(2 * precision * sensitivity, precision + sensitivity),
        "auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) == 2 else float("nan"),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def youden_threshold(y: np.ndarray, prob: np.ndarray) -> float:
    """Threshold maximising sensitivity + specificity - 1. Choose it on validation data only."""
    fpr, tpr, thresholds = roc_curve(y, prob)
    best = int(np.argmax(tpr - fpr))
    return float(min(thresholds[best], 1.0))


def bootstrap_ci(
    y: np.ndarray,
    prob: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    keys: tuple[str, ...] = ("accuracy", "sensitivity", "specificity", "f1", "auc"),
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """95% percentile CIs, resampling whole patients so correlated images stay together."""
    y, prob, groups = np.asarray(y), np.asarray(prob), np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(unique), len(unique))])
        if len(np.unique(y[idx])) < 2:
            continue
        m = binary_metrics(y[idx], prob[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}
