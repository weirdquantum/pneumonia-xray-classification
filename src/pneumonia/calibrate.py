"""Post-hoc probability calibration of trained runs, fitted on validation and scored on test.

    python -m pneumonia.calibrate runs/main/*/seed42

Two methods are compared against the raw model output:
  * temperature scaling  p = sigmoid(z / T)       (rescales confidence only)
  * Platt scaling        p = sigmoid(a * z + b)   (also removes the shift that the
                                                   class-weighted training loss puts on every logit)
Each run gets a ``calibration.json``. Ranking metrics (AUC) are unchanged by
either method because both are monotone in the logit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def nll(prob: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(prob, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(prob: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((prob - y) ** 2))


def ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on p(pneumonia)."""
    bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(prob[mask].mean() - y[mask].mean())
    return float(total)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.exp(np.linspace(np.log(0.05), np.log(50), 600))
    losses = [nll(sigmoid(logits / t), y) for t in grid]
    return float(grid[int(np.argmin(losses))])


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logits.reshape(-1, 1), y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


class Calibrator:
    """Platt and temperature calibrators fitted on one run's validation logits."""

    def __init__(self, val_logits: np.ndarray, val_y: np.ndarray):
        self.temperature = fit_temperature(val_logits, val_y)
        self.a, self.b = fit_platt(val_logits, val_y)

    def apply(self, logits: np.ndarray, method: str) -> np.ndarray:
        if method == "raw":
            return sigmoid(logits)
        if method == "temperature":
            return sigmoid(logits / self.temperature)
        if method == "platt":
            return sigmoid(self.a * logits + self.b)
        raise ValueError(method)


METHODS = ("raw", "temperature", "platt")


def scores(prob: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {"ece": ece(prob, y), "brier": brier(prob, y), "nll": nll(prob, y),
            "saturated": float(((prob < 0.01) | (prob > 0.99)).mean())}


def load_predictions(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(run / "val_predictions.csv"), pd.read_csv(run / "test_predictions.csv")


def calibrate_run(run: Path) -> dict:
    val, test = load_predictions(run)
    cal = Calibrator(val["logit"].to_numpy(), val["label"].to_numpy())
    result = {"temperature": cal.temperature, "platt": {"a": cal.a, "b": cal.b}}
    for split, df in (("val", val), ("test", test)):
        logits, y = df["logit"].to_numpy(), df["label"].to_numpy()
        result[split] = {m: scores(cal.apply(logits, m), y) for m in METHODS}
    (run / "calibration.json").write_text(json.dumps(result, indent=2))
    return result


def reliability_figure(runs: list[Path], labels: list[str], out: Path, n_bins: int = 10) -> None:
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs), 3.4), squeeze=False, sharey=True)
    for ax, run, label in zip(axes[0], runs, labels, strict=True):
        val, test = load_predictions(run)
        cal = Calibrator(val["logit"].to_numpy(), val["label"].to_numpy())
        y = test["label"].to_numpy()
        for method, style in (("raw", "o-"), ("platt", "s-")):
            prob = cal.apply(test["logit"].to_numpy(), method)
            bins = np.minimum((prob * n_bins).astype(int), n_bins - 1)
            xs = [prob[bins == b].mean() for b in range(n_bins) if (bins == b).any()]
            ys = [y[bins == b].mean() for b in range(n_bins) if (bins == b).any()]
            ax.plot(xs, ys, style, ms=4, label=f"{method} (ECE {ece(prob, y):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(title=label, xlabel="Predicted p(pneumonia)", xlim=(0, 1), ylim=(0, 1))
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7, loc="upper left")
    axes[0, 0].set_ylabel("Observed pneumonia rate (test)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args(argv)
    for run in args.runs:
        r = calibrate_run(run)
        t = r["test"]
        print(f"{run}: test ECE raw {t['raw']['ece']:.3f} -> temperature {t['temperature']['ece']:.3f} -> platt {t['platt']['ece']:.3f}")


if __name__ == "__main__":
    main()
