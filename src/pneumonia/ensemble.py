"""Average the calibrated predictions of several runs into one ensemble "run".

    python -m pneumonia.ensemble --out runs/ensemble/seed42 runs/main/*/seed42

Each member is Platt-calibrated on its own validation logits first, so models
with different logit scales contribute equally. The ensemble then picks its
threshold on validation and is scored on test exactly like a single model; its
output directory has the same files (metrics.json, val/test_predictions.csv).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

from .calibrate import Calibrator, load_predictions
from .metrics import binary_metrics, bootstrap_ci, youden_threshold


def build_ensemble(runs: list[Path], out: Path) -> dict:
    members = [load_predictions(r) for r in runs]
    val0, test0 = members[0]
    for run, (val, test) in zip(runs, members, strict=True):
        if not (val["path"].equals(val0["path"]) and test["path"].equals(test0["path"])):
            raise ValueError(f"{run} was evaluated on different images; ensemble only runs that share a split seed")

    val_probs, test_probs = [], []
    for val, test in members:
        cal = Calibrator(val["logit"].to_numpy(), val["label"].to_numpy())
        val_probs.append(cal.apply(val["logit"].to_numpy(), "platt"))
        test_probs.append(cal.apply(test["logit"].to_numpy(), "platt"))
    val_prob, test_prob = np.mean(val_probs, axis=0), np.mean(test_probs, axis=0)
    val_y, test_y = val0["label"].to_numpy(), test0["label"].to_numpy()
    threshold = youden_threshold(val_y, val_prob)

    metrics = [json.loads((r / "metrics.json").read_text()) for r in runs]
    results = {
        "model": "ensemble",
        "members": [str(r) for r in runs],
        "params": sum(m["params"] for m in metrics),
        "train_minutes": sum(m["train_minutes"] for m in metrics),
        "val_threshold": threshold,
        "val": binary_metrics(val_y, val_prob, threshold),
        "test_at_0.5": binary_metrics(test_y, test_prob, 0.5),
        "test": binary_metrics(test_y, test_prob, threshold),
        "test_ci95": bootstrap_ci(test_y, test_prob, test0["patient_id"].to_numpy(), threshold),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    clip = lambda p: np.clip(p, 1e-7, 1 - 1e-7)  # noqa: E731
    val0.assign(logit=logit(clip(val_prob)), prob=val_prob).to_csv(out / "val_predictions.csv", index=False)
    test0.assign(logit=logit(clip(test_prob)), prob=test_prob).to_csv(out / "test_predictions.csv", index=False)
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    t = build_ensemble(args.runs, args.out)["test"]
    print(f"ensemble of {len(args.runs)}: acc={t['accuracy']:.4f} auc={t['auc']:.4f} "
          f"sens={t['sensitivity']:.4f} spec={t['specificity']:.4f}")


if __name__ == "__main__":
    main()
