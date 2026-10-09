import numpy as np
import pandas as pd
import pytest

from pneumonia.calibrate import Calibrator, ece
from pneumonia.ensemble import build_ensemble


def shifted_overconfident_logits(n=2000, seed=0):
    """Labels drawn from sigmoid(z); the 'model' reports 4*z - 2: overconfident and shifted."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return 4 * z - 2, y


def test_platt_and_temperature_reduce_ece():
    logits, y = shifted_overconfident_logits()
    cal = Calibrator(logits[:1000], y[:1000])
    test_logits, test_y = logits[1000:], y[1000:]
    raw = ece(cal.apply(test_logits, "raw"), test_y)
    assert ece(cal.apply(test_logits, "temperature"), test_y) < raw
    assert ece(cal.apply(test_logits, "platt"), test_y) < 0.05 < raw
    assert cal.a == pytest.approx(0.25, abs=0.05) and cal.b == pytest.approx(0.5, abs=0.15)


def write_run(path, logits, y, name):
    path.mkdir(parents=True)
    for split, sl in (("val", slice(0, 300)), ("test", slice(300, 600))):
        pd.DataFrame({"path": [f"{split}{i}" for i in range(300)], "label": y[sl], "patient_id": [f"p{i // 2}" for i in range(300)],
                      "logit": logits[sl]}).to_csv(path / f"{split}_predictions.csv", index=False)
    (path / "metrics.json").write_text(f'{{"model": "{name}", "params": 10, "train_minutes": 1.0}}')


def test_ensemble_runs_and_rejects_mismatched_splits(tmp_path):
    logits, y = shifted_overconfident_logits(600)
    write_run(tmp_path / "a", logits, y, "a")
    write_run(tmp_path / "b", logits + np.random.default_rng(1).normal(0, 1, 600), y, "b")
    r = build_ensemble([tmp_path / "a", tmp_path / "b"], tmp_path / "ens")
    assert 0.5 < r["test"]["auc"] <= 1 and r["params"] == 20
    assert len(pd.read_csv(tmp_path / "ens" / "test_predictions.csv")) == 300

    bad = pd.read_csv(tmp_path / "b" / "val_predictions.csv").iloc[::-1]
    bad.to_csv(tmp_path / "b" / "val_predictions.csv", index=False)
    with pytest.raises(ValueError, match="different images"):
        build_ensemble([tmp_path / "a", tmp_path / "b"], tmp_path / "ens2")
