import numpy as np

from pneumonia.metrics import binary_metrics, bootstrap_ci, youden_threshold


def test_binary_metrics_known_confusion():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 3)
    assert m["sensitivity"] == 0.75
    assert m["specificity"] == 0.75
    assert m["accuracy"] == 0.75
    assert m["auc"] == 13.5 / 16  # 16 pos-neg pairs; the 0.2 vs 0.2 tie counts as half


def test_youden_threshold_separates_perfectly_separable_data():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    t = youden_threshold(y, p)
    assert binary_metrics(y, p, t)["accuracy"] == 1.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    groups = np.arange(300) // 2
    point = binary_metrics(y, p, 0.5)
    ci = bootstrap_ci(y, p, groups, 0.5, n_boot=200)
    for k in ("accuracy", "auc", "sensitivity"):
        assert ci[k][0] <= point[k] <= ci[k][1]
