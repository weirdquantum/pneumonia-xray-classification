"""End-to-end smoke test: train -> evaluate on a tiny synthetic dataset, on CPU."""

import json

import numpy as np
import pandas as pd

from pneumonia.train import main as train_main


def make_fake_data(root, n_per_split=24, size=64):
    rng = np.random.default_rng(0)
    rows, images = [], []
    for split in ("train", "val", "test"):
        for i in range(n_per_split):
            label = i % 2
            rows.append({"path": f"{split}/{i}.jpeg", "split": split, "label": label, "patient_id": f"{split}{i // 2}"})
            img = rng.integers(0, 80, (size, size), dtype=np.uint8)
            if label:
                img[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] += 120  # learnable signal
            images.append(img)
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)
    np.save(root / "images.npy", np.stack(images))


def test_train_writes_predictions_and_metrics(tmp_path):
    data, out = tmp_path / "data", tmp_path / "run"
    data.mkdir()
    make_fake_data(data)
    train_main(["--model", "simple_cnn", "--data-dir", str(data), "--out", str(out), "--epochs", "2",
                "--batch-size", "8", "--image-size", "56", "--num-workers", "0", "--device", "cpu"])

    metrics = json.loads((out / "metrics.json").read_text())
    assert {"val", "test", "test_ci95", "val_threshold"} <= metrics.keys()
    for split, n in (("val", 24), ("test", 24)):
        preds = pd.read_csv(out / f"{split}_predictions.csv")
        assert len(preds) == n
        assert np.allclose(preds["prob"], 1 / (1 + np.exp(-preds["logit"])), atol=1e-6)
    assert (out / "best.pt").exists() and (out / "history.csv").exists()
