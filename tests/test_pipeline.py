"""End-to-end smoke test: train -> evaluate on a tiny synthetic dataset, on CPU."""

import json

import numpy as np
import pandas as pd

from pneumonia.data import load_split
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
                "--batch-size", "4", "--image-size", "56", "--num-workers", "0", "--device", "cpu"])

    metrics = json.loads((out / "metrics.json").read_text())
    assert {"val", "test", "test_ci95", "val_threshold"} <= metrics.keys()
    for split in ("val", "test"):
        preds = pd.read_csv(out / f"{split}_predictions.csv")
        assert len(preds) == len(load_split(data, split, augment=False)[1])
        assert np.allclose(preds["prob"], 1 / (1 + np.exp(-preds["logit"])), atol=1e-6)
    assert (out / "best.pt").exists() and (out / "history.csv").exists()


def test_split_seed_redraws_val_but_never_moves_test(tmp_path):
    make_fake_data(tmp_path)
    (_, val_a), (_, val_b) = (load_split(tmp_path, "val", False, split_seed=s) for s in (1, 2))
    (_, test_a), (_, test_b) = (load_split(tmp_path, "test", False, split_seed=s) for s in (1, 2))
    assert set(val_a.index) != set(val_b.index)
    assert set(test_a.index) == set(test_b.index)
    train_a = load_split(tmp_path, "train", False, split_seed=1)[1]
    assert set(train_a["patient_id"]).isdisjoint(val_a["patient_id"])


def test_head_stage_keeps_backbone_batchnorm_frozen(tmp_path):
    import torch

    from pneumonia.models import build_model
    from pneumonia.train import train_one_epoch

    spec = build_model("resnet50", pretrained=False)
    for p in spec.model.parameters():
        p.requires_grad = False
    for p in spec.head.parameters():
        p.requires_grad = True
    before = spec.model.bn1.running_mean.clone()
    loader = [(torch.randn(4, 3, 64, 64) * 5 + 3, torch.tensor([0.0, 1.0, 0.0, 1.0]))]
    opt = torch.optim.AdamW(spec.head.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    train_one_epoch(spec.model, spec.head, True, loader, torch.nn.BCEWithLogitsLoss(), opt, None, scaler, torch.device("cpu"), False)
    assert torch.equal(spec.model.bn1.running_mean, before)
    train_one_epoch(spec.model, spec.head, False, loader, torch.nn.BCEWithLogitsLoss(), opt, None, scaler, torch.device("cpu"), False)
    assert not torch.equal(spec.model.bn1.running_mean, before)
