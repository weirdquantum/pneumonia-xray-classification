"""Train one model, select it on the validation split, then evaluate once on test.

    python -m pneumonia.train --model resnet50

Pretrained models are trained in two stages: a frozen-backbone warm-up of the
new head (``--head-epochs``), then full fine-tuning with a one-cycle schedule.
The test split is only touched after training, with the checkpoint and the
decision threshold both chosen on validation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit as sigmoid
from torch import nn
from torch.utils.data import DataLoader

from . import get_device, seed_everything
from .data import load_split
from .metrics import binary_metrics, bootstrap_ci, youden_threshold
from .models import ABLATION_MODEL_NAMES, MODEL_NAMES, build_model

# Per-model defaults; any of them can be overridden on the command line.
_CNN = dict(epochs=30, head_epochs=0, lr=1e-3, batch_size=32, patience=8)
DEFAULTS = {
    "simple_cnn": _CNN,
    "simple_cnn_nobn": _CNN,
    "simple_cnn_v1": _CNN,
    "resnet50": dict(epochs=12, head_epochs=2, lr=1e-4, batch_size=32, patience=5),
    "densenet121": dict(epochs=12, head_epochs=2, lr=1e-4, batch_size=32, patience=5),
    "efficientnet_b0": dict(epochs=12, head_epochs=2, lr=3e-4, batch_size=32, patience=5),
    "vit_b_16": dict(epochs=10, head_epochs=2, lr=3e-5, batch_size=32, patience=4),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_NAMES + ABLATION_MODEL_NAMES)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=None, help="run directory (default: runs/<model>)")
    p.add_argument("--epochs", type=int, help="total epochs including head warm-up")
    p.add_argument("--head-epochs", type=int)
    p.add_argument("--lr", type=float, help="fine-tuning learning rate")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--patience", type=int, help="early-stopping patience on validation loss")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-shuffle", action="store_true", help="ablation: feed training data in class-sorted order (the v1 bug)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision (only used on CUDA)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split-seed", type=int, default=42, help="seed of the patient-level train/val split")
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop each training epoch after N steps (evaluation stays full)")
    args = p.parse_args(argv)
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.out = Path(args.out or f"runs/{args.model}")
    return args


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device, drop_last: bool = False) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
    )


def autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logits, labels)`` for the whole loader."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        with autocast(device, amp):
            out = model(x.to(device, non_blocking=True)).squeeze(1)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def train_one_epoch(model, head, head_only, loader, criterion, optimizer, scheduler, scaler, device, amp, max_steps: int = 0) -> float:
    if head_only:
        model.eval()  # keep the frozen backbone's BatchNorm statistics and dropout fixed
        head.train()
    else:
        model.train()
    total, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(x).squeeze(1)
        loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def run_environment(device: torch.device) -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                                cwd=Path(__file__).parent).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else device.type,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": commit,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} device={device} amp={amp} out={args.out}")

    split_kw = dict(image_size=args.image_size, split_seed=args.split_seed)
    train_ds, train_rows = load_split(args.data_dir, "train", augment=not args.no_augment, **split_kw)
    val_ds, val_rows = load_split(args.data_dir, "val", augment=False, **split_kw)
    test_ds, test_rows = load_split(args.data_dir, "test", augment=False, **split_kw)
    train_loader = make_loader(train_ds, args.batch_size, not args.no_shuffle, args.num_workers, device, drop_last=True)
    val_loader = make_loader(val_ds, args.batch_size * 2, False, args.num_workers, device)
    test_loader = make_loader(test_ds, args.batch_size * 2, False, args.num_workers, device)

    n_pos = int(train_rows["label"].sum())
    n_neg = len(train_rows) - n_pos
    print(f"train={len(train_ds)} (normal {n_neg} / pneumonia {n_pos})  val={len(val_ds)}  test={len(test_ds)}")
    pos_weight = 1.0 if args.no_class_weight else n_neg / n_pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    spec = build_model(args.model, image_size=args.image_size)
    model = spec.model.to(device)
    head_epochs = args.head_epochs if spec.pretrained else 0
    n_params = sum(p.numel() for p in model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    def val_loss_of(logits: np.ndarray, labels: np.ndarray) -> float:
        with torch.no_grad():
            return criterion(torch.from_numpy(logits).to(device), torch.from_numpy(labels).to(device)).item()

    # Checkpoints are selected on validation loss: val AUC saturates near 0.999 and cannot rank epochs.
    history, best_loss, best_epoch, stale = [], np.inf, -1, 0
    start = time.time()
    for epoch in range(args.epochs):
        if epoch == 0 and head_epochs > 0:
            for p in model.parameters():
                p.requires_grad = False
            for p in spec.head.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(spec.head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
            scheduler, stage = None, "head"
        if epoch == head_epochs:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            steps = (args.epochs - head_epochs) * (args.max_steps or len(train_loader))
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=steps, pct_start=0.1)
            stage = "finetune"

        t0 = time.time()
        train_loss = train_one_epoch(model, spec.head, stage == "head", train_loader, criterion, optimizer, scheduler,
                                     scaler, device, amp, args.max_steps)
        val_logit, val_y = predict(model, val_loader, device, amp)
        val_loss = val_loss_of(val_logit, val_y)
        vm = binary_metrics(val_y, sigmoid(val_logit))
        history.append(dict(epoch=epoch + 1, stage=stage, train_loss=train_loss, val_loss=val_loss, val_auc=vm["auc"],
                            val_acc=vm["accuracy"], val_sens=vm["sensitivity"], val_spec=vm["specificity"], seconds=time.time() - t0))
        print(f"epoch {epoch + 1:2d}/{args.epochs} [{stage}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={vm['auc']:.4f} val_sens={vm['sensitivity']:.3f} val_spec={vm['specificity']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), args.out / "best.pt")
        elif stage == "finetune":
            stale += 1
            if stale >= args.patience:
                print(f"early stopping: no val loss improvement for {args.patience} epochs")
                break
    train_seconds = time.time() - start
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)

    # Final evaluation: best checkpoint, threshold picked on validation, single pass over test.
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device))
    val_logit, val_y = predict(model, val_loader, device, amp)
    test_logit, test_y = predict(model, test_loader, device, amp)
    val_prob, test_prob = sigmoid(val_logit), sigmoid(test_logit)
    threshold = youden_threshold(val_y, val_prob)

    results = {
        "model": args.model,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": run_environment(device),
        "params": n_params,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train_minutes": train_seconds / 60,
        "val_loss": val_loss_of(val_logit, val_y),
        "val_threshold": threshold,
        "val": binary_metrics(val_y, val_prob, threshold),
        "test_at_0.5": binary_metrics(test_y, test_prob, 0.5),
        "test": binary_metrics(test_y, test_prob, threshold),
        "test_ci95": bootstrap_ci(test_y, test_prob, test_rows["patient_id"].to_numpy(), threshold),
    }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    # Raw logits are kept so runs can be calibrated or ensembled later without re-running inference.
    val_rows.assign(logit=val_logit, prob=val_prob).to_csv(args.out / "val_predictions.csv", index=False)
    test_rows.assign(logit=test_logit, prob=test_prob).to_csv(args.out / "test_predictions.csv", index=False)

    t = results["test"]
    print(f"TEST (threshold {threshold:.3f} from val): acc={t['accuracy']:.4f} auc={t['auc']:.4f} "
          f"sens={t['sensitivity']:.4f} spec={t['specificity']:.4f} f1={t['f1']:.4f}")


if __name__ == "__main__":
    main()
