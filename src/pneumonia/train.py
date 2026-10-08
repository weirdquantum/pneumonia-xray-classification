"""Train one model, select it on the validation split, then evaluate once on test.

    python -m pneumonia.train --model resnet50

Pretrained models are trained in two stages: a frozen-backbone warm-up of the
new head (``--head-epochs``), then full fine-tuning with a cosine schedule.
The test split is only touched after training, with the checkpoint and the
decision threshold both chosen on validation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from . import get_device, seed_everything
from .data import load_split
from .metrics import binary_metrics, bootstrap_ci, youden_threshold
from .models import MODEL_NAMES, build_model

# Per-model defaults; any of them can be overridden on the command line.
DEFAULTS = {
    "simple_cnn": dict(epochs=30, head_epochs=0, lr=1e-3, batch_size=32, patience=8),
    "resnet50": dict(epochs=12, head_epochs=2, lr=1e-4, batch_size=32, patience=5),
    "densenet121": dict(epochs=12, head_epochs=2, lr=1e-4, batch_size=32, patience=5),
    "efficientnet_b0": dict(epochs=12, head_epochs=2, lr=3e-4, batch_size=32, patience=5),
    "vit_b_16": dict(epochs=10, head_epochs=2, lr=3e-5, batch_size=32, patience=4),
}


class FocalLoss(nn.Module):
    """Binary focal loss; ``alpha`` weights the positive class."""

    def __init__(self, alpha: float, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODEL_NAMES)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=None, help="run directory (default: runs/<model>)")
    p.add_argument("--epochs", type=int, help="total epochs including head warm-up")
    p.add_argument("--head-epochs", type=int)
    p.add_argument("--lr", type=float, help="fine-tuning learning rate")
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--patience", type=int, help="early-stopping patience on validation AUC")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=("bce", "focal"), default="bce")
    p.add_argument("--no-class-weight", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--from-scratch", action="store_true", help="ignore ImageNet weights")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    args = p.parse_args(argv)
    for key, value in DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.out = Path(args.out or f"runs/{args.model}")
    return args


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
        drop_last=shuffle,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device, max_steps: int = 0) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, max_steps: int = 0) -> float:
    model.train()
    total, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x).squeeze(1), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)
    device = get_device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model} device={device} out={args.out}")

    train_ds, train_rows = load_split(args.data_dir, "train", augment=not args.no_augment, image_size=args.image_size)
    val_ds, val_rows = load_split(args.data_dir, "val", augment=False, image_size=args.image_size)
    test_ds, test_rows = load_split(args.data_dir, "test", augment=False, image_size=args.image_size)
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, device)
    val_loader = make_loader(val_ds, args.batch_size * 2, False, args.num_workers, device)
    test_loader = make_loader(test_ds, args.batch_size * 2, False, args.num_workers, device)

    n_pos = int(train_rows["label"].sum())
    n_neg = len(train_rows) - n_pos
    print(f"train={len(train_ds)} (normal {n_neg} / pneumonia {n_pos})  val={len(val_ds)}  test={len(test_ds)}")
    if args.loss == "focal":
        criterion = FocalLoss(alpha=0.5 if args.no_class_weight else n_neg / (n_neg + n_pos))
    else:
        pos_weight = 1.0 if args.no_class_weight else n_neg / n_pos
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    spec = build_model(args.model, pretrained=not args.from_scratch)
    model = spec.model.to(device)
    head_epochs = args.head_epochs if spec.pretrained else 0
    n_params = sum(p.numel() for p in model.parameters())

    history, best_score, best_epoch, stale = [], (-np.inf, -np.inf), -1, 0
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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.max_steps)
        val_prob, val_y = predict(model, val_loader, device, args.max_steps)
        val_loss = criterion(torch.logit(torch.tensor(val_prob).clamp(1e-6, 1 - 1e-6)).to(device), torch.tensor(val_y).to(device)).item()
        vm = binary_metrics(val_y, val_prob)
        history.append(dict(epoch=epoch + 1, stage=stage, train_loss=train_loss, val_loss=val_loss, val_auc=vm["auc"],
                            val_acc=vm["accuracy"], val_sens=vm["sensitivity"], val_spec=vm["specificity"], seconds=time.time() - t0))
        print(f"epoch {epoch + 1:2d}/{args.epochs} [{stage}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_auc={vm['auc']:.4f} val_sens={vm['sensitivity']:.3f} val_spec={vm['specificity']:.3f} ({time.time() - t0:.0f}s)", flush=True)

        score = (round(vm["auc"], 4), -val_loss)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch + 1, 0
            torch.save(model.state_dict(), args.out / "best.pt")
        elif stage == "finetune":
            stale += 1
            if stale >= args.patience:
                print(f"early stopping: no val AUC improvement for {args.patience} epochs")
                break
    train_seconds = time.time() - start
    pd.DataFrame(history).to_csv(args.out / "history.csv", index=False)

    # Final evaluation: best checkpoint, threshold picked on validation, single pass over test.
    model.load_state_dict(torch.load(args.out / "best.pt", map_location=device))
    val_prob, val_y = predict(model, val_loader, device, args.max_steps)
    test_prob, test_y = predict(model, test_loader, device, args.max_steps)
    test_rows = test_rows.iloc[: len(test_prob)]
    threshold = youden_threshold(val_y, val_prob)

    results = {
        "model": args.model,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "params": n_params,
        "best_epoch": best_epoch,
        "train_minutes": train_seconds / 60,
        "val_threshold": threshold,
        "val": binary_metrics(val_y, val_prob, threshold),
        "test_at_0.5": binary_metrics(test_y, test_prob, 0.5),
        "test": binary_metrics(test_y, test_prob, threshold),
        "test_ci95": bootstrap_ci(test_y, test_prob, test_rows["patient_id"].to_numpy(), threshold),
    }
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    test_rows.assign(prob=test_prob).to_csv(args.out / "test_predictions.csv", index=False)

    t = results["test"]
    print(f"TEST (threshold {threshold:.3f} from val): acc={t['accuracy']:.4f} auc={t['auc']:.4f} "
          f"sens={t['sensitivity']:.4f} spec={t['specificity']:.4f} f1={t['f1']:.4f}")


if __name__ == "__main__":
    main()
