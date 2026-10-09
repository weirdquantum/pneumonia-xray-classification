"""Grad-CAM heatmaps for a trained run, plus a simple shortcut-learning check.

    python -m pneumonia.gradcam --run runs/resnet50

Writes ``figures/gradcam_<model>.png`` (most confident TP / TN / FP / FN test
cases) and ``<run>/gradcam.json`` with the share of heatmap mass that falls in
the image border. Lungs sit in the centre of a chest film, so a border share
well above the uniform baseline suggests the model is reading text markers,
borders or collimation edges instead of lung tissue.
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
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from . import get_device
from .data import IMAGENET_MEAN, IMAGENET_STD, load_split
from .models import build_model

BORDER = 0.125  # outer fraction of width/height on each side counted as "border"


class GradCAM:
    def __init__(self, model: nn.Module, layer: nn.Module, reshape=None):
        self.model, self.reshape = model, reshape
        self._acts = self._grads = None
        layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._acts = output
        output.register_hook(lambda grad: setattr(self, "_grads", grad))

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cams, probs)``; each CAM explains the predicted class and is scaled to [0, 1]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x).squeeze(1)
        sign = torch.where(logits >= 0, 1.0, -1.0)  # explain "normal" predictions with the negated logit
        (logits * sign).sum().backward()
        acts, grads = self._acts, self._grads
        if self.reshape is not None:
            acts, grads = self.reshape(acts), self.reshape(grads)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        flat = cam.flatten(1)
        cam = (cam - flat.min(1).values[:, None, None]) / (flat.max(1).values - flat.min(1).values + 1e-8)[:, None, None]
        return cam.detach(), torch.sigmoid(logits).detach()


def border_share(cams: torch.Tensor) -> torch.Tensor:
    """Share of each heatmap's mass in the outer border; NaN for an all-zero heatmap."""
    h, w = cams.shape[-2:]
    bh, bw = int(h * BORDER), int(w * BORDER)
    total = cams.flatten(1).sum(1)
    centre = cams[:, bh : h - bh, bw : w - bw].flatten(1).sum(1)
    return torch.where(total > 0, 1 - centre / total.clamp_min(1e-12), torch.nan)


def to_gray(x: torch.Tensor) -> np.ndarray:
    return (x.cpu() * IMAGENET_STD + IMAGENET_MEAN)[0].clamp(0, 1).numpy()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--figures", default="figures", type=Path)
    p.add_argument("--n", type=int, default=4, help="examples per outcome")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    results = json.loads((args.run / "metrics.json").read_text())
    name, threshold = results["model"], results["val_threshold"]
    device = get_device(args.device)
    spec = build_model(name, pretrained=False)
    spec.model.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu"))
    model = spec.model.to(device).eval()
    cam = GradCAM(model, spec.cam_layer, spec.cam_reshape)

    test_ds, _ = load_split(args.data_dir, "test", augment=False, image_size=results["config"]["image_size"])
    preds = pd.read_csv(args.run / "test_predictions.csv")

    shares = []
    for x, _ in DataLoader(test_ds, batch_size=16):
        cams, _ = cam(x.to(device))
        shares.append(border_share(cams).cpu())
    shares = torch.cat(shares).numpy()
    uniform = 1 - (1 - 2 * BORDER) ** 2
    summary = {
        "border_share_mean": float(np.nanmean(shares)),
        "border_share_uniform_baseline": uniform,
        "empty_heatmaps": int(np.isnan(shares).sum()),
    }
    (args.run / "gradcam.json").write_text(json.dumps(summary, indent=2))
    print(f"{name}: mean Grad-CAM border share {summary['border_share_mean']:.3f} (uniform heatmap would be {uniform:.3f}; "
          f"{summary['empty_heatmaps']} empty heatmaps excluded)")

    pred = preds["prob"] >= threshold
    label = preds["label"] == 1
    outcomes = {
        "True positive": preds[pred & label].nlargest(args.n, "prob"),
        "True negative": preds[~pred & ~label].nsmallest(args.n, "prob"),
        "False positive": preds[pred & ~label].nlargest(args.n, "prob"),
        "False negative": preds[~pred & label].nsmallest(args.n, "prob"),
    }
    fig, axes = plt.subplots(len(outcomes), args.n, figsize=(2.6 * args.n, 2.9 * len(outcomes)), squeeze=False)
    for row, (title, rows) in enumerate(outcomes.items()):
        for col in range(args.n):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(rows):
                continue
            i = preds.index.get_loc(rows.index[col])
            x, _ = test_ds[i]
            heat, prob = cam(x.unsqueeze(0).to(device))
            ax.imshow(to_gray(x), cmap="gray")
            ax.imshow(heat[0].cpu().numpy(), cmap="jet", alpha=0.35)
            ax.set_title(f"p(pneumonia)={prob.item():.2f}", fontsize=8)
        axes[row, 0].text(-0.08, 0.5, title, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right", fontsize=10)
    fig.suptitle(f"Grad-CAM — {name} (threshold {threshold:.2f})")
    fig.tight_layout()
    args.figures.mkdir(exist_ok=True)
    fig.savefig(args.figures / f"gradcam_{name}.png", dpi=120)
    print(f"saved {args.figures / f'gradcam_{name}.png'}")


if __name__ == "__main__":
    main()
