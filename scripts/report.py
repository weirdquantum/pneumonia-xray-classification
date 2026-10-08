"""Collect every run under runs/ into a results table and comparison figures.

    python scripts/report.py

Outputs results/results.md, results/results.csv, figures/roc_curves.png and
figures/confusion_matrices.png.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve

ORDER = ["simple_cnn", "resnet50", "densenet121", "efficientnet_b0", "vit_b_16"]
LABELS = {
    "simple_cnn": "CNN (from scratch)",
    "resnet50": "ResNet50 (fine-tuned)",
    "densenet121": "DenseNet121 (fine-tuned)",
    "efficientnet_b0": "EfficientNet-B0 (fine-tuned)",
    "vit_b_16": "ViT-B/16 (fine-tuned)",
}


def fmt(value: float, ci: tuple[float, float] | None = None) -> str:
    s = f"{100 * value:.1f}"
    return f"{s} ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})" if ci else s


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    runs = {}
    for path in Path("runs").glob("*/metrics.json"):
        runs[path.parent.name] = json.loads(path.read_text())
    names = [n for n in ORDER if n in runs] + sorted(set(runs) - set(ORDER))
    if not names:
        raise SystemExit("no runs found under runs/")

    rows = []
    for n in names:
        r, t, ci = runs[n], runs[n]["test"], runs[n]["test_ci95"]
        cam_file = Path("runs") / n / "gradcam.json"
        cam = json.loads(cam_file.read_text())["border_share_mean"] if cam_file.exists() else float("nan")
        rows.append({
            "Model": LABELS.get(n, n),
            "Params (M)": f"{r['params'] / 1e6:.1f}",
            "AUC": fmt(t["auc"], ci["auc"]),
            "Accuracy": fmt(t["accuracy"], ci["accuracy"]),
            "Sensitivity": fmt(t["sensitivity"], ci["sensitivity"]),
            "Specificity": fmt(t["specificity"], ci["specificity"]),
            "F1": fmt(t["f1"], ci["f1"]),
            "Acc @0.5": fmt(r["test_at_0.5"]["accuracy"]),
            "Threshold": f"{r['val_threshold']:.2f}",
            "CAM border share": f"{cam:.2f}",
            "Train (min)": f"{r['train_minutes']:.0f}",
        })
    table = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    table.to_csv("results/results.csv", index=False)
    Path("results/results.md").write_text(to_markdown(table) + "\n")
    print(to_markdown(table))

    Path("figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in names:
        preds = pd.read_csv(Path("runs") / n / "test_predictions.csv")
        fpr, tpr, _ = roc_curve(preds["label"], preds["prob"])
        ax.plot(fpr, tpr, label=f"{LABELS.get(n, n)}  AUC={runs[n]['test']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", title="Test ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/roc_curves.png", dpi=130)

    fig, axes = plt.subplots(1, len(names), figsize=(3 * len(names), 3), squeeze=False)
    for ax, n in zip(axes[0], names):
        t = runs[n]["test"]
        cm = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["Normal", "Pneum."])
        ax.set_yticks([0, 1], ["Normal", "Pneum."])
        ax.set_xlabel("Predicted")
        ax.set_title(LABELS.get(n, n), fontsize=9)
    axes[0, 0].set_ylabel("True")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrices.png", dpi=130)
    print("saved results/ and figures/")


if __name__ == "__main__":
    main()
