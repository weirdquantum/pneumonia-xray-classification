"""Turn finished runs into tables and figures.

    python scripts/report.py --runs-root runs --out .

Reads runs/main/<model>/seed<k>, runs/ensemble/seed<k> and runs/ablation/<name>,
and writes:
    results/main.md         mean ± std over seeds for every model and the ensemble
    results/first_seed.md   first-seed runs with patient-level bootstrap 95% CIs
    results/calibration.md  test calibration before/after temperature and Platt scaling
    results/ablation.md     CNN ablations and the ResNet50 linear probe vs. the full CNN
    results/all_runs.csv    one row per run
    figures/roc_curves.png, confusion_matrices.png, reliability.png, ablation.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from pneumonia.calibrate import reliability_figure

ORDER = ["simple_cnn", "resnet50", "densenet121", "efficientnet_b0", "vit_b_16", "ensemble"]
LABELS = {
    "simple_cnn": "CNN (from scratch)",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNet-B0",
    "vit_b_16": "ViT-B/16",
}


def label(row) -> str:
    if row["model"] == "ensemble":
        return f"Ensemble ({row['n_members']} models)"
    return LABELS.get(row["model"], row["model"])
METRICS = ["auc", "accuracy", "sensitivity", "specificity", "f1"]


def to_markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def mean_sd(values) -> str:
    v = 100 * np.asarray(values, dtype=float)
    return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def load_runs(runs_root: Path) -> pd.DataFrame:
    rows = []
    for kind, pattern in (("main", "main/*/seed*"), ("ensemble", "ensemble/seed*"), ("ablation", "ablation/*")):
        for run in sorted(runs_root.glob(pattern)):
            if not (run / "metrics.json").exists():
                continue
            m = json.loads((run / "metrics.json").read_text())
            row = {"kind": kind, "run": str(run.relative_to(runs_root)), "path": run, "model": m["model"],
                   "seed": int(run.name[4:]) if run.name.startswith("seed") else None,
                   "name": run.name if kind == "ablation" else m["model"],
                   "params": m["params"], "threshold": m["val_threshold"], "train_minutes": m["train_minutes"],
                   "best_epoch": m.get("best_epoch"), "epochs_run": m.get("epochs_run"),
                   "acc_at_0.5": m["test_at_0.5"]["accuracy"], "n_members": len(m.get("members", []))}
            row.update({k: m["test"][k] for k in METRICS + ["tp", "tn", "fp", "fn"]})
            row.update({f"{k}_ci": tuple(m["test_ci95"][k]) for k in METRICS})
            for extra, key in (("gradcam.json", "border_share_mean"), ("ablation.json", "description")):
                if (run / extra).exists():
                    row[key] = json.loads((run / extra).read_text())[key]
            if (run / "calibration.json").exists():
                cal = json.loads((run / "calibration.json").read_text())["test"]
                for method in ("raw", "temperature", "platt"):
                    row[f"ece_{method}"] = cal[method]["ece"]
                    row[f"brier_{method}"] = cal[method]["brier"]
                row["saturated_raw"] = cal["raw"]["saturated"]
            rows.append(row)
    return pd.DataFrame(rows)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(_o=df["model"].map({m: i for i, m in enumerate(ORDER)}).fillna(99)).sort_values(["_o", "seed"]).drop(columns="_o")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--first-seed", type=int, default=42)
    args = p.parse_args()
    runs = load_runs(args.runs_root)
    if runs.empty:
        raise SystemExit(f"no finished runs under {args.runs_root}")
    results, figures = args.out / "results", args.out / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    runs.drop(columns="path").to_csv(results / "all_runs.csv", index=False)
    tables = {}

    models = ordered(runs[runs["kind"].isin(["main", "ensemble"])])
    if not models.empty:
        rows = []
        for model, g in models.groupby("model", sort=False):
            rows.append({"Model": label(g.iloc[0]), "Seeds": len(g),
                         **{k.upper() if k == "auc" else k.capitalize(): mean_sd(g[k]) for k in METRICS},
                         "Threshold": f"{g['threshold'].mean():.2f} ± {g['threshold'].std(ddof=1):.2f}" if len(g) > 1 else f"{g['threshold'].mean():.2f}",
                         "Train (min)": f"{g['train_minutes'].mean():.0f}"})
        tables["main"] = pd.DataFrame(rows)

        first = models[models["seed"] == args.first_seed]
        rows = []
        for _, r in first.iterrows():
            ci = lambda k: f"{pct(r[k])} ({pct(r[k + '_ci'][0])}–{pct(r[k + '_ci'][1])})"  # noqa: E731
            rows.append({"Model": label(r), "Params (M)": f"{r['params'] / 1e6:.1f}",
                         **{k.upper() if k == "auc" else k.capitalize(): ci(k) for k in METRICS},
                         "Acc @0.5": pct(r["acc_at_0.5"]), "Threshold": f"{r['threshold']:.2f}",
                         "CAM border share": f"{r['border_share_mean']:.2f}" if pd.notna(r.get("border_share_mean")) else "–"})
        tables["first_seed"] = pd.DataFrame(rows)

        if "ece_raw" in models:
            rows = []
            for model, g in models.groupby("model", sort=False):
                rows.append({"Model": label(g.iloc[0]), "Saturated (raw)": mean_sd(g["saturated_raw"]),
                             "ECE raw": f"{g['ece_raw'].mean():.3f}", "ECE temperature": f"{g['ece_temperature'].mean():.3f}",
                             "ECE Platt": f"{g['ece_platt'].mean():.3f}", "Brier raw": f"{g['brier_raw'].mean():.3f}",
                             "Brier Platt": f"{g['brier_platt'].mean():.3f}"})
            tables["calibration"] = pd.DataFrame(rows)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        for _, r in first.iterrows():
            preds = pd.read_csv(r["path"] / "test_predictions.csv")
            fpr, tpr, _ = roc_curve(preds["label"], preds["prob"])
            ax.plot(fpr, tpr, lw=2.2 if r["model"] == "ensemble" else 1.2, label=f"{label(r)}  AUC={r['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set(xlabel="1 − Specificity", ylabel="Sensitivity", title=f"Test ROC curves (seed {args.first_seed})")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(figures / "roc_curves.png", dpi=130)

        fig, axes = plt.subplots(1, len(first), figsize=(3 * len(first), 3), squeeze=False)
        for ax, (_, r) in zip(axes[0], first.iterrows()):
            cm = [[r["tn"], r["fp"]], [r["fn"], r["tp"]]]
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
            ax.set_xticks([0, 1], ["Normal", "Pneum."])
            ax.set_yticks([0, 1], ["Normal", "Pneum."])
            ax.set_xlabel("Predicted")
            ax.set_title(label(r), fontsize=9)
        axes[0, 0].set_ylabel("True")
        fig.tight_layout()
        fig.savefig(figures / "confusion_matrices.png", dpi=130)

        with_preds = [r for _, r in first.iterrows() if (r["path"] / "val_predictions.csv").exists()]
        if with_preds:
            reliability_figure([r["path"] for r in with_preds], [label(r) for r in with_preds],
                               figures / "reliability.png")

    ablation = runs[runs["kind"] == "ablation"]
    reference = runs[(runs["kind"] == "main") & (runs["model"] == "simple_cnn") & (runs["seed"] == args.first_seed)]
    if not ablation.empty and not reference.empty:
        ref = reference.iloc[0]
        rows = [{"Variant": "full (reference)", "Description": "v2 CNN: shuffle, augment, class weight, BatchNorm, 224px, 30 epochs",
                 "AUC": pct(ref["auc"]), "Accuracy": pct(ref["accuracy"]), "Sensitivity": pct(ref["sensitivity"]),
                 "Specificity": pct(ref["specificity"]), "Acc @0.5": pct(ref["acc_at_0.5"]), "Δ AUC": "–", "Δ Accuracy": "–"}]
        for _, r in ablation.iterrows():
            rows.append({"Variant": r["name"], "Description": r.get("description", ""), "AUC": pct(r["auc"]),
                         "Accuracy": pct(r["accuracy"]), "Sensitivity": pct(r["sensitivity"]), "Specificity": pct(r["specificity"]),
                         "Acc @0.5": pct(r["acc_at_0.5"]), "Δ AUC": f"{100 * (r['auc'] - ref['auc']):+.1f}",
                         "Δ Accuracy": f"{100 * (r['accuracy'] - ref['accuracy']):+.1f}"})
        tables["ablation"] = pd.DataFrame(rows)

        names = ["full (reference)"] + list(ablation["name"])
        aucs = [ref["auc"]] + list(ablation["auc"])
        accs = [ref["accuracy"]] + list(ablation["accuracy"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(names) + 1.2), sharey=True)
        for ax, vals, title in ((axes[0], aucs, "Test AUC"), (axes[1], accs, "Test accuracy (val threshold)")):
            ax.barh(names, [100 * v for v in vals], color=["#555"] + ["#4C78A8"] * (len(names) - 1))
            ax.axvline(100 * vals[0], color="#555", ls="--", lw=0.8)
            ax.set(title=title, xlim=(min(100 * min(vals) - 3, 90), 100))
            for i, v in enumerate(vals):
                ax.text(100 * v + 0.2, i, f"{100 * v:.1f}", va="center", fontsize=8)
        axes[0].invert_yaxis()
        fig.tight_layout()
        fig.savefig(figures / "ablation.png", dpi=130)

    for name, table in tables.items():
        (results / f"{name}.md").write_text(to_markdown(table))
        print(f"\n## {name}\n{to_markdown(table)}")
    print(f"saved {results}/ and {figures}/")


if __name__ == "__main__":
    main()
