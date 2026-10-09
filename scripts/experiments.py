"""Run every experiment end to end, resumably.

    python scripts/experiments.py --runs-root runs            # full plan
    python scripts/experiments.py --plan smoke --runs-root /tmp/smoke

Steps (each finished run is skipped on the next invocation, so a disconnected
Colab session can simply re-run this command):
  1. main      every model x every seed; each seed also re-draws the patient-level val split
  2. ablation  one-factor-at-a-time changes to the from-scratch CNN, plus a ResNet50 linear probe
  3. gradcam   heatmaps for the first seed's main runs
  4. ensemble  Platt-calibrated average of the main models, per seed
  5. calibrate calibration.json for every main and ensemble run
  6. report    tables in results/, figures in figures/

Runs train in --work-dir (fast local disk) and are copied to --runs-root when
complete. Model weights are kept only for the first seed's main runs (needed
for Grad-CAM); everything else keeps metrics and predictions only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_MODELS = ["simple_cnn", "resnet50", "efficientnet_b0", "densenet121", "vit_b_16"]

# name -> (description, train arguments). The reference point is the main simple_cnn run at the first seed.
ABLATIONS = {
    "no_augment": ("CNN without data augmentation", ["--model", "simple_cnn", "--no-augment"]),
    "no_class_weight": ("CNN without class-weighted loss", ["--model", "simple_cnn", "--no-class-weight"]),
    "no_batchnorm": ("CNN without BatchNorm", ["--model", "simple_cnn_nobn"]),
    "low_res_112": ("CNN at 112x112 input", ["--model", "simple_cnn", "--image-size", "112"]),
    "epochs_5": ("CNN trained for 5 epochs", ["--model", "simple_cnn", "--epochs", "5"]),
    "no_shuffle": ("CNN with class-sorted (unshuffled) batches, the v1 bug", ["--model", "simple_cnn", "--no-shuffle"]),
    "v1_like": ("v1 recipe: v1 CNN, 100px, no augment/class weight/shuffle, 5 epochs, batch 10",
                ["--model", "simple_cnn_v1", "--image-size", "100", "--no-augment", "--no-class-weight",
                 "--no-shuffle", "--epochs", "5", "--batch-size", "10"]),
    "resnet50_linear_probe": ("ResNet50 with frozen ImageNet features, head only (v1-style transfer)",
                              ["--model", "resnet50", "--epochs", "5", "--head-epochs", "5"]),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(*cmd: str) -> bool:
    log(" ".join(cmd))
    returncode = subprocess.run(cmd, cwd=ROOT).returncode
    if returncode != 0:
        log(f"FAILED (exit {returncode}): {' '.join(cmd)}")
    return returncode == 0


def train(name: str, args: list[str], dest: Path, opts, keep_weights: bool) -> bool:
    if (dest / "metrics.json").exists():
        log(f"skip {name}: already done")
        return True
    work = opts.work_dir / dest.relative_to(opts.runs_root)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ok = run(sys.executable, "-m", "pneumonia.train", "--data-dir", str(opts.data_dir), "--out", str(work), *args, *opts.extra)
    if ok:
        if not keep_weights:
            (work / "best.pt").unlink(missing_ok=True)
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(work, dest)
        shutil.rmtree(work, ignore_errors=True)
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--plan", choices=("full", "smoke"), default="full")
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/pneumonia_work"))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--results-dir", type=Path, default=None, help="tables/figures destination (default: next to --runs-root)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--models", nargs="+", default=MAIN_MODELS)
    p.add_argument("--skip-ablation", action="store_true")
    opts = p.parse_args()
    opts.runs_root, opts.work_dir = opts.runs_root.resolve(), opts.work_dir.resolve()
    out_root = (opts.results_dir or opts.runs_root.parent).resolve()
    opts.extra = []  # appended after each run's own arguments, so it wins on conflicts (argparse keeps the last value)
    ablations = dict(ABLATIONS)
    if opts.plan == "smoke":  # a few steps per epoch: checks the whole pipeline in minutes
        opts.extra = ["--epochs", "2", "--max-steps", "3"]
        opts.models = [m for m in opts.models if m in ("simple_cnn", "resnet50")]
        opts.seeds = opts.seeds[:2]
        ablations = {k: ABLATIONS[k] for k in ("no_shuffle", "resnet50_linear_probe")}
    if not (opts.data_dir / "images.npy").exists():
        sys.exit(f"{opts.data_dir}/images.npy not found: run scripts/prepare_data.py first")

    py, runs, first = sys.executable, opts.runs_root, opts.seeds[0]
    failed, start = [], time.time()
    for seed in opts.seeds:
        for model in opts.models:
            dest = runs / "main" / model / f"seed{seed}"
            args = ["--model", model, "--seed", str(seed), "--split-seed", str(seed)]
            if not train(f"{model} seed {seed}", args, dest, opts, keep_weights=seed == first):
                failed.append(str(dest))
    if not opts.skip_ablation:
        for name, (description, args) in ablations.items():
            dest = runs / "ablation" / name
            seed_args = ["--seed", str(first), "--split-seed", str(first)]
            if not train(f"ablation {name}", [*args, *seed_args], dest, opts, keep_weights=False):
                failed.append(str(dest))
            elif not (dest / "ablation.json").exists():
                (dest / "ablation.json").write_text(json.dumps({"description": description, "args": args}, indent=2))

    for model in opts.models:
        dest = runs / "main" / model / f"seed{first}"
        needs_cam = (dest / "best.pt").exists() and not (dest / "gradcam.json").exists()
        if needs_cam and not run(py, "-m", "pneumonia.gradcam", "--run", str(dest), "--data-dir", str(opts.data_dir),
                                 "--figures", str(out_root / "figures")):
            failed.append(f"gradcam {dest}")

    for seed in opts.seeds:
        members = [runs / "main" / m / f"seed{seed}" for m in opts.models]
        members = [str(m) for m in members if (m / "metrics.json").exists()]
        if len(members) >= 2 and not run(py, "-m", "pneumonia.ensemble", "--out", str(runs / "ensemble" / f"seed{seed}"), *members):
            failed.append(f"ensemble seed{seed}")

    to_calibrate = sorted(str(p.parent) for p in runs.glob("main/*/seed*/metrics.json"))
    to_calibrate += sorted(str(p.parent) for p in runs.glob("ensemble/seed*/metrics.json"))
    if to_calibrate and not run(py, "-m", "pneumonia.calibrate", *to_calibrate):
        failed.append("calibrate")
    if not run(py, str(ROOT / "scripts" / "report.py"), "--runs-root", str(runs), "--out", str(out_root), "--first-seed", str(first)):
        failed.append("report")

    log(f"finished in {(time.time() - start) / 60:.0f} min; {len(failed)} failed step(s)")
    for f in failed:
        log(f"  failed: {f}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
