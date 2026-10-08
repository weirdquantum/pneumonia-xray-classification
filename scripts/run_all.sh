#!/usr/bin/env bash
# Reproduce every result: data cache -> train each model -> Grad-CAM -> results table.
# Usage: bash scripts/run_all.sh [model ...]   (default: all models)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
MODELS=${*:-resnet50 efficientnet_b0 densenet121 vit_b_16 simple_cnn}

[ -f data/images.npy ] || $PY scripts/prepare_data.py --raw "${RAW_DIR:-.}" --out data

mkdir -p logs
for m in $MODELS; do
  $PY -m pneumonia.train --model "$m" 2>&1 | tee "logs/train_$m.log"
  $PY -m pneumonia.gradcam --run "runs/$m" 2>&1 | tee -a "logs/train_$m.log"
done
$PY scripts/report.py
