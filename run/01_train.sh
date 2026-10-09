#!/usr/bin/env bash
# Train Variants A-E and the endpoint baselines for every seed (resumable).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for SEED in $SEEDS; do
  for V in $VARIANTS; do
    "$PY" -m scripts.train --config "configs/variant_${V}.yaml" --seed "$SEED" --manifest "$MANIFEST" \
      --data-root "$DATA_ROOT" --output "$RUNS" --device "$DEVICE" --num-workers "$NUM_WORKERS" --skip-existing
  done
  for B in $BASELINES; do
    "$PY" -m scripts.train --config "configs/baseline_${B}.yaml" --seed "$SEED" --manifest "$MANIFEST" \
      --data-root "$DATA_ROOT" --output "$RUNS" --device "$DEVICE" --num-workers "$NUM_WORKERS" --skip-existing
  done
done
