#!/usr/bin/env bash
# Held-out evaluation of every trained run: per-case metrics at E1-E4, MVR/BMVR, BHR/BCR (Tables 1 and 3).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
for RUN in "$RUNS"/*_seed*; do
  [[ -f "$RUN/best.pt" ]] || { echo "skip (no best.pt): $RUN"; continue; }
  "$PY" -m scripts.evaluate --run-dir "$RUN" --manifest "$MANIFEST" --data-root "$DATA_ROOT" \
    --device "$DEVICE" --num-workers "$NUM_WORKERS" --skip-existing
done
