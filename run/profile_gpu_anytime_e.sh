#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# BMAS-E standardized compute profiling (NO TRAINING)
#
# IMPORTANT:
# The archived BMAS result folders may not contain best.pt anymore. That is OK.
# Latency/FLOPs/VRAM/parameter counts depend on the architecture/exit path, not on
# the learned weights. If a checkpoint exists, it will be loaded; otherwise the
# model is instantiated from config_resolved.json and profiled with initialized
# weights. Dice/HD95 are read from the already-frozen held-out summary JSON.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/brisc2025}"
VENV_ROOT="${VENV_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
CKPT_ROOT="${CKPT_ROOT:-${OUTPUT_ROOT}/runs}"
E_RUN_PREFIX="${E_RUN_PREFIX:-bmas_E_preserve_dominant}"
PROFILE_OUT="${PROFILE_OUT:-${OUTPUT_ROOT}/profiling_anytime_e}"
SEEDS="${SEEDS:-42 2026 3407}"
DEVICE="${DEVICE:-cuda}"
WARMUP="${WARMUP:-50}"
RUNS="${RUNS:-200}"
MEMORY_RUNS="${MEMORY_RUNS:-10}"

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "WARNING: DATA_ROOT does not exist: $DATA_ROOT" >&2
  echo "Dataset is not required for architecture-only profiling because Dice/HD95 are read from frozen summaries." >&2
fi
if [[ -n "$VENV_ROOT" ]]; then PYTHON_BIN="$VENV_ROOT/bin/python"; else PYTHON_BIN="$(command -v python3 || command -v python)"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: venv Python executable not found/executable: $PYTHON_BIN" >&2
  exit 2
fi

mkdir -p logs "$PROFILE_OUT"

# Force the exact requested venv interpreter. Loading another Python module can
# shadow the venv on the HPC node, so this profiling runner does not do that.
if [[ -n "$VENV_ROOT" && -f "$VENV_ROOT/bin/activate" ]]; then
  source "$VENV_ROOT/bin/activate"
fi
[[ -n "$VENV_ROOT" ]] && export PATH="$VENV_ROOT/bin:$PATH"
export PYTHONNOUSERSITE=1
hash -r

cd "$REPO_ROOT"

echo "Python executable: $PYTHON_BIN"
"$PYTHON_BIN" - <<'PY_PRECHECK'
import sys
print("sys.executable:", sys.executable)
try:
    import torch
    import torchvision
    print("torch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("cuda_available:", torch.cuda.is_available())
except Exception as exc:
    print("FATAL: required profiling dependency import failed:", repr(exc), file=sys.stderr)
    raise
PY_PRECHECK

resolve_checkpoint() {
  local run_dir="$1"
  local candidates=(
    "best.pt" "best_model.pt" "checkpoint_best.pt" "model_best.pt"
    "best.pth" "last.pt" "last.pth"
  )
  local name
  for name in "${candidates[@]}"; do
    if [[ -f "${run_dir}/${name}" ]]; then
      printf '%s\n' "${run_dir}/${name}"
      return 0
    fi
  done
  # Also search a few levels below the run directory in case checkpoints were
  # archived into a checkpoints/ or weights/ subfolder.
  local fallback
  fallback=$(find "$run_dir" -maxdepth 3 -type f \( -name '*.pt' -o -name '*.pth' \) | sort | head -n 1 || true)
  if [[ -n "$fallback" ]]; then
    printf '%s\n' "$fallback"
    return 0
  fi
  return 1
}

resolve_metrics_summary() {
  local run_dir="$1"
  local candidates=(
    "${run_dir}/eval_test_official/summary.json"
    "${run_dir}/eval_test_official_E/summary.json"
    "${run_dir}/eval_test_paper/summary.json"
    "${run_dir}/eval_test/summary.json"
    "${run_dir}/eval_val/summary.json"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -f "$p" ]]; then
      printf '%s\n' "$p"
      return 0
    fi
  done
  return 1
}

echo "============================================================"
echo "BMAS-E standardized anytime profiling"
echo "Start:        $(date '+%F %T')"
echo "Project:      $PROJECT_ROOT"
echo "Dataset:      $DATA_ROOT"
echo "Environment:  $VENV_ROOT"
echo "Run root:     $CKPT_ROOT"
echo "Output:       $PROFILE_OUT"
echo "Seeds:        $SEEDS"
echo "Warmup/runs:  $WARMUP / $RUNS"
echo "Batch size:   1"
echo "Mode:         checkpoint if available; otherwise architecture-only profiling"
echo "Expected runtime: approximately 2-8 minutes for 3 seeds, excluding queue time."
echo "============================================================"

for SEED in $SEEDS; do
  RUN_DIR="${CKPT_ROOT}/${E_RUN_PREFIX}_seed${SEED}"
  if [[ ! -d "$RUN_DIR" ]]; then
    echo "ERROR: BMAS-E run directory not found for seed ${SEED}: $RUN_DIR" >&2
    exit 3
  fi

  CONFIG="${RUN_DIR}/config_resolved.json"
  if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config_resolved.json not found: $CONFIG" >&2
    exit 3
  fi

  METRICS_SUMMARY=$(resolve_metrics_summary "$RUN_DIR") || {
    echo "ERROR: no frozen evaluation summary found for seed ${SEED} in $RUN_DIR" >&2
    echo "Expected one of eval_test_official_E/summary.json, eval_test_paper/summary.json, eval_test/summary.json, eval_val/summary.json" >&2
    exit 3
  }

  CKPT=""
  if CKPT_FOUND=$(resolve_checkpoint "$RUN_DIR"); then
    CKPT="$CKPT_FOUND"
    PROFILE_MODE="trained-checkpoint"
  else
    PROFILE_MODE="architecture-only"
    echo "WARNING: no .pt/.pth checkpoint found for seed ${SEED}." >&2
    echo "         Continuing safely in architecture-only mode for latency/FLOPs/VRAM/params." >&2
    echo "         Dice/HD95 will be taken from: $METRICS_SUMMARY" >&2
  fi

  SEED_OUT="${PROFILE_OUT}/seed${SEED}"
  mkdir -p "$SEED_OUT"

  echo "------------------------------------------------------------"
  echo "[$(date '+%F %T')] seed=${SEED}"
  echo "mode=${PROFILE_MODE}"
  echo "config=${CONFIG}"
  echo "metrics=${METRICS_SUMMARY}"
  [[ -n "$CKPT" ]] && echo "checkpoint=${CKPT}"
  echo "------------------------------------------------------------"

  ARGS=(
    --config "$CONFIG"
    --metrics-summary "$METRICS_SUMMARY"
    --seed "$SEED"
    --output-dir "$SEED_OUT"
    --device "$DEVICE"
    --batch-size 1
    --warmup "$WARMUP"
    --runs "$RUNS"
    --memory-runs "$MEMORY_RUNS"
    --amp
  )
  if [[ -n "$CKPT" ]]; then
    ARGS+=(--checkpoint "$CKPT")
  fi

  "$PYTHON_BIN" -m scripts.profile_anytime_e "${ARGS[@]}"
done

"$PYTHON_BIN" -m scripts.summarize_anytime_profile \
  --root "$PROFILE_OUT" \
  --output "$PROFILE_OUT/profile_summary_3seeds.csv"

echo "============================================================"
echo "Completed: $(date '+%F %T')"
echo "Main CSV:  $PROFILE_OUT/profile_summary_3seeds.csv"
echo "Paper MD:  $PROFILE_OUT/PAPER_TABLE_ANYTIME_PROFILE.md"
echo "Per-seed:  $PROFILE_OUT/seed*/profile_by_exit.csv"
echo "Hardware:  $PROFILE_OUT/seed*/hardware.json"
echo "============================================================"
