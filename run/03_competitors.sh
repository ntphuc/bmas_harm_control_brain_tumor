#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# BMAS paper competitor experiments on the SAME BRISC split/protocol.
# Comparators:
#   1) MESS-style same-backbone multi-exit + stage-2 distillation
#   2) ADP-C-style same-backbone confidence-adaptive prediction mechanism
#
# These are controlled reimplementations for fair BRISC comparison. They are
# NOT claimed to be byte-identical executions of the authors' original repos.
# -----------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.."
MANIFEST="${MANIFEST:-data/brisc2025_manifest.csv}"
DATA_ROOT="${DATA_ROOT:-datasets/brisc2025}"
OUT="${OUT:-outputs/competitors}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-12}"
JOINT_EPOCHS="${JOINT_EPOCHS:-40}"
MESS_STAGE2_EPOCHS="${MESS_STAGE2_EPOCHS:-20}"
PATIENCE="${PATIENCE:-10}"
MAX_DICE_DROP="${MAX_DICE_DROP:-0.001}"
ADPC_GRID="${ADPC_GRID:-0.90 0.94 0.96 0.98 0.99}"
SEEDS="${SEEDS:-42 2026 3407}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

mkdir -p "$OUT" logs

if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: BRISC manifest not found: $MANIFEST" >&2
  echo "Set MANIFEST=/path/to/brisc2025_manifest.csv before running." >&2
  exit 2
fi

run_eval() {
  python -m competitors.evaluate \
    --checkpoint "$1" \
    --manifest "$MANIFEST" \
    --data-root "$DATA_ROOT" \
    --split "$2" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --output-dir "$3" \
    "${@:4}"
}

for SEED in $SEEDS; do
  echo "============================================================"
  echo "[$(date '+%F %T')] Seed $SEED / MESS-style"
  echo "============================================================"

  MESS_S1="$OUT/mess_style_joint_seed${SEED}/best.pt"
  if [[ "$SKIP_EXISTING" == "1" && -f "$MESS_S1" ]]; then
    echo "[resume] found $MESS_S1; skipping MESS joint training"
  else
    python -m competitors.train \
      --method mess_style \
      --stage joint \
      --manifest "$MANIFEST" \
      --data-root "$DATA_ROOT" \
      --output "$OUT" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --epochs "$JOINT_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --patience "$PATIENCE"
  fi

  MESS_CKPT="$OUT/mess_style_mess_stage2_seed${SEED}/best.pt"
  if [[ "$SKIP_EXISTING" == "1" && -f "$MESS_CKPT" ]]; then
    echo "[resume] found $MESS_CKPT; skipping MESS stage-2 training"
  else
    python -m competitors.train \
      --method mess_style \
      --stage mess_stage2 \
      --stage1-checkpoint "$MESS_S1" \
      --manifest "$MANIFEST" \
      --data-root "$DATA_ROOT" \
      --output "$OUT" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --epochs "$MESS_STAGE2_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --patience "$PATIENCE"
  fi
  MESS_VAL="$OUT/mess_style_mess_stage2_seed${SEED}/eval_val_for_calibration"
  run_eval "$MESS_CKPT" val "$MESS_VAL" --mess-thresholds 0.98 0.98 0.98

  MESS_POLICY="$OUT/mess_style_mess_stage2_seed${SEED}/mess_policy.json"
  python -m competitors.calibrate_mess \
    --per-case-csv "$MESS_VAL/per_case_metrics.csv" \
    --method mess_style \
    --max-dice-drop "$MAX_DICE_DROP" \
    --output "$MESS_POLICY"

  read -r MT1 MT2 MT3 < <(python - "$MESS_POLICY" <<'PY'
import json, sys
p=json.load(open(sys.argv[1]))
print(*p["thresholds"])
PY
)

  MESS_TEST="$OUT/mess_style_mess_stage2_seed${SEED}/eval_test_calibrated"
  run_eval "$MESS_CKPT" test "$MESS_TEST" --mess-thresholds "$MT1" "$MT2" "$MT3"

  echo "============================================================"
  echo "[$(date '+%F %T')] Seed $SEED / ADP-C-style"
  echo "============================================================"

  ADPC_CKPT="$OUT/adpc_style_joint_seed${SEED}/best.pt"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ADPC_CKPT" ]]; then
    echo "[resume] found $ADPC_CKPT; skipping ADP-C-style joint training"
  else
    python -m competitors.train \
      --method adpc_style \
      --stage joint \
      --manifest "$MANIFEST" \
      --data-root "$DATA_ROOT" \
      --output "$OUT" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --epochs "$JOINT_EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --patience "$PATIENCE"
  fi
  ADPC_SUMMARIES=()
  for T in $ADPC_GRID; do
    TAG="${T/./p}"
    ADPC_VAL_DIR="$OUT/adpc_style_joint_seed${SEED}/eval_val_t${TAG}"
    run_eval "$ADPC_CKPT" val "$ADPC_VAL_DIR" --adpc-threshold "$T"
    ADPC_SUMMARIES+=(--summary "$ADPC_VAL_DIR/summary.json")
  done

  ADPC_POLICY="$OUT/adpc_style_joint_seed${SEED}/adpc_policy.json"
  python -m competitors.select_adpc_threshold \
    "${ADPC_SUMMARIES[@]}" \
    --max-dice-drop "$MAX_DICE_DROP" \
    --output "$ADPC_POLICY"

  ADPC_T=$(python - "$ADPC_POLICY" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selected_threshold"])
PY
)

  ADPC_TEST="$OUT/adpc_style_joint_seed${SEED}/eval_test_calibrated"
  run_eval "$ADPC_CKPT" test "$ADPC_TEST" --adpc-threshold "$ADPC_T"
done

python scripts/summarize_competitors.py \
  --out-root "$OUT" \
  --output-csv "$OUT/competitor_summary_3seeds.csv"

echo "============================================================"
echo "[$(date '+%F %T')] Competitor experiments completed."
echo "Main summary: $OUT/competitor_summary_3seeds.csv"
echo "============================================================"
