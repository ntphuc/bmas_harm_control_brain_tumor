#!/usr/bin/env bash
# =============================================================================
# Reproduce every experiment of the BMAS paper and its Online Resource 1.
#
#   bash reproduce_all.sh                 # all steps
#   STEPS="02 05 06" bash reproduce_all.sh   # a subset
#
#   00 data manifest            02 held-out evaluation     04 GPU profile (E)
#   01 train A-E + baselines    03 MESS/ADP-C comparators   05 analyses (conditioning, severity,
#                                                             post-hoc, stopping, CPU, nnU-Net, figures)
#   06 ESM + paper figures
# Settings: see run/common.sh and run/05_revision_experiments.sh (environment variables).
# =============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STEPS="${STEPS:-00 01 02 03 04 05 06}"
mkdir -p logs
for S in $STEPS; do
  case "$S" in
    00) bash run/00_prepare_data.sh ;;
    01) bash run/01_train.sh ;;
    02) bash run/02_evaluate.sh ;;
    03) bash run/03_competitors.sh ;;
    04) bash run/04_profile_gpu.sh ;;
    05) CKPT_ROOT=outputs/runs EXP_OUT=outputs/revision bash run/05_revision_experiments.sh ;;
    06) bash run/06_build_esm.sh ;;
    *) echo "unknown step $S" ;;
  esac
  rc=$?
  echo "[$(date '+%F %T')] step $S finished with exit code $rc"
  [[ $rc -ne 0 && "${STRICT:-0}" == "1" ]] && exit $rc
done
