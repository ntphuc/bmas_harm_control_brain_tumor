#!/usr/bin/env bash
# Fixed-exit GPU profile of Variant E (Table 2, GPU columns).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CKPT_ROOT="${CKPT_ROOT:-outputs/runs}" PROFILE_OUT="${PROFILE_OUT:-outputs/profiling_anytime_e}" \
  bash run/profile_gpu_anytime_e.sh
