#!/usr/bin/env bash
# =============================================================================
# BMAS — analyses beyond the frozen evaluation (conditioning, severity, post-hoc, stopping, CPU, nnU-Net)
#
#   bash RUN_REVISION_EXPERIMENTS.sh              # all stages
#   bash RUN_REVISION_EXPERIMENTS.sh deps         # only install/check dependencies (run on a login node
#                                                 # first if compute nodes have no internet)
#   bash RUN_REVISION_EXPERIMENTS.sh analysis report   # any subset, in the order given
#
# Stages (default order):
#   deps       check torch/numpy/scipy/pandas/cv2/matplotlib; if missing, build a private overlay venv
#              that sees the shared venv's torch and pip-installs only what is missing
#   discover   locate run directories of A-E + baselines, frozen per-case CSVs and checkpoints
#   v4         exit paths, GFLOPs per exit (exit_compute.csv)                         [torch]
#   v1         match manifest samples to original BRISC files -> official split, tumor type   [numpy,cv2]
#   analysis   X1 X2 X3 X4 X6 X8 V2 V3 + Table 5 sanity check, from frozen per-case CSVs    [pure Python]
#   gpuprofile run run/profile_gpu_anytime_e.sh only if its summary is missing   [torch, GPU]
#   cpu        X11 CPU fixed-exit latency + end-to-end time                           [torch]
#   infer      re-run checkpoints: per-case outputs, post-hoc transforms, masks        [torch,numpy,scipy,cv2]
#   x5         new connected components at E4                                         [pure Python]
#   posthoc    X9 post-hoc smoothing selection + X10 early stopping                   [pure Python]
#   frontier   regression rate vs final Dice across damping strengths, all variants  [pure Python]
#   external   X14 zero-shot test on an external set (BraTS slices or a 2-D folder)  [torch, nibabel]
#   figures    Fig. 4 (X7) and Fig. 5                                                 [numpy,matplotlib]
#   nnunet     X12 nnU-Net 2-D: export -> plan -> train fold 0 -> predict -> evaluate  [nnunetv2]
#   report     REVISION_RESULTS.md collecting everything
#
# Each stage logs to $EXP_OUT/logs/<stage>.log and records OK/FAILED/SKIPPED in
# $EXP_OUT/stage_status.tsv. A failed or skipped stage does not stop later stages
# (set STRICT=1 to stop at the first failure). Re-running skips finished work.
# =============================================================================
set -uo pipefail

# ----------------------------------------------------------------- settings
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/brisc2025}"
RAW_BRISC_ROOT="${RAW_BRISC_ROOT:-$DATA_ROOT $(dirname "$DATA_ROOT")}"   # searched recursively (venvs/outputs pruned)
VENV_ROOT="${VENV_ROOT:-}"   # empty = use python3 from PATH
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
CKPT_ROOT="${CKPT_ROOT:-${OUTPUT_ROOT}/runs}"
SEARCH_ROOTS="${SEARCH_ROOTS:-${CKPT_ROOT} ${OUTPUT_ROOT}/competitors}"
MANIFEST="${MANIFEST:-${PROJECT_ROOT}/data/brisc2025_manifest.csv}"
EXP_OUT="${EXP_OUT:-${OUTPUT_ROOT}/revision}"
GPU_PROFILE_DIR="${GPU_PROFILE_DIR:-${OUTPUT_ROOT}/profiling_anytime_e}"
VARIANT_MAP="${VARIANT_MAP:-}"              # optional TSV: variant<TAB>/path/to/run_seed{seed}
SEEDS="${SEEDS:-42 2026 3407}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
CPU_THREADS="${CPU_THREADS:-1 ${SLURM_CPUS_PER_TASK:-4}}"
FIG_SEED="${FIG_SEED:-42}"
INFER_VARIANTS="${INFER_VARIANTS:-A B C D E MESS ADPC}"   # models re-run from checkpoints (missing ones are skipped)
POSTHOC_VARIANTS="${POSTHOC_VARIANTS:-A B C D E MESS}"  # models that also get damped/averaged exits
GAMMAS="${GAMMAS:-0.25 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95}"
FIG_CASES="${FIG_CASES:-}"
# X14 external set (zero-shot). Leave EXTERNAL_SOURCE empty to skip.
EXTERNAL_SOURCE="${EXTERNAL_SOURCE:-}"          # BraTS root (subject folders) or a folder of 2-D images
EXTERNAL_FORMAT="${EXTERNAL_FORMAT:-brats}"      # brats | folder
EXTERNAL_MASKS="${EXTERNAL_MASKS:-}"            # masks folder when EXTERNAL_FORMAT=folder
EXTERNAL_LABEL="${EXTERNAL_LABEL:-tc}"          # tc (necrosis+enhancing, visible on T1c) | wt
EXTERNAL_SLICES="${EXTERNAL_SLICES:-1}"         # slices per subject (1 keeps the set subject-independent)
EXTERNAL_MAX_SUBJECTS="${EXTERNAL_MAX_SUBJECTS:-0}"                  # optional: two case ids to force Fig. 4 rows
X2_BOOTSTRAP="${X2_BOOTSTRAP:-500}"
V2_BOOTSTRAP="${V2_BOOTSTRAP:-5000}"
ALLOW_PIP="${ALLOW_PIP:-1}"
OVERLAY_VENV="${OVERLAY_VENV:-${PROJECT_ROOT}/.venv_revision_overlay}"
RUN_NNUNET="${RUN_NNUNET:-1}"
NNUNET_TRAINER="${NNUNET_TRAINER:-nnUNetTrainer}"   # default self-configuration (1000 epochs)
NNUNET_DATASET_ID="${NNUNET_DATASET_ID:-501}"
NNUNET_CHECKPOINT="${NNUNET_CHECKPOINT:-checkpoint_final.pth}"
STRICT="${STRICT:-0}"
DEFAULT_STAGES="deps discover v4 v1 analysis gpuprofile cpu infer x5 posthoc frontier external figures nnunet report"
STAGES="${*:-${STAGES:-$DEFAULT_STAGES}}"

cd "$REPO_ROOT" || exit 2
CODE_DIR="$(pwd)"
mkdir -p "$EXP_OUT/logs"
STATUS="$EXP_OUT/stage_status.tsv"
touch "$STATUS"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CODE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# Numerical libraries otherwise size their thread pools (and buffers) to every core of the node,
# not to the CPUs Slurm granted; on large nodes this wastes memory and can trigger OOM kills.
NTHREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NTHREADS}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-$NTHREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$NTHREADS}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$NTHREADS}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-4}"
export MPLBACKEND=Agg MPLCONFIGDIR="${MPLCONFIGDIR:-$EXP_OUT/.matplotlib}"
mkdir -p "$MPLCONFIGDIR"

if [[ -n "$VENV_ROOT" ]]; then BASE_PY="${VENV_ROOT}/bin/python"; else BASE_PY="$(command -v python3 || command -v python)"; VENV_ROOT="$(dirname "$(dirname "$BASE_PY")")"; fi
if [[ ! -x "$BASE_PY" ]]; then
  echo "ERROR: Python not found: $BASE_PY" >&2
  exit 2
fi

resolve_python() {
  if [[ -x "${OVERLAY_VENV}/bin/python" ]]; then
    PY="${OVERLAY_VENV}/bin/python"
    export PATH="${OVERLAY_VENV}/bin:${VENV_ROOT}/bin:$PATH"
  else
    PY="$BASE_PY"
    export PATH="${VENV_ROOT}/bin:$PATH"
  fi
  hash -r
  eval "$("$PY" - <<'PY'
import importlib
def has(*mods):
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            return 0
    return 1
print(f"HAVE_TORCH={has('torch', 'torchvision')}")
print(f"HAVE_SCI={has('numpy', 'scipy', 'pandas', 'cv2')}")
print(f"HAVE_MPL={has('numpy', 'matplotlib')}")
print(f"HAVE_NNUNET={has('nnunetv2')}")
PY
)"
}

record() {  # stage status note
  local tmp
  tmp="$(mktemp)"
  awk -F'\t' -v s="$1" '$1 != s' "$STATUS" > "$tmp" 2>/dev/null || true
  printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$(date '+%F %T')" >> "$tmp"
  mv "$tmp" "$STATUS"
}

job_memory() {  # current/limit memory of this job's cgroup (v2 or v1), for OOM diagnosis
  local cur="" lim=""
  if [[ -r /sys/fs/cgroup/memory.current ]]; then
    cur=$(cat /sys/fs/cgroup/memory.current); lim=$(cat /sys/fs/cgroup/memory.max)
  elif [[ -r /sys/fs/cgroup/memory/memory.usage_in_bytes ]]; then
    cur=$(cat /sys/fs/cgroup/memory/memory.usage_in_bytes); lim=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
  fi
  [[ -n "$cur" ]] && awk -v c="$cur" -v l="$lim" 'BEGIN{
      printf "job memory in use %.1f GB", c/1073741824;
      if (l ~ /^[0-9]+$/ && l < 1e17) printf " of %.1f GB", l/1073741824; printf "\n"}'
}

run_stage() {  # name, command...
  local name="$1"; shift
  local log="$EXP_OUT/logs/${name}.log"
  echo "================ [$(date '+%F %T')] stage: $name ================"
  sync  # flush pending writes so page cache from earlier stages does not count against this one
  job_memory | sed 's/^/[mem] before: /'
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  job_memory | sed 's/^/[mem] after:  /' | tee -a "$log"
  if [[ $rc -eq 0 ]]; then
    record "$name" OK "log: logs/${name}.log"
  elif [[ $rc -eq 3 ]]; then
    record "$name" SKIPPED "required input missing; see logs/${name}.log"
    echo "--- stage $name skipped: required input missing (see log)" >&2
  elif [[ $rc -eq 137 ]]; then
    record "$name" FAILED "killed (exit 137, out of memory); see [mem] lines in logs/${name}.log"
    echo "!!! stage $name killed (exit 137 = out of memory); continuing" >&2
    [[ "$STRICT" == "1" ]] && exit "$rc"
  else
    record "$name" FAILED "exit $rc; see logs/${name}.log"
    echo "!!! stage $name failed (exit $rc); continuing" >&2
    [[ "$STRICT" == "1" ]] && exit "$rc"
  fi
  return 0
}

skip_stage() {
  echo "---------------- stage $1 SKIPPED: $2"
  record "$1" SKIPPED "$2"
}

need_map() {
  [[ -f "$EXP_OUT/discover/variant_map.json" ]]
}

# ----------------------------------------------------------------- stages
stage_deps() {
  resolve_python
  echo "python: $PY"
  echo "torch=$HAVE_TORCH sci=$HAVE_SCI matplotlib=$HAVE_MPL nnunetv2=$HAVE_NNUNET"
  local missing=()
  "$PY" -c 'import numpy' 2>/dev/null || missing+=(numpy)
  "$PY" -c 'import scipy' 2>/dev/null || missing+=(scipy)
  "$PY" -c 'import pandas' 2>/dev/null || missing+=(pandas)
  "$PY" -c 'import cv2' 2>/dev/null || missing+=(opencv-python-headless)
  "$PY" -c 'import matplotlib' 2>/dev/null || missing+=(matplotlib)
  "$PY" -c 'import yaml' 2>/dev/null || missing+=(PyYAML)
  "$PY" -c 'import tqdm' 2>/dev/null || missing+=(tqdm)
  if [[ -n "$EXTERNAL_SOURCE" && "$EXTERNAL_FORMAT" == "brats" ]]; then
    "$PY" -c 'import nibabel' 2>/dev/null || missing+=(nibabel)
  fi
  if [[ "$RUN_NNUNET" == "1" ]]; then
    "$PY" -c 'import nnunetv2' 2>/dev/null || missing+=(nnunetv2)
    # nnU-Net still imports distutils, removed from the standard library in Python 3.12;
    # setuptools provides the replacement module.
    "$PY" -c 'import distutils.file_util' 2>/dev/null || missing+=(setuptools)
  fi
  if [[ ${#missing[@]} -eq 0 ]]; then
    echo "all dependencies present"
    return 0
  fi
  echo "missing: ${missing[*]}"
  if [[ "$ALLOW_PIP" != "1" ]]; then
    echo "ALLOW_PIP=0: not installing; stages needing these will be skipped"
    return 0
  fi
  if [[ ! -x "${OVERLAY_VENV}/bin/python" ]]; then
    echo "creating overlay venv at $OVERLAY_VENV (shared venv is not modified)"
    "$BASE_PY" -m venv "$OVERLAY_VENV" || { echo "venv creation failed"; return 1; }
    local ovl_site
    ovl_site="$("${OVERLAY_VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    "$BASE_PY" - "$ovl_site" <<'PY'
import sys, sysconfig
paths = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
with open(f"{sys.argv[1]}/zz_shared_venv.pth", "w") as f:
    f.write("\n".join(sorted(paths)) + "\n")
print("linked shared site-packages:", sorted(paths))
PY
  fi
  "${OVERLAY_VENV}/bin/python" -m pip install --no-input --upgrade pip >/dev/null 2>&1 || true
  if ! "${OVERLAY_VENV}/bin/python" -m pip install --no-input "${missing[@]}"; then
    echo "pip install failed (no internet on this node?). Run 'bash RUN_REVISION_EXPERIMENTS.sh deps' on a" \
         "login node, then resubmit." >&2
    return 1
  fi
  resolve_python
  echo "after install: torch=$HAVE_TORCH sci=$HAVE_SCI matplotlib=$HAVE_MPL nnunetv2=$HAVE_NNUNET"
  "$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
}

stage_discover() {
  local args=(--roots $SEARCH_ROOTS --seeds $SEEDS --output-dir "$EXP_OUT/discover")
  [[ -n "$VARIANT_MAP" ]] && args+=(--override "$VARIANT_MAP")
  "$PY" -m experiments.discover_runs "${args[@]}" | tee "$EXP_OUT/discover/discover.log.txt"
  local rc=${PIPESTATUS[0]}
  { echo "# Run discovery"; echo; echo '```'; cat "$EXP_OUT/discover/discover.log.txt"; echo '```'; } \
    > "$EXP_OUT/discover/discover.log.md"
  return "$rc"
}

stage_analysis() {
  local args=(--variant-map "$EXP_OUT/discover/variant_map.json" --output-dir "$EXP_OUT/analysis"
              --x2-bootstrap "$X2_BOOTSTRAP" --v2-bootstrap "$V2_BOOTSTRAP")
  [[ -f "$EXP_OUT/v4/exit_compute.csv" ]] && args+=(--exit-compute "$EXP_OUT/v4/exit_compute.csv")
  [[ -f "$EXP_OUT/v1/case_metadata.csv" ]] && args+=(--metadata "$EXP_OUT/v1/case_metadata.csv")
  "$PY" -m experiments.analyze_frozen "${args[@]}"
}

stage_gpuprofile() {
  if [[ -f "$GPU_PROFILE_DIR/profile_summary_3seeds.csv" ]]; then
    echo "GPU profile already present: $GPU_PROFILE_DIR/profile_summary_3seeds.csv"
    return 0
  fi
  PROJECT_ROOT="$PROJECT_ROOT" DATA_ROOT="$DATA_ROOT" VENV_ROOT="$VENV_ROOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
    CKPT_ROOT="$CKPT_ROOT" PROFILE_OUT="$GPU_PROFILE_DIR" SEEDS="$SEEDS" bash "$CODE_DIR/run/profile_gpu_anytime_e.sh"
}

stage_infer() {
  local dev="$DEVICE"
  if [[ "$dev" == "cuda" ]] && ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA not available: inference falls back to CPU (slow)"
    dev=cpu
  fi
  "$PY" -m experiments.infer_exits --variant-map "$EXP_OUT/discover/variant_map.json" \
    --manifest "$MANIFEST" --data-root "$DATA_ROOT" --out-root "$EXP_OUT/infer" \
    --device "$dev" --num-workers "$NUM_WORKERS" --mask-seed "$FIG_SEED" \
    --variants $INFER_VARIANTS --posthoc-variants $POSTHOC_VARIANTS --gammas $GAMMAS
}

stage_posthoc() {
  local args=(--infer-root "$EXP_OUT/infer" --seeds $SEEDS --output-dir "$EXP_OUT/posthoc")
  [[ -f "$EXP_OUT/v4/exit_compute.csv" ]] && args+=(--exit-compute "$EXP_OUT/v4/exit_compute.csv")
  [[ -f "$GPU_PROFILE_DIR/profile_summary_3seeds.csv" ]] && args+=(--gpu-profile "$GPU_PROFILE_DIR/profile_summary_3seeds.csv")
  [[ -f "$EXP_OUT/cpu/cpu_profile_summary.csv" ]] && args+=(--cpu-profile "$EXP_OUT/cpu/cpu_profile_summary.csv")
  "$PY" -m experiments.posthoc_stopping "${args[@]}"
}

stage_figures() {
  local args=(--infer-root "$EXP_OUT/infer" --analysis-dir "$EXP_OUT/analysis" --seed "$FIG_SEED"
              --output-dir "$EXP_OUT/figures")
  [[ -n "$FIG_CASES" ]] && args+=(--cases $FIG_CASES)
  "$PY" -m experiments.make_figures "${args[@]}"
}

stage_external() {
  local ext="$EXP_OUT/external"
  if [[ ! -f "$ext/data/manifest.csv" ]]; then
    local prep=(--format "$EXTERNAL_FORMAT" --source "$EXTERNAL_SOURCE" --label "$EXTERNAL_LABEL"
                --slices-per-subject "$EXTERNAL_SLICES" --max-subjects "$EXTERNAL_MAX_SUBJECTS" --output-dir "$ext/data")
    [[ -n "$EXTERNAL_MASKS" ]] && prep+=(--masks "$EXTERNAL_MASKS")
    "$PY" -m experiments.external_prepare "${prep[@]}" || return 1
  fi
  local dev="$DEVICE"
  if [[ "$dev" == "cuda" ]] && ! "$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    dev=cpu
  fi
  "$PY" -m experiments.infer_exits --variant-map "$EXP_OUT/discover/variant_map.json" \
    --manifest "$ext/data/manifest.csv" --data-root "$ext/data" --splits test --out-root "$ext/infer" \
    --device "$dev" --num-workers "$NUM_WORKERS" --mask-seed "$FIG_SEED" --no-compare-frozen \
    --variants $INFER_VARIANTS --posthoc-variants $POSTHOC_VARIANTS --gammas $GAMMAS || return 1
  "$PY" -m experiments.external_eval --infer-root "$ext/infer" --seeds $SEEDS \
    --selection "$EXP_OUT/frontier/frontier_selection.json" --output-dir "$ext/eval"
}

stage_nnunet() {
  local base="$EXP_OUT/nnunet"
  export nnUNet_raw="$base/nnUNet_raw" nnUNet_preprocessed="$base/nnUNet_preprocessed" nnUNet_results="$base/nnUNet_results"
  export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-${SLURM_CPUS_PER_TASK:-4}}"
  export nnUNet_compile="${nnUNet_compile:-f}"
  local did; did=$(printf '%03d' "$NNUNET_DATASET_ID")
  local dsname="Dataset${did}_BRISC"
  mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
  if [[ ! -f "$nnUNet_raw/$dsname/dataset.json" ]]; then
    "$PY" -m experiments.nnunet_export --manifest "$MANIFEST" --data-root "$DATA_ROOT" --nnunet-raw "$nnUNet_raw" \
      --dataset-id "$NNUNET_DATASET_ID" --output-dir "$base/export" || return 1
  fi
  if [[ ! -f "$nnUNet_preprocessed/$dsname/.preprocess_done" ]]; then
    rm -rf "$nnUNet_preprocessed/$dsname"
    nnUNetv2_plan_and_preprocess -d "$NNUNET_DATASET_ID" -c 2d --verify_dataset_integrity \
      -np "${SLURM_CPUS_PER_TASK:-4}" || return 1
    touch "$nnUNet_preprocessed/$dsname/.preprocess_done"
  fi
  cp "$base/export/splits_final.json" "$nnUNet_preprocessed/$dsname/splits_final.json"
  local fold_dir="$nnUNet_results/$dsname/${NNUNET_TRAINER}__nnUNetPlans__2d/fold_0"
  if [[ ! -f "$fold_dir/checkpoint_final.pth" ]]; then
    local cont=()
    [[ -f "$fold_dir/checkpoint_latest.pth" ]] && cont=(--c)
    nnUNetv2_train "$NNUNET_DATASET_ID" 2d 0 -tr "$NNUNET_TRAINER" ${cont[@]+"${cont[@]}"} || return 1
  fi
  if [[ ! -f "$base/pred_test/.done" ]]; then
    nnUNetv2_predict -i "$nnUNet_raw/$dsname/imagesTs" -o "$base/pred_test" -d "$NNUNET_DATASET_ID" -c 2d -f 0 \
      -tr "$NNUNET_TRAINER" -chk "$NNUNET_CHECKPOINT" || return 1
    touch "$base/pred_test/.done"
  fi
  "$PY" -m experiments.nnunet_eval --pred-dir "$base/pred_test" --labels-dir "$nnUNet_raw/$dsname/labelsTs" \
    --name-map "$base/export/nnunet_name_map.csv" --variant-map "$EXP_OUT/discover/variant_map.json" \
    --trainer "$NNUNET_TRAINER" --output-dir "$base/eval"
}

# ----------------------------------------------------------------- driver
resolve_python
echo "============================================================"
echo "BMAS SIVP revision experiments — start $(date '+%F %T')"
echo "code:        $CODE_DIR"
echo "python:      $PY"
echo "project:     $PROJECT_ROOT"
echo "runs:        $CKPT_ROOT"
echo "manifest:    $MANIFEST"
echo "raw BRISC:   $RAW_BRISC_ROOT"
echo "output:      $EXP_OUT"
echo "seeds:       $SEEDS"
echo "stages:      $STAGES"
echo "host:        $(hostname)  SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || true
echo "threads:     OMP/MKL/OPENBLAS=$NTHREADS"
job_memory | sed 's/^/memory:      /'
echo "Slurm mem:   ${SLURM_MEM_PER_NODE:-${SLURM_MEM_PER_CPU:-unknown}} MB"
echo "============================================================"

for STAGE in $STAGES; do
  resolve_python
  case "$STAGE" in
    deps)      run_stage deps stage_deps ;;
    discover)  mkdir -p "$EXP_OUT/discover"; run_stage discover stage_discover ;;
    v4)        if [[ $HAVE_TORCH == 1 ]]; then
                 run_stage v4 "$PY" -m experiments.exit_graph --output-dir "$EXP_OUT/v4"
               else skip_stage v4 "torch not importable"; fi ;;
    v1)        if [[ $HAVE_SCI != 1 ]]; then skip_stage v1 "numpy/scipy/pandas/cv2 missing (run deps)"
               elif [[ ! -f "$MANIFEST" ]]; then skip_stage v1 "manifest not found: $MANIFEST"
               elif [[ -f "$EXP_OUT/v1/case_metadata.csv" ]]; then record v1 OK "cached"
               else run_stage v1 "$PY" -m experiments.v1_split_match --manifest "$MANIFEST" \
                      --data-root "$DATA_ROOT" --raw-root $RAW_BRISC_ROOT --output-dir "$EXP_OUT/v1"; fi ;;
    analysis)  if need_map; then run_stage analysis stage_analysis
               else skip_stage analysis "run discover first"; fi ;;
    gpuprofile) if [[ $HAVE_TORCH == 1 ]]; then run_stage gpuprofile stage_gpuprofile
               else skip_stage gpuprofile "torch not importable"; fi ;;
    cpu)       if [[ $HAVE_TORCH != 1 ]]; then skip_stage cpu "torch not importable"
               elif ! need_map; then skip_stage cpu "run discover first"
               else
                 cpu_args=(--variant-map "$EXP_OUT/discover/variant_map.json" --threads $CPU_THREADS
                           --output-dir "$EXP_OUT/cpu")
                 [[ $HAVE_SCI == 1 && -f "$MANIFEST" ]] && cpu_args+=(--manifest "$MANIFEST" --data-root "$DATA_ROOT")
                 run_stage cpu "$PY" -m experiments.profile_cpu "${cpu_args[@]}"
               fi ;;
    infer)     if [[ $HAVE_TORCH != 1 || $HAVE_SCI != 1 ]]; then skip_stage infer "needs torch + numpy/scipy/pandas/cv2"
               elif ! need_map; then skip_stage infer "run discover first"
               elif [[ ! -f "$MANIFEST" ]]; then skip_stage infer "manifest not found: $MANIFEST"
               else run_stage infer stage_infer; fi ;;
    x5)        if [[ -f "$EXP_OUT/infer/INFER_DONE" ]]; then
                 run_stage x5 "$PY" -m experiments.x5_components --infer-root "$EXP_OUT/infer" --seeds $SEEDS \
                   --output-dir "$EXP_OUT/x5"
               else skip_stage x5 "no inference outputs (checkpoints missing?)"; fi ;;
    posthoc)   if [[ -f "$EXP_OUT/infer/INFER_DONE" ]]; then run_stage posthoc stage_posthoc
               else skip_stage posthoc "no inference outputs (checkpoints missing?)"; fi ;;
    frontier)  if [[ -f "$EXP_OUT/infer/INFER_DONE" ]]; then
                 run_stage frontier "$PY" -m experiments.frontier --infer-roots "$EXP_OUT/infer" --seeds $SEEDS \
                   --output-dir "$EXP_OUT/frontier"
               else skip_stage frontier "no inference outputs"; fi ;;
    external)  if [[ -z "$EXTERNAL_SOURCE" ]]; then skip_stage external "EXTERNAL_SOURCE not set"
               elif [[ $HAVE_TORCH != 1 || $HAVE_SCI != 1 ]]; then skip_stage external "needs torch + numpy/scipy/cv2"
               elif ! need_map; then skip_stage external "run discover first"
               else run_stage external stage_external; fi ;;
    figures)   if [[ $HAVE_MPL != 1 ]]; then skip_stage figures "matplotlib/numpy missing"
               else run_stage figures stage_figures; fi ;;
    nnunet)    if [[ "$RUN_NNUNET" != "1" ]]; then skip_stage nnunet "RUN_NNUNET=0"
               elif [[ $HAVE_NNUNET != 1 || $HAVE_SCI != 1 ]]; then skip_stage nnunet "nnunetv2 not installed (run deps on a login node)"
               elif ! need_map; then skip_stage nnunet "run discover first"
               else run_stage nnunet stage_nnunet; fi ;;
    report)    run_stage report "$PY" -m experiments.build_report --exp-out "$EXP_OUT" ;;
    *)         echo "unknown stage: $STAGE" >&2 ;;
  esac
done

echo "============================================================"
echo "Finished $(date '+%F %T')"
echo "Status:  $STATUS"
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
echo "Report:  $EXP_OUT/REVISION_RESULTS.md"
echo "============================================================"
