# Shared settings for the run/ scripts. Override any variable from the environment.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-python3}"
DATA_ROOT="${DATA_ROOT:-datasets/brisc2025}"
MANIFEST="${MANIFEST:-data/brisc2025_manifest.csv}"
RUNS="${RUNS:-outputs/runs}"
SEEDS="${SEEDS:-42 2026 3407}"
VARIANTS="${VARIANTS:-A B C D E}"
BASELINES="${BASELINES:-unet deeplabv3 effb0_unet}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-4}}"
mkdir -p logs outputs
