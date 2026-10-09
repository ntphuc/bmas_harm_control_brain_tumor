# GPU profiling (step 04)

> Paths below are relative to the repository root; defaults are set in the run/ scripts.

# BMAS-E standardized fixed-exit profiling

This profiling job uses the **existing trained BMAS-E checkpoints**. It does **not** retrain any model.

## Fixed server paths

```text
Dataset:      datasets/brisc2025
Environment:  <your venv>
Project:      <repo>
Outputs:      <repo>/outputs
BMAS checkpoints:
              <repo>/outputs/outputs_ablations/
```

Expected checkpoint directories:

```text
bmas_Cdoubleprime_preserve_dominant_seed42/
bmas_Cdoubleprime_preserve_dominant_seed2026/
bmas_Cdoubleprime_preserve_dominant_seed3407/
```

The runner automatically prefers `best.pt`, `best_model.pt`, `checkpoint_best.pt`, `model_best.pt`, then `last.pt`. If those names differ, it falls back to the first `.pt/.pth` file in the seed directory and prints the resolved path into the Slurm log.

## Run

```bash
mkdir -p logs
sbatch run/slurm_reproduce.slurm
```

`run_uit.slurm` intentionally contains only Slurm configuration and one shell-script call. All dataset, environment, checkpoint and output paths are kept in `run/profile_gpu_anytime_e.sh`.

## What is measured

For each of E1, E2, E3 and E4, on the same allocated GPU and with batch size 1:

- model-only CUDA latency: mean, SD, p50 and p95 after 50 warm-up iterations and 200 timed runs;
- GFLOPs for one 256x256 input;
- PyTorch peak allocated/reserved CUDA memory;
- total model parameters and parameters actually executed by that exit;
- Dice and HD95 on the frozen test split.

The three seed results are aggregated into:

```text
<repo>/outputs/profiling_anytime_e/profile_summary_3seeds.csv
<repo>/outputs/profiling_anytime_e/PAPER_TABLE_ANYTIME_PROFILE.md
```

Hardware/software metadata are saved per seed (`hardware.json`) so the paper can state the GPU model, CUDA/PyTorch versions, batch size, warm-up and timing protocol.

## Expected runtime

Approximately **8-20 minutes for all three seeds**, excluding Slurm queue time. HD95 evaluation is typically the slowest component.

## Optional overrides

If the manifest is not under `<repo>/data/brisc2025_manifest.csv`, submit with an explicit path:

```bash
MANIFEST=/absolute/path/brisc2025_manifest.csv sbatch run/slurm_reproduce.slurm
```


## Archived checkpoints may be absent
If `best.pt` was removed from the archived result directory, the profiling script now continues in **architecture-only** mode. This is valid for latency, FLOPs, VRAM, and parameter counts because these depend on the executed model graph, not the learned values. Dice/HD95 are read from the existing frozen `eval_test_official_E/summary.json`. No retraining is performed.


## HPC dependency fix
The profiling-only scripts no longer import NumPy or pandas. The runner invokes `<your venv>/bin/python` explicitly and checks PyTorch/torchvision/CUDA before profiling. This fixes `ModuleNotFoundError: No module named numpy` without installing packages or changing the shared environment.
