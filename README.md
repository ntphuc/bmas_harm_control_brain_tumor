# BMAS: Measuring and reducing case-wise regressions across exits in multi-exit brain-tumor MRI segmentation

[![Pillow](https://img.shields.io/badge/Pillow-%3E%3D10.0-blue)](https://python-pillow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.4-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-%3E%3D0.19-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/)
[![NumPy](https://img.shields.io/badge/NumPy-%3E%3D1.26-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-%3E%3D2.2-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)

This repository provides the official PyTorch implementation of the paper

>**BMAS: Measuring and reducing case-wise regressions across exits in multi-exit brain-tumor MRI segmentation**.

**Authors:** Nguyen Thien Phuc, Tran Cao Minh, Kiet Van Nguyen, Vu Minh Tran, and Ha Minh Tan.

**Manuscript status:** Submitted to *Signal, Image and Video Processing*.

> **Repository Summary**: BMAS is a four-exit segmentation network (EfficientNet-B0 encoder, residual exits: each exit adds a correction to the previous logits). The repository trains the five objectives studied in the paper (A–E) and the endpoint baselines, evaluates every exit on the held-out set, and runs all analyses of the paper and of Online Resource 1: case-wise regression rates (MVR, BMVR), local boundary transitions (BHR, BCR), conditioning on starting quality, regression size and final-exit regret, connected components, post-hoc exit smoothing (damping, cumulative averaging), area-normalized early stopping, GPU/CPU compute profiles, the MESS-style/ADP-C-style comparators, and a 2-D nnU-Net baseline.

## Repository layout

| Path | Description |
|---|---|
| `bmas/` | BMASNet and baseline models, data loading, loss functions, and evaluation metrics. |
| `competitors/` | MESS-style and ADP-C-style multi-exit comparators using the same backbone. |
| `configs/` | YAML configurations for variants A–E and endpoint baselines. |
| `scripts/` | Data preparation, training, evaluation, and GPU profiling scripts. |
| `experiments/` | Analysis scripts, post-hoc smoothing and early stopping, CPU profiling, nnU-Net integration, and figure and supplementary material generation. |
| `run/` | Shell scripts for reproduction steps 00–06 and a Slurm job script. |
| `reproduce_all.sh` | Entry point for running the full reproduction pipeline in order. |
| `docs/` | Documentation for the analysis pipeline and GPU profiling. |

## Installation

Using **venv**:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or using **conda**:

```bash
conda create -n bmas python=3.11 -y
conda activate bmas
pip install -r requirements.txt
```

An NVIDIA GPU with CUDA support is recommended for training, while analyses can run on CPU.

## Data

Download the [BRISC 2025 dataset](https://www.kaggle.com/datasets/briscdataset/brisc2025/), originally introduced by Fateh et al. in *Scientific Data*, **13**, 361 (2026) ([DOI: 10.1038/s41597-026-06753-y](https://doi.org/10.1038/s41597-026-06753-y)), and organize the dataset as follows:

```text
datasets/brisc2025/segmentation_task/
├── train/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

If the dataset is stored elsewhere, set `DATA_ROOT` to its location.

Run `run/00_prepare_data.sh` to generate `data/brisc2025_manifest.csv`.
The official test set is kept as the held-out set (860 images). The development
set is split into training (3,343 images) and validation (590 images) using
seed 42. This split is reused across all training seeds.

To use an existing manifest from the paper, set
`MANIFEST=/path/to/manifest.csv`.

## Reproducing the paper

```bash
bash reproduce_all.sh                    # everything, in order
STEPS="02 05 06" bash reproduce_all.sh   # selected steps
sbatch run/slurm_reproduce.slurm         # same on a Slurm cluster (edit account/partition/GPU)
```

| Step | Script | What it does |
|---|---|---|
| 00 | `run/00_prepare_data.sh` | data manifest from the BRISC download |
| 01 | `run/01_train.sh` | Variants A–E and baselines (U-Net, DeepLabV3, EfficientNet-B0 U-Net) × seeds 42, 2026, 3407; 40 epochs, AdamW, best validation Exit-4 Dice |
| 02 | `run/02_evaluate.sh` | held-out evaluation of all four exits: per-case metrics, MVR/BMVR, BHR/BCR |
| 03 | `run/03_competitors.sh` | MESS-style and ADP-C-style comparators, policy calibration on validation |
| 04 | `run/04_profile_gpu.sh` | fixed-exit GPU latency, FLOPs, memory of Variant E |
| 05 | `run/05_revision_experiments.sh` | all further analyses (see `docs/revision_pipeline.md`), CPU profiling, nnU-Net |
| 06 | `run/06_build_esm.sh` | Online Resource 1 (`outputs/esm/ESM_1.tex/.pdf`) and pipeline figures of the paper |

Every number in the ESM is read from these outputs; `experiments/make_esm.py` contains no typed-in results.

### Where each result of the paper comes from

| Paper item | Output |
|---|---|
| Table 1 (endpoint quality) | `outputs/runs/*/eval_test_official/summary.json`; paired tests in `outputs/revision/analysis/v2_paired_stats.csv`; nnU-Net in `outputs/revision/nnunet/eval/` |
| Table 2 (cost per exit) | `outputs/profiling_anytime_e/profile_summary_3seeds.csv` (GPU), `outputs/revision/cpu/` (CPU) |
| Table 3 (case-wise regression) | `outputs/runs/*/eval_test_official/summary.json`; `outputs/revision/analysis/FROZEN_ANALYSES.md`; MESS-style in `outputs/competitors/competitor_summary_3seeds.csv` |
| Odds ratios, quintiles, regret, tolerance, subgroups | `outputs/revision/analysis/` (x2, x3/x4, x6, x8) |
| Connected components | `outputs/revision/x5/`, `outputs/esm/esm_extra.json` |
| Table 4 (post-hoc rules) | `outputs/revision/posthoc/x9_posthoc.csv`, `outputs/revision/frontier/` |
| Table 5 (early stopping) | `outputs/revision/posthoc/x10_stopping.csv` |
| Fig. 2 (qualitative cases) | `outputs/revision/figures/fig4_qualitative.png` |
| Fig. 3 (damping frontier) | `outputs/paper_figures/fig_frontier_col.pdf` |
| Online Resource 1 | `outputs/esm/ESM_1.pdf` |

## Notes on reproducibility

* **Run-to-run variation.** GPU kernels are not bit-deterministic, so retrained models will differ
  slightly from the paper's checkpoints; conclusions should be compared through the reported
  intervals, not the last digit.
* **Variants D and E.** `bmas/transition_losses.py` implements the local transition objectives as
  described in the paper (7×7 boundary band, detached previous exit, good/bad weighting power 2,
  preserve margin 0, top-20% tail, transition weights β = [0.2, 0.5, 1.0], λ warm-up and ramp).
* **Frozen vs re-inferred outputs.** The endpoint and regression-rate results use the evaluation of
  step 02. Component statistics, post-hoc rules, oracle gains, and early stopping re-run the same
  checkpoints in FP32 (step 05); differences between the two are at the level of rounding.
* **Exploratory analyses.** Only the A–E objectives, checkpoints, regression definitions, and the
  endpoint and regression-rate comparisons were fixed before the held-out set was first evaluated.
  All other analyses are exploratory; their tunable parameters (damping γ, stopping δ) are chosen on
  validation data only.
* **nnU-Net** is trained once with its default 2-D configuration (1,000 epochs) on the same split.

## Citation
If this repository contributes to your research, please consider citing the paper:
```bibtex
@article{phuc2026bmas,
  title   = {{BMAS}: Measuring and reducing case-wise regressions across exits in multi-exit brain-tumor {MRI} segmentation},
  author  = {Phuc, Nguyen Thien and Minh, Tran Cao and Nguyen, Kiet Van and Tran, Vu Minh and Tan, Ha Minh},
  journal = {Signal, Image and Video Processing},
  note    = {Submitted},
  year    = {2026}
}
```

## Contributing

We welcome contributions, suggestions, and improvements to this repository.
If you plan to propose significant changes or new features, please open an
issue to discuss your ideas with the maintainers before submitting a pull
request. This helps ensure that the proposed contributions align with the
scope and objectives of the project.

## Correspondence

**Ha Minh Tan** (Corresponding Author)  
University of Information Technology, Vietnam National University Ho Chi Minh City (VNU-HCM)  
Email: [tanhm@uit.edu.vn](mailto:tanhm@uit.edu.vn)
