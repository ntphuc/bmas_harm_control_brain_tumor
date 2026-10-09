# BMAS: Measuring and reducing case-wise regressions across exits in multi-exit brain-tumor MRI segmentation

Code for the paper *BMAS: Measuring and reducing case-wise regressions across exits in multi-exit
brain-tumor MRI segmentation* (Nguyen Thien Phuc, Tran Cao Minh, Kiet Van Nguyen, Vu Minh Tran,
Ha Minh Tan; submitted to *Signal, Image and Video Processing*).

BMAS is a four-exit segmentation network (EfficientNet-B0 encoder, residual exits: each exit adds a
correction to the previous logits). The repository trains the five objectives studied in the paper
(A–E) and the endpoint baselines, evaluates every exit on the held-out set, and runs all analyses of
the paper and of Online Resource 1: case-wise regression rates (MVR, BMVR), local boundary transitions
(BHR, BCR), conditioning on starting quality, regression size and final-exit regret, connected
components, post-hoc exit smoothing (damping, cumulative averaging), area-normalized early stopping,
GPU/CPU compute profiles, the MESS-style/ADP-C-style comparators, and a 2-D nnU-Net baseline.

## Repository layout

```
bmas/              model (BMASNet + baselines), data loader, losses (A-C), transition losses (D, E),
                   metrics (Dice, IoU, HD95, ASSD, Boundary-IoU), local boundary metrics (BHR/BCR)
competitors/       same-backbone MESS-style and ADP-C-style multi-exit comparators
configs/           one YAML per variant (A-E) and per endpoint baseline
scripts/           prepare_brisc, train, evaluate, GPU profiling helpers
experiments/       analyses, post-hoc rules, stopping, CPU profiling, nnU-Net, figures, ESM builder
run/               numbered shell steps 00-06 (+ Slurm job)
reproduce_all.sh   runs every step in order
docs/              details of the analysis pipeline and of GPU profiling
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # install torch/torchvision matching your CUDA first if needed
```

Python 3.10–3.12, PyTorch ≥ 2.4. A CUDA GPU is needed for training; analyses run on CPU.

## Data

Download BRISC 2025 (Fateh et al., *Scientific Data*, 2026; arXiv:2506.14318) and place it so that
`datasets/brisc2025/segmentation_task/{train,test}/{images,masks}/` exists, or set `DATA_ROOT`.
`run/00_prepare_data.sh` writes `data/brisc2025_manifest.csv`: the BRISC test images form the held-out
set (860 images), and the development images are split once with seed 42 into training (3,343) and
validation (590) images, reused for every training seed. If you have the manifest used for the paper,
pass it with `MANIFEST=/path/to/manifest.csv` instead.

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

```bibtex
@article{phuc2026bmas,
  title   = {{BMAS}: Measuring and reducing case-wise regressions across exits in multi-exit brain-tumor {MRI} segmentation},
  author  = {Phuc, Nguyen Thien and Minh, Tran Cao and Nguyen, Kiet Van and Tran, Vu Minh and Tan, Ha Minh},
  journal = {Signal, Image and Video Processing},
  note    = {Submitted},
  year    = {2026}
}
```

## Contact

Ha Minh Tan, University of Information Technology, VNU-HCM — tanhm@uit.edu.vn
