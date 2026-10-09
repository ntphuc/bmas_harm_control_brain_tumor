# Analysis pipeline (step 05)

> Paths below are relative to the repository root; defaults are set in the run/ scripts.

# BMAS — SIVP revision experiments

One entry point runs every experiment requested in the revision plan:

```bash
mkdir -p logs
bash run/05_revision_experiments.sh deps      # once, on a login node if compute nodes have no internet
sbatch run/slurm_reproduce.slurm                       # everything else on the GPU node
```

Results: `$EXP_OUT/REVISION_RESULTS.md` (default `outputs/revision/`).
Stage status: `$EXP_OUT/stage_status.tsv`; per-stage logs: `$EXP_OUT/logs/`.

## Stages and what they produce

| Stage | Revision item | Needs | Output |
|---|---|---|---|
| deps | — | internet if packages are missing | private overlay venv `.venv_revision_overlay` (shared venv untouched) |
| discover | — | archived run folders | `discover/variant_map.json`, `all_candidate_runs.csv` |
| v4 | V4 exit paths, GFLOPs per exit | torch | `v4/v4_exit_graph.md`, `exit_compute.csv` |
| v1 | V1 official split, tumor type for X8 | numpy, cv2, original BRISC files | `v1/split_contingency.md`, `case_metadata.csv` |
| analysis | X1 X2 X3 X4 X6 X8 V2 V3 + Table 5 check | frozen per-case CSVs only | `analysis/FROZEN_ANALYSES.md` + CSVs |
| gpuprofile | Table 3 (existing) | torch, GPU | reuses `run/profile_gpu_anytime_e.sh` only if its summary is missing |
| cpu | X11 | torch | `cpu/CPU_PROFILE.md` |
| infer | inputs for X5 X7 X9 X10 | torch + **checkpoints** | `infer/<variant>/seed<s>/<split>/per_case_*.csv`, masks |
| x5 | X5 new components at E4 | infer | `x5/X5_COMPONENTS.md` |
| posthoc | X9 post-hoc smoothing, X10 early stopping | infer | `posthoc/POSTHOC_STOPPING.md` |
| frontier | regression rate vs final Dice for every variant across damping strengths | infer | `frontier/FRONTIER.md`, `fig_frontier.png` |
| figures | Fig. 4 (X7), Fig. 5 | matplotlib | `figures/*.png/pdf` |
| nnunet | X12 nnU-Net 2-D | nnunetv2, ~12–20 h GPU (estimate) | `nnunet/eval/NNUNET.md` |
| report | — | — | `REVISION_RESULTS.md` |

Run a subset: `bash run/05_revision_experiments.sh analysis figures report`.
A failed or skipped stage does not stop the others (`STRICT=1` to stop). Re-running skips finished work.

## Settings (environment variables)

| Variable | Default |
|---|---|
| `PROJECT_ROOT` | `<repo>` |
| `DATA_ROOT` / `RAW_BRISC_ROOT` | `datasets/brisc2025` |
| `VENV_ROOT` | `<your venv>` |
| `CKPT_ROOT` | `$PROJECT_ROOT/outputs/outputs_ablations` |
| `MANIFEST` | `$PROJECT_ROOT/data/brisc2025_manifest.csv` |
| `EXP_OUT` | `$PROJECT_ROOT/outputs/revision_sivp` |
| `VARIANT_MAP` | empty = auto-discovery; or a TSV `variant<TAB>/path/run_seed{seed}` |
| `SEEDS` | `42 2026 3407` |
| `RUN_NNUNET` | `1` |
| `NNUNET_TRAINER` | `nnUNetTrainer` (default 1000 epochs) |
| `FIG_CASES` | empty = automatic choice; or two case ids for Fig. 4 |
| `ALLOW_PIP` | `1` |

## Check after `discover`

Auto-discovery labels run folders from their config and, when a label has several
candidates, keeps the one whose frozen Exit-4 Dice is closest to the manuscript.
Read `discover/discover.log.md`. Any `WARNING ... differs from manuscript` means the
wrong folder may have been picked: pin it with `VARIANT_MAP`, e.g.

```
A	<repo>/outputs/outputs_ablations/<A_prefix>_seed{seed}
B	<repo>/outputs/outputs_ablations/<B_prefix>_seed{seed}
E	<repo>/outputs/outputs_ablations/bmas_Cdoubleprime_preserve_dominant_seed{seed}
```

The "Table 5 recomputed" block at the top of `FROZEN_ANALYSES.md` must say OK for A–E
before any new number is used in the paper.

## Definitions used

- Regression: Dice(k+1) < Dice(k) − 0.001; HD95(k+1) > HD95(k) + 0.5 px.
- X2: transitions stratified by quintiles of Dice_k / HD95_k; logistic model
  `regression ~ variant + Dice_k + log1p(HD95_k) + transition`, image-cluster bootstrap CI.
- X3: mean size of flagged regressions; CVaR95 = mean of the largest 5% of changes.
- X4: R(Dice) = share of images with Dice4 < max(Dice1..3) − 0.001 (HD95 analogous).
- X9: cumulative averaging z̃k = mean(z1..zk); damping z̃(k+1) = z̃k + γ(z(k+1) − zk),
  γ ∈ {0.25, 0.5, 0.75} chosen per seed on validation (lowest MVR with final Dice within 0.001).
- X10: stop at the first exit k ≤ 3 with (uncertain px + 1)/(predicted tumour px + 1) < δ,
  uncertain = probability in [0.4, 0.6]; δ calibrated on validation for ε ∈ {0.005, 0.01, 0.02}.
  **Update Section 5.3 of the manuscript to this exact ratio.**
- V2: per-image values are averaged over seeds (metric averaging), then paired.
- V3 (from `bmas/metrics.py`): HD95 = ASSD = image diagonal (362.04 px) when exactly one of
  prediction/reference is empty; 0 when both are empty.

## Not covered (needs material outside this package)

- V7 loss equations for D/E and X13 extra seeds: the original BMAS training code
  (`RUN_NOGATE_3SEEDS.sh`, D/E losses) is not in this package.
- If `best.pt` files were deleted from the archived runs, `infer`, `x5`, `posthoc` and Fig. 4
  are skipped; restoring or retraining the checkpoints is the only way to obtain X5, X7, X9, X10.

## Testing status

Run end-to-end on synthetic data without torch (discover, v1, analysis, x5, posthoc, figures,
nnU-Net export/eval, report); statistics checked against SciPy. Torch-dependent stages
(v4, cpu, infer) were only syntax- and lint-checked: watch their logs on the first run.

## Second run (after the first results)

Changes in this version:

- `infer` now writes damped and cumulatively averaged exits for **all** variants
  (`POSTHOC_VARIANTS`, default `A B C D E`) on a finer grid (`GAMMAS`, default
  `0.25 0.5 0.6 0.7 0.75 0.8 0.85 0.9 0.95`). New stage `frontier` compares the curves.
  This is the decisive test of whether objective E adds anything beyond post-hoc damping.
- `deps` installs `setuptools` when nnU-Net needs `distutils` (removed in Python 3.12).
  Interrupted nnU-Net preprocessing is redone automatically.
- Fig. 4 unpacks only the masks it draws.

Re-run only what changed (existing inference outputs are re-computed because new transforms are requested):

```bash
bash run/05_revision_experiments.sh deps                              # login node: installs setuptools
RAW_BRISC_ROOT=/path/to/original/BRISC2025 sbatch run/slurm_reproduce.slurm    # or STAGES below
STAGES="infer x5 posthoc frontier figures v1 analysis nnunet report" sbatch run/slurm_reproduce.slurm
```

`RAW_BRISC_ROOT` must point to the original BRISC download that contains the
`brisc2025_<split>_<id>_<tumor>_<plane>_<seq>.jpg/.png` files; without it V1 (official split) and
the tumor-type subgroups of X8 cannot run.

## Third version: experiments that decide whether the paper holds

| ID | Question | What the code does | Output |
|---|---|---|---|
| F1 | Does objective E add anything beyond post-hoc damping? | damping γ ∈ {0.25 … 0.95} for A–E (and MESS if its checkpoint exists); E vs B, E vs A, B vs A, MESS vs B compared at four matched final-Dice levels (γ chosen on validation), BH-corrected | `frontier/FRONTIER.md`, `frontier_matched.csv`, `fig_frontier.png`, `frontier_selection.json` |
| X14 | Do the findings survive distribution shift? | builds a subject-independent external set (one T1c slice per BraTS subject, tumor core = labels except edema) and re-runs every checkpoint zero-shot; damping uses the γ chosen on BRISC | `external/eval/EXTERNAL.md` |
| MESS | Is damping specific to residual exits? | infers MESS-style/ADP-C-style checkpoints (if present), their raw-exit MVR/BMVR (verifies ESM Table S5) and their damping frontier | `analysis/FROZEN_ANALYSES.md`, `frontier/` |
| X5+ | How much of the late HD95 increase comes from new fragments? | adds mean ΔHD95 with/without a new component and its share of the total | `x5/X5_COMPONENTS.md` |
| V1 | Official split, tumor type | searches `RAW_BRISC_ROOT` (several folders allowed, venvs/outputs pruned) | `v1/split_contingency.md` |
| X12 | nnU-Net 2-D | `distutils` fix; interrupted preprocessing redone | `nnunet/eval/NNUNET.md` |

Baselines with single-exit per-case files (EffB0-U-Net, DeepLabV3, U-Net) are now recognised, so
V2 also reports E against them (separate BH family), and against the independent-exit comparators.

### Run

```bash
bash run/05_revision_experiments.sh deps                     # login node (setuptools, nibabel if BraTS)

# everything except nnU-Net (about 1-2 h on the MIG slice):
STAGES="discover v1 analysis infer x5 posthoc frontier external figures report" \
RAW_BRISC_ROOT="/path/to/BRISC2025_original" \
EXTERNAL_SOURCE="/path/to/BraTS2021_TrainingData" EXTERNAL_FORMAT=brats \
sbatch run/slurm_reproduce.slurm

# nnU-Net separately (long; default trainer = 1000 epochs):
STAGES="deps nnunet report" sbatch run/slurm_reproduce.slurm
```

`EXTERNAL_FORMAT=folder EXTERNAL_SOURCE=/imgs EXTERNAL_MASKS=/masks` accepts any 2-D image/mask
folder instead of BraTS. Check that the external data do not share sources with BRISC before
reporting them as external.

### Reading the decisive results

- `frontier/FRONTIER.md`, rows "E vs B": positive MVR/BMVR improvement with CI above zero at matched
  Dice → E adds something beyond damping (keep it as a contribution). CI straddling zero at every
  level → the frontiers coincide; keep the paper's current framing (damping as the default control).
- `external/eval/EXTERNAL.md`: the paper's claims hold under shift if E vs A and E vs B keep positive
  MVR/BMVR improvements and damped B still matches E.

## Fix for job 2208 (v1 exit 1, figures exit 137 / OOM)

- `v1`: when no original BRISC files are found the stage is now SKIPPED (exit 3) with the reason,
  instead of FAILED. Set `RAW_BRISC_ROOT` to the original BRISC 2025 download to run it.
- `figures`: Fig. 4 picks its two cases from the per-case CSVs first and unpacks only those masks
  (peak memory ~0.2 GB at full size, measured). Stage logs now end with `[mem] peak RSS` and the
  job's cgroup usage before/after each stage, so an OOM can be traced to the stage that grew.
- The run script limits OMP/MKL/OpenBLAS threads to `SLURM_CPUS_PER_TASK` and flushes pending writes
  (`sync`) before every stage; exit 137 is reported as an out-of-memory kill in `stage_status.tsv`.

Re-run only these stages:

```bash
STAGES="v1 figures report" RAW_BRISC_ROOT=/path/to/BRISC2025_original sbatch run/slurm_reproduce.slurm
```
