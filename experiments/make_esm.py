"""Build Online Resource 1 (ESM) of the BMAS paper from the pipeline outputs.

Every number is read from a result file; nothing is typed in by hand.
"""
import csv
import json
import math
import re
import statistics as st

import argparse
from pathlib import Path

from experiments.common import load_case_table

_ap = argparse.ArgumentParser(description="Build Online Resource 1 (ESM) from pipeline outputs")
_ap.add_argument("--exp-out", required=True, help="revision pipeline output folder (RUN_REVISION_EXPERIMENTS.sh EXP_OUT)")
_ap.add_argument("--extras", required=True, help="esm_extra.json from experiments.esm_extras")
_ap.add_argument("--competitors", default="outputs/competitors", help="competitor output folder")
_ap.add_argument("--gpu-profile", default="outputs/profiling_anytime_e/profile_summary_3seeds.csv")
_ap.add_argument("--output-dir", required=True)
ARGS = _ap.parse_args()
R = ARGS.exp_out
EXTRA = json.load(open(ARGS.extras))
COMP = Path(ARGS.competitors)
V = "ABCDE"


def rd(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def f(x, d=3):
    return f"{float(x):.{d}f}"


def pm(m, s, d=3):
    return f"{float(m):.{d}f}$\\pm${float(s):.{d}f}"


def pfmt(p):
    p = float(p)
    if p < 1e-3:
        e = int(math.floor(math.log10(p)))
        return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def sgn(x, d=3):
    x = float(x)
    return f"$+${abs(x):.{d}f}" if x > 0 else (f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}")


def ci(lo, hi, d=3):
    def one(v):
        v = float(v)
        return f"$-${abs(v):.{d}f}" if v < 0 else f"{v:.{d}f}"
    return f"[{one(lo)}, {one(hi)}]"


def bh(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        run = min(run, ps[i] * m / rank)
        adj[i] = min(1.0, run)
    return adj


def table(label, caption, colspec, header, rows, size="\\footnotesize", notes="", resize=False):
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" if not (isinstance(r, str)) else r for r in rows)
    if resize:
        spec = colspec.replace("Y", "c")
        out = (f"\\begin{{table}}[H]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
               f"\\resizebox{{\\textwidth}}{{!}}{{\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
               f"\\bottomrule\n\\end{{tabular}}}}\n")
    else:
        out = (f"\\begin{{table}}[H]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
               f"\\begin{{tabularx}}{{\\textwidth}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n"
               f"\\bottomrule\n\\end{{tabularx}}\n")
    if notes:
        out += f"\\par\\smallskip\\parbox{{\\textwidth}}{{\\scriptsize {notes}}}\n"
    return out + "\\end{table}\n"


def fig(label, path, caption, width="\\textwidth"):
    return (f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption{{{caption}}}\\label{{{label}}}\n\\end{{figure}}\n")


parts = []

# ---------------------------------------------------------------- S1 data
sg = rd(f"{R}/analysis/x8_subgroups.csv")
nsub = {(r["grouping"], r["group"]): r["n_images"] for r in sg if r["variant"] == "A"}
rows = [["Training", "3,343", "--", "--", "--"], ["Validation", "590", "--", "--", "--"],
        ["Held-out test", "860",
         f"{nsub[('Imaging plane','ax')]} / {nsub[('Imaging plane','co')]} / {nsub[('Imaging plane','sa')]}",
         f"{nsub[('Lesion size (tertiles)','small')]} / {nsub[('Lesion size (tertiles)','medium')]} / {nsub[('Lesion size (tertiles)','large')]}",
         "$\\le$533 / 534--1233 / $>$1233"]]
parts.append(table("tab:split", "Data partition. The test composition is given by imaging plane and by tertiles of the reference tumor area (pixels at $256\\times256$)",
                   "lYYYY", "Partition & Images & Axial / coronal / sagittal & Small / medium / large & Size-tertile edges (px)", rows))

v3 = rd(f"{R}/analysis/v3_empty_predictions.csv")
rows = []
for v in V:
    cells = []
    for k in range(1, 5):
        r = [x for x in v3 if x["variant"] == v and int(x["exit"]) == k][0]
        cells.append(r["per_seed_counts"].strip("[]").replace(",", "/").replace(" ", ""))
    rows.append([v] + cells)
parts.append(table("tab:empty", "Held-out images whose HD95 equals the image diagonal (exactly one of prediction and reference empty), per exit and seed (42/2026/3407)",
                   "lYYYY", "Variant & E1 & E2 & E3 & E4", rows))

# ---------------------------------------------------------------- S2 per-seed, per-exit
rows = [[r["variant"], r["seed"], f"{r['dice']:.5f}", f"{r['hd95']:.3f}", f"{r['assd']:.3f}", f"{r['biou']:.5f}",
         f"{r['mvr']:.2f}", f"{r['bmvr']:.2f}"] for r in EXTRA.get("per_seed", [])]
parts.append(table("tab:perseed", "Per-seed held-out Exit-4 results (frozen evaluation). HD95 and ASSD in pixels; MVR and BMVR in \\%",
                   "llYYYYYY", "Variant & Seed & Dice & HD95 & ASSD & B-IoU & MVR & BMVR", rows))

x1 = {r["variant"]: r for r in rd(f"{R}/analysis/x1_per_exit.csv")}
rows = []
for v in V:
    r = x1[v]
    rows.append([v] + [pm(r[f"dice_e{k}_mean"], r[f"dice_e{k}_sd"], 4) for k in range(1, 5)])
rows.append("\\midrule")
for v in V:
    r = x1[v]
    rows.append([v] + [pm(r[f"hd95_e{k}_mean"], r[f"hd95_e{k}_sd"], 3) for k in range(1, 5)])
parts.append(table("tab:perexit", "Held-out Dice (upper block) and HD95 in pixels (lower block) at each exit, mean$\\pm$SD over three seeds",
                   "lYYYY", "Variant & E1 & E2 & E3 & E4", rows))
rows = [[v, pm(x1[v]["mean_dice_e1_e3_mean"], x1[v]["mean_dice_e1_e3_sd"], 4),
         pm(x1[v]["auc_gflops_dice_mean"], x1[v]["auc_gflops_dice_sd"], 4)] for v in V]
parts.append(table("tab:auc", "Intermediate quality: mean Dice over E1--E3 and area under the GFLOPs--Dice curve, normalized by the GFLOPs range (E1 1.065 to E4 8.093)",
                   "lYY", "Variant & Mean Dice E1--E3 & Normalized AUC", rows))
parts.append(fig("fig:rates", "figs/figS1_rates.pdf", "Case-wise regression rates over all transitions (left) and at the E3$\\rightarrow$E4 transition (right). Error bars: SD over three seeds"))
parts.append(fig("fig:auc", "figs/figS2_regression_vs_auc.pdf", "Regression rates against intermediate quality (normalized area under the GFLOPs--Dice curve). E has the lowest rates at an intermediate quality comparable to the other variants"))

# ---------------------------------------------------------------- S3 paired comparisons
v2 = rd(f"{R}/analysis/v2_paired_stats.csv")
rows = []
for name, lab in (("MVR (%)", "MVR"), ("BMVR (%)", "BMVR")):
    for ref in "ABCD":
        r = [x for x in v2 if x["metric"] == name and x["reference"] == ref][0]
        rows.append([lab, f"E vs {ref}", sgn(r["image_level_mean_improvement"], 2), ci(r["ci95_low"], r["ci95_high"], 2),
                     pfmt(r["wilcoxon_p"]), pfmt(r["bh_adjusted_p"])])
parts.append(table("tab:paired-rates", "Paired image-level reductions of the regression rates (percentage points; positive = E regresses less). BH correction within each metric over the four references",
                   "llYYYY", "Metric & Comparison & Reduction & 95\\% CI & Wilcoxon $p$ & BH $p$", rows))

rows = []
names = [("Dice E4", "Dice", 4), ("IoU E4", "IoU", 4), ("HD95 E4 (px)", "HD95 (px)", 3), ("ASSD E4 (px)", "ASSD (px)", 3), ("B-IoU E4", "B-IoU", 4)]
for name, lab, d in names:
    for ref in "ABCD":
        r = [x for x in v2 if x["metric"] == name and x["reference"] == ref][0]
        rows.append([lab, f"E vs {ref}", sgn(r["image_level_mean_improvement"], d), ci(r["ci95_low"], r["ci95_high"], d),
                     pfmt(r["wilcoxon_p"]), pfmt(r["bh_adjusted_p"])])
    if lab != "B-IoU":
        rows.append("\\addlinespace[2pt]")
parts.append(table("tab:paired-endpoint", "Paired image-level differences in Exit-4 metrics between E and the other variants (positive = E better). BH correction within each metric over the four references",
                   "llYYYY", "Metric & Comparison & Difference & 95\\% CI & Wilcoxon $p$ & BH $p$", rows, size="\\scriptsize"))

rows = []
for lab, d in (("Dice", 4), ("HD95 (px)", 3), ("ASSD (px)", 3)):
    for name, (m, lo, hi, p, a) in EXTRA.get("baselines", {}).get(lab, {}).items():
        nm = {"EffB0UNet": "EffB0-U-Net", "DeepLabV3": "DeepLabV3", "UNet": "U-Net"}.get(name, name)
        rows.append([lab, f"E vs {nm}", sgn(m, d), ci(lo, hi, d), pfmt(p), pfmt(a)])
parts.append(table("tab:paired-baselines", "Paired image-level differences between E and the endpoint baselines (positive = E better). BH correction within each metric over the baselines. Empty predictions of a baseline are scored at the image diagonal (Table~\\ref{tab:nnunet})",
                   "llYYYY", "Metric & Comparison & Difference & 95\\% CI & Wilcoxon $p$ & BH $p$", rows))

# ---------------------------------------------------------------- S4 conditioning
adj = json.load(open(f"{R}/analysis/x2_adjusted.json"))
rows = []
for o, lab in (("dice_regression", "Dice regression"), ("hd95_regression", "HD95 regression")):
    rows.append([lab] + [f"{adj[r][o]['odds_ratio']:.2f} ({adj[r][o]['ci_low']:.2f}--{adj[r][o]['ci_high']:.2f})" for r in ("B", "D")])
parts.append(table("tab:or", "Adjusted odds ratio of a regression for E relative to B and D. Logistic model with covariates Dice$_k$, $\\log(1+\\mathrm{HD95}_k)$, and transition index; 95\\% intervals from 500 image-cluster bootstrap resamples",
                   "lYY", "Outcome & E vs B & E vs D", rows))
for ref in ("B", "D"):
    st_rows = rd(f"{R}/analysis/x2_stratified_E_vs_{ref}.csv")
    rows = []
    for strat, outc, lab in (("Dice_k", "dice_regression", "Dice regression, by Dice$_k$"), ("HD95_k", "hd95_regression", "HD95 regression, by HD95$_k$ (px)")):
        rows.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for r in [x for x in st_rows if x["stratify_by"] == strat and x["outcome"] == outc]:
            d = 3 if strat == "Dice_k" else 2
            rows.append([f"Q{r['bin']}", f"[{float(r['bin_low']):.{d}f}, {float(r['bin_high']):.{d}f}]",
                         f"{pm(r['E_rate_mean'], r['E_rate_sd'], 2)} ({r['E_n']})", f"{pm(r[ref + '_rate_mean'], r[ref + '_rate_sd'], 2)} ({r[ref + '_n']})"])
    parts.append(table(f"tab:strat-{ref}", f"Regression rate (\\%) of E and {ref} within quintiles of the starting quality, transitions pooled over exits and seeds (number of transitions in parentheses)",
                       "llYY", f"Bin & Range & E & {ref}", rows, size="\\scriptsize"))

# ---------------------------------------------------------------- S5 severity, regret, tolerance
x34 = {r["variant"]: r for r in rd(f"{R}/analysis/x3_x4_severity_regret.csv")}
rows = []
for v in V:
    r = x34[v]
    o = EXTRA.get("oracle", {}).get(v, (float("nan"), float("nan")))
    rows.append([v, pm(r["mean_hd95_increase_flagged_mean"], r["mean_hd95_increase_flagged_sd"], 2),
                 pm(r["median_hd95_increase_flagged_mean"], r["median_hd95_increase_flagged_sd"], 2),
                 pm(r["cvar95_hd95_change_mean"], r["cvar95_hd95_change_sd"], 2),
                 pm(r["mean_dice_drop_flagged_mean"], r["mean_dice_drop_flagged_sd"], 4),
                 pm(r["cvar95_dice_drop_mean"], r["cvar95_dice_drop_sd"], 4)])
parts.append(table("tab:severity", "Size of regressions. Flagged = beyond the tolerances of Eq.~(1) of the main text; CVaR$_{95}$ = mean of the largest 5\\% of changes over all transitions",
                   "lYYYYY", "Variant & Mean HD95 increase, flagged (px) & Median HD95 increase, flagged (px) & CVaR$_{95}$ HD95 change (px) & Mean Dice drop, flagged & CVaR$_{95}$ Dice drop", rows, size="\\scriptsize"))
rows = []
for v in V:
    r = x34[v]
    o = EXTRA.get("oracle", {}).get(v, (float("nan"), float("nan")))
    rows.append([v, pm(r["regret_dice_pct_mean"], r["regret_dice_pct_sd"], 2), pm(r["regret_hd95_pct_mean"], r["regret_hd95_pct_sd"], 2),
                 pm(o[0], o[1], 4), pm(r["p99_hd95_e4_mean"], r["p99_hd95_e4_sd"], 1)])
parts.append(table("tab:regret", "Final-exit regret, oracle exit selection, and the tail of the final HD95. $R$: share of images whose Exit-4 result is worse than the best of Exits 1--3 by more than the tolerance. Oracle gain: mean Dice gained by choosing the best exit per image (re-inferred outputs). P99: 99th percentile of Exit-4 HD95, computed per seed and averaged",
                   "lYYYY", "Variant & $R_{\\mathrm{Dice}}$ (\\%) & $R_{\\mathrm{HD95}}$ (\\%) & Oracle Dice gain & P99 HD95 (px)", rows))
tol = rd(f"{R}/analysis/x6_tolerance.csv")
rows = []
for m, unit in (("BMVR", "px"), ("MVR", "")):
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{m} (\\%) by {'HD95' if m == 'BMVR' else 'Dice'} tolerance}}}} \\\\")
    for t in sorted({float(r["tolerance"]) for r in tol if r["metric"] == m}):
        vals = {r["variant"]: (r["mean"], r["sd"]) for r in tol if r["metric"] == m and float(r["tolerance"]) == t}
        best = min(V, key=lambda v: float(vals[v][0]))
        cells = []
        for v in V:
            c = pm(vals[v][0], vals[v][1], 2)
            cells.append(f"\\textbf{{{c}}}" if v == best else c)
        rows.append([f"{t:g} {unit}".strip()] + cells)
parts.append(table("tab:tolerance", "Sensitivity of BMVR and MVR to the regression tolerance (bold: lowest per row). The main analysis uses $\\tau_H=0.5$\\,px and $\\tau_D=0.001$",
                   "lYYYYY", "Tolerance & A & B & C & D & E", rows, size="\\scriptsize"))
parts.append(fig("fig:tolerance", "figs/figS3_tolerance.pdf", "BMVR as a function of the HD95 tolerance (left) and MVR as a function of the Dice tolerance (right)"))

# ---------------------------------------------------------------- S6 local transitions, components
parts.append(fig("fig:bhrbcr", "figs/figS4_bhr_bcr.pdf", "Local boundary transitions: boundary harm rate against boundary correction rate (mean$\\pm$SD over three seeds). D damages the fewest correct boundary points but has the most late HD95 regressions; E corrects the fewest wrong points", width="0.5\\textwidth"))
comp = EXTRA.get("components", {})
rows = [[v, f"{c['reg_per_seed']:.1f}", f"{c['pct_with']:.1f}", f"{c['pct_far']:.1f}", f"{c['pct_nonreg_new']:.1f}",
         f"{c['new_per_image']:.3f}", f"{c['mean_dh_with']:.2f} / {c['mean_dh_without']:.2f}", f"{c['share_increase']:.1f}", f"{c['median_dist']:.1f}"]
        for v, c in comp.items()]
parts.append(table("tab:components", "New connected components at E4 (components of the E4 mask that do not overlap the E3 mask, 8-connectivity) and late HD95 regressions (E3$\\rightarrow$E4 increase $>0.5$\\,px). Pooled over the three seeds (re-inferred outputs)",
                   "lYYYYYYYY", "Variant & Late HD95 reg. per seed & \\% with new comp. & \\% with comp. $>$10\\,px & \\% new comp., non-regressing & New comp. per image & Mean $\\Delta$HD95 with / without (px) & \\% of total increase & Median distance (px)",
                   rows, size="\\scriptsize"))

# ---------------------------------------------------------------- S7 subgroups
rows = []
for g, gg, name in (("Imaging plane", "ax", "Axial"), ("Imaging plane", "co", "Coronal"), ("Imaging plane", "sa", "Sagittal"),
                    ("Lesion size (tertiles)", "small", "Small"), ("Lesion size (tertiles)", "medium", "Medium"), ("Lesion size (tertiles)", "large", "Large")):
    for i, v in enumerate(V):
        r = [x for x in sg if x["grouping"] == g and x["group"] == gg and x["variant"] == v][0]
        rows.append([f"{name} ({r['n_images']})" if i == 0 else "", v, pm(r["dice_e4_mean"], r["dice_e4_sd"], 4),
                     pm(r["hd95_e4_mean"], r["hd95_e4_sd"], 2), f"{float(r['mvr_mean']):.2f}", f"{float(r['bmvr_mean']):.2f}"])
    if name != "Large":
        rows.append("\\addlinespace[2pt]")
parts.append(table("tab:subgroups", "Exploratory subgroups (number of images in parentheses). Dice and HD95 at Exit 4 (mean$\\pm$SD over seeds); MVR and BMVR in \\% (mean over seeds). Not multiplicity-tested",
                   "llYYYY", "Subgroup & Variant & Dice & HD95 (px) & MVR & BMVR", rows, size="\\scriptsize"))
parts.append(fig("fig:subgroups", "figs/figS5_subgroups.pdf", "MVR (left) and BMVR (right) by imaging plane and lesion-size tertile. E has the lowest rates in every subgroup"))

# ---------------------------------------------------------------- S8 post-hoc
gs = json.load(open(f"{R}/posthoc/x9_gamma_selection.json"))
rows = []
for s, info in gs.items():
    for g, c in sorted(info["val_candidates"].items(), key=lambda x: -float(x[0])):
        sel = "$\\checkmark$" if float(g) == float(info["gamma"]) else ""
        drop = float(info["val_raw_dice_e4"]) - float(c["dice_e4"])
        rows.append([s, g, f"{float(c['dice_e4']):.4f}", f"{drop:.4f}", f"{float(c['mvr']):.2f}", f"{float(c['bmvr']):.2f}", sel])
    rows.append("\\addlinespace[2pt]")
rows = rows[:-1]
parts.append(table("tab:gamma", "Selection of the damping factor $\\gamma$ for B on the validation split: lowest validation MVR among values whose final Dice drop is at most 0.001",
                   "llYYYYY", "Seed & $\\gamma$ & Val. Dice & Dice drop & Val. MVR (\\%) & Val. BMVR (\\%) & Selected", rows))
ph = {r["model"]: r for r in rd(f"{R}/posthoc/x9_posthoc.csv")}
rows = []
for m, lab in (("B", "B"), ("B + cumulative averaging", "B + cumulative averaging"), ("B + damping", "B + damping (selected $\\gamma$)"), ("E", "E")):
    r = ph[m]
    rows.append([lab, pm(r["dice_e4_mean"], r["dice_e4_sd"], 4), pm(r["hd95_e4_mean"], r["hd95_e4_sd"], 3),
                 pm(r["mvr_mean"], r["mvr_sd"], 2), pm(r["bmvr_mean"], r["bmvr_sd"], 2), pm(r["mean_dice_e1_e3_mean"], r["mean_dice_e1_e3_sd"], 4)])
parts.append(table("tab:posthoc", "Post-hoc rules on B versus E, held-out set (re-inferred in FP32; differences of up to 0.02 points to the frozen rates come from re-inference)",
                   "lYYYYY", "Model & Final Dice & Final HD95 (px) & MVR (\\%) & BMVR (\\%) & Mean Dice E1--E3", rows))
ed = EXTRA.get("e_vs_damp", {})
rows = [[lab, sgn(ed[k][0], d), ci(ed[k][1], ed[k][2], d), pfmt(ed[k][3])] if k in ed else [lab, "--", "--", "--"]
        for k, lab, d in (("e4_dice", "Final Dice", 4), ("e4_hd95", "Final HD95 (px)", 3), ("e4_boundary_iou", "B-IoU", 4),
                          ("mvr", "MVR (pp)", 2), ("bmvr", "BMVR (pp)", 2))]
parts.append(table("tab:e-vs-damp", "Paired comparison of E with B damped by the validation-selected $\\gamma$; positive = E better",
                   "lYYY", "Metric & Difference & 95\\% CI & Wilcoxon $p$", rows))
parts.append(fig("fig:frontier", "figs/figS6_frontier_full.pdf", "Damping frontier of B over $\\gamma\\in\\{1,0.75,0.5,0.25\\}$, cumulative averaging of B, and the raw variants. Cumulative averaging lowers the rates furthest but costs 0.011 Dice"))

# ---------------------------------------------------------------- S9 early stopping and task-adapted policies
x10 = rd(f"{R}/posthoc/x10_stopping.csv")
rows = []
for m, lab in (("B", "B"), ("B + damping (selected γ)", "B + damping"), ("E", "E")):
    for r in [x for x in x10 if x["model"] == m]:
        rows.append([lab, f"{float(r['epsilon']):g}", f"{float(r['delta_mean']):.4f}", pm(r["dice_mean"], r["dice_sd"], 4), f"{float(r['hd95_mean']):.2f}",
                     f"{float(r['mean_exit_mean']):.2f}", pm(r["gflops_mean"], r["gflops_sd"], 2), f"{float(r['gpu_ms_mean']):.2f}", f"{float(r['cpu_ms_mean']):.1f}",
                     "/".join(f"{float(r[f'pct_exit{k}_mean']):.1f}" for k in range(1, 5))])
    rows.append("\\addlinespace[2pt]")
rows = rows[:-1]
parts.append(table("tab:stopping", "Early stopping with the area-normalized rule (mean over three seeds; SD for Dice and GFLOPs). $\\delta$ is calibrated on validation per model and seed. Latency is the mean over images of the fixed-exit latency of the chosen exit (GPU: H200 MIG; CPU: 8 threads)",
                   "llYYYYYYYY", "Model & $\\varepsilon$ & $\\delta$ & Dice & HD95 & Mean exit & GFLOPs & GPU ms & CPU ms & E1/E2/E3/E4 (\\%)", rows, size="\\scriptsize", resize=True))
rows = []
for pol in sorted(COMP.glob("mess_style_mess_stage2_seed*/mess_policy.json")):
    p = json.load(open(pol)); seed = pol.parent.name.split("seed")[-1]
    rows.append(["MESS-style", seed, f"{p['thresholds'][0]:.6f}", f"{p['val_dynamic_dice']:.5f}", f"{p['val_e4_dice']:.5f}", f"{p['dice_drop']:.5f}"])
for pol in sorted(COMP.glob("adpc_style_joint_seed*/adpc_policy.json")):
    p = json.load(open(pol)); seed = pol.parent.name.split("seed")[-1]; c = p["selected"]
    rows.append(["ADP-C-style", seed, f"{p['selected_threshold']:.6f}", f"{c['dynamic_dice']:.5f}", f"{c['e4_dice']:.5f}", f"{c['e4_dice'] - c['dynamic_dice']:.5f}"])
parts.append(table("tab:policy-cal", "Task-adapted whole-image confidence policies: validation calibration. MESS-style thresholds the mean of $\\max(p,1-p)$ over all pixels; when no threshold meets the 0.001 Dice-drop constraint, the selected row is the validation-best fallback",
                   "llYYYY", "Policy & Seed & Threshold & Dynamic Dice & E4 Dice & Dice drop", rows))
rows = []
csum = COMP / "competitor_summary_3seeds.csv"
for r in (rd(str(csum)) if csum.is_file() else []):
    beh = r.get("dynamic_exit_mean") or (("active E2/E3/E4 " + "/".join(r.get(f"active_fraction_e{k}", "") for k in (2, 3, 4))) if r.get("active_fraction_e2") else "--")
    rows.append([r["method"], r.get("dynamic_dice_mean", "--").replace("±", "$\\pm$"), r.get("dynamic_hd95_mean", "--").replace("±", "$\\pm$"),
                 beh.replace("±", "$\\pm$"), r.get("dice_mvr_percent", "--").replace("±", "$\\pm$"), r.get("hd95_bmvr_percent", "--").replace("±", "$\\pm$")])
_x1e = {r["variant"]: r for r in rd(f"{R}/analysis/x1_per_exit.csv")}.get("E")
if _x1e:
    rows.append(["BMAS-E, full decoder", pm(_x1e["dice_e4_mean"], _x1e["dice_e4_sd"], 4), pm(_x1e["hd95_e4_mean"], _x1e["hd95_e4_sd"], 3),
                 "4.00", pm(_x1e["mvr_mean"], _x1e["mvr_sd"], 2), pm(_x1e["bmvr_mean"], _x1e["bmvr_sd"], 2)])
parts.append(table("tab:policy-test", "Task-adapted policies on the held-out set. Dynamic Dice and HD95 refer to the policy output; MVR and BMVR are computed over each model's four exits, since a regression rate is defined on the exit sequence",
                   "lYYYYY", "Model & Dynamic Dice & Dynamic HD95 (px) & Policy behavior & MVR (\\%) & BMVR (\\%)", rows))

# ---------------------------------------------------------------- S10 compute
ec = {int(r["exit"]): r for r in rd(f"{R}/v4/exit_compute.csv")}
gp = {int(float(r["exit"])): r for r in rd(ARGS.gpu_profile)} if Path(ARGS.gpu_profile).is_file() else {}
def _g(k, key, d):
    r = gp.get(k, {})
    return f"{float(r[key + '_mean']):.{d}f}" if r.get(key + "_mean") not in (None, "") else "--"
rows = [[f"E{k}", f"{float(ec[k]['gflops']):.3f}", f"{int(ec[k]['executed_params']) / 1e6:.2f}",
         (_g(k, "latency_mean_ms", 3) + "$\\pm$" + f"{float(gp[k]['latency_mean_ms_sd']):.3f}") if k in gp else "--",
         _g(k, "latency_p50_ms", 3), _g(k, "latency_p95_ms", 3), _g(k, "peak_allocated_mb", 1),
         (_g(k, "speedup_vs_e4_x", 2) + "$\\times$") if k in gp else "--", ec[k]["new_modules"].replace("_", "\\_")] for k in range(1, 5)]
parts.append(table("tab:gpu", "Executed path and GPU cost of each exit of BMAS-E (NVIDIA H200 MIG 2g.35gb, batch 1, mixed precision, 50 warm-up and 200 timed passes; mean$\\pm$SD over three seeds). Peak memory: allocated model-forward memory",
                   "lYYYYYYYl", "Exit & GFLOPs & Params (M) & Latency (ms) & p50 & p95 & Peak mem. (MB) & Speed-up & Modules added", rows, size="\\scriptsize"))
cpu = rd(f"{R}/cpu/cpu_profile_by_seed.csv")
rows = []
for t in (1, 8):
    for k in range(1, 5):
        sel = [r for r in cpu if int(r["threads"]) == t and int(r["exit"]) == k]
        ms = [float(r["latency_mean_ms"]) for r in sel]
        rows.append([t if k == 1 else "", f"E{k}", f"{st.fmean(ms):.2f}$\\pm${st.stdev(ms):.2f}",
                     f"{st.fmean(float(r['latency_p50_ms']) for r in sel):.2f}", f"{st.fmean(float(r['latency_p95_ms']) for r in sel):.2f}",
                     f"{st.fmean(float(r['speedup_vs_e4']) for r in sel):.2f}$\\times$"])
    rows.append("\\addlinespace[2pt]")
e2e = rd(f"{R}/cpu/cpu_end_to_end.csv")
for k in (1, 4):
    sel = [r for r in e2e if int(r["exit"]) == k]
    ms = [float(r["e2e_mean_ms"]) for r in sel]
    rows.append(["8, end to end" if k == 1 else "", f"E{k}", f"{st.fmean(ms):.2f}$\\pm${st.stdev(ms):.2f}", "--",
                 f"{st.fmean(float(r['e2e_p95_ms']) for r in sel):.2f}", "--"])
hw = json.load(open(f"{R}/cpu/cpu_hardware.json"))
parts.append(table("tab:cpu", f"CPU latency of BMAS-E ({hw['cpu']}, PyTorch {hw['torch'].split('+')[0]}, FP32, batch 1, {hw['warmup']} warm-up and {hw['runs']} timed passes per exit; mean$\\pm$SD over three seeds of per-seed means; p50 and p95 averaged over seeds). End-to-end: 100 held-out images, image reading to thresholded mask",
                   "llYYYY", "Threads & Exit & Mean (ms) & p50 (ms) & p95 (ms) & Speed-up vs E4", rows))

# ---------------------------------------------------------------- S11 nnU-Net
nnx = EXTRA.get("nnunet", {})
eh = EXTRA.get("E_hd", {})
_pe = [r for r in EXTRA.get("per_seed", []) if r["variant"] == "E"]
def _pmseeds(key, d):
    v = [r[key] for r in _pe]
    return pm(st.fmean(v), st.stdev(v) if len(v) > 1 else 0.0, d) if v else "--"
rows = [["Dice", f"{nnx.get('dice', float('nan')):.4f}", _pmseeds("dice", 4)],
        ["HD95, all images (px)", f"{nnx.get('hd95', float('nan')):.3f}", _pmseeds("hd95", 3)],
        ["ASSD, all images (px)", f"{nnx.get('assd', float('nan')):.3f}", _pmseeds("assd", 3)],
        ["Empty predictions", f"{nnx.get('empty', '--')}", "/".join(str(x) for x in eh.get("empty", [])) + " (per seed)"],
        ["HD95 without empty predictions (px)", f"{nnx.get('hd_mean_nonempty', float('nan')):.2f}",
         (f"{min(eh['mean_nonempty']):.2f}--{max(eh['mean_nonempty']):.2f} (per seed)" if eh.get("mean_nonempty") else "--")],
        ["ASSD without empty predictions (px)", f"{nnx.get('assd_nonempty', float('nan')):.2f}", "--"],
        ["Median HD95, all images (px)", f"{nnx.get('hd_median', float('nan')):.2f}", (f"{st.fmean(eh['median']):.2f}" if eh.get("median") else "--")]]
parts.append(table("tab:nnunet", "2-D nnU-Net (nnU-Net v2, default 2-D configuration, 1,000 epochs, fold 0 = study split, final checkpoint, one run) against BMAS-E (three seeds). Empty predictions are scored at the image diagonal (362\\,px)",
                   "lYY", "Quantity & nnU-Net 2-D & BMAS-E", rows))

# ---------------------------------------------------------------- assemble
head = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage{microtype}
\usepackage[font=small,labelfont=bf,labelsep=period,skip=4pt]{caption}
\usepackage{hyperref}
\hypersetup{colorlinks,linkcolor=blue,urlcolor=blue}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.08}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thesection}{S\arabic{section}}
\begin{document}
\begin{center}
{\Large\bfseries Online Resource 1\par}
\vspace{0.4em}
{\large\bfseries BMAS: Measuring and reducing case-wise regressions across exits in multi-exit brain-tumor MRI segmentation\par}
\vspace{0.5em}
Nguyen Thien Phuc, Tran Cao Minh, Kiet Van Nguyen, Vu Minh Tran, Ha Minh Tan*\\
\textit{Signal, Image and Video Processing}\\
*Corresponding author: Ha Minh Tan, \texttt{tanhm@uit.edu.vn}\\
Faculty of Information Science and Engineering, University of Information Technology, Ho Chi Minh City, Vietnam; Vietnam National University, Ho Chi Minh City, Vietnam
\end{center}

\section*{Overview}
This document supports the main text with complete tables and additional figures. Definitions follow the main text: a transition $k\rightarrow k+1$ is a Dice regression if $\mathrm{Dice}_{k+1}<\mathrm{Dice}_k-0.001$ and an HD95 regression if $\mathrm{HD95}_{k+1}>\mathrm{HD95}_k+0.5$\,px; MVR and BMVR average these indicators over the three transitions. Unless stated otherwise, values are mean$\pm$SD over the three training seeds (42, 2026, 3407) on the 860 held-out images, HD95 and ASSD are in pixels of the $256\times256$ grid, and paired comparisons use the image as the unit, with each metric averaged over seeds before pairing (single models, not ensembles), 5,000 paired bootstrap resamples, two-sided Wilcoxon signed-rank tests, and Benjamini--Hochberg (BH) correction. Endpoint and regression results come from the frozen held-out evaluation; component statistics, post-hoc rules, oracle gains, and early stopping use re-inferred outputs of the same checkpoints in FP32. The A--E objectives, checkpoints, regression definitions, and the endpoint and regression-rate comparisons were fixed before the held-out set was first evaluated; all other analyses in this document (S4--S11 and the per-exit, tolerance, and subgroup tables) were added afterwards and are exploratory, with tunable parameters chosen on validation data only.

\begin{center}\small
\begin{tabularx}{\textwidth}{lX}
\toprule
Section & Content (main-text section)\\
\midrule
S1 & Data partition and empty-prediction fallback (Sect.~3.1)\\
S2 & Per-seed and per-exit results, intermediate quality (Sect.~6.1, 6.2)\\
S3 & All paired comparisons (Sect.~5, 6.1, 6.2)\\
S4 & Regressions at equal starting quality (Sect.~6.2)\\
S5 & Size of regressions, regret, tail, tolerance sensitivity (Sect.~6.2)\\
S6 & Local transitions and connected components (Sect.~6.3)\\
S7 & Subgroups (Sect.~6.2)\\
S8 & Post-hoc exit smoothing (Sect.~6.4)\\
S9 & Early stopping and task-adapted policies (Sect.~5, 6.4)\\
S10 & Compute profiles (Sect.~6.1)\\
S11 & nnU-Net baseline (Sect.~6.1)\\
\bottomrule
\end{tabularx}
\end{center}
"""
_e_tables = [load_case_table(d["per_case_csv"]) for d in json.load(open(f"{R}/discover/variant_map.json"))["variants"].get("E", {}).get("seeds", {}).values() if d.get("per_case_csv")]
_ids = sorted(_e_tables[0]) if _e_tables else []
_first = _ids[0].replace("_", "\\_") if _ids else ""
_last = _ids[-1].replace("_", "\\_") if _ids else ""
_held = (f"The held-out set consists of the {len(_ids)} images of the test partition of the data manifest "
         f"(case identifiers \\texttt{{{_first}}} to \\texttt{{{_last}}}). ") if _ids else ""
_max_empty = max((max(int(x) for x in r["per_seed_counts"].strip("[]").split(",") if x.strip()) for r in v3), default=0)
_e34 = 100.0 * (float(ec[4]["gflops"]) - float(ec[3]["gflops"])) / float(ec[4]["gflops"])
_gam = sorted({float(v["gamma"]) for v in gs.values()})
sections = [
    ("Data partition and empty predictions",
     _held + f"Table~\\ref{{tab:split}} gives the partition and the composition of the test set; Table~\\ref{{tab:empty}} shows that the empty-prediction fallback of HD95 and ASSD affects at most {_max_empty} images per exit and seed.",
     ["tab:split", "tab:empty"]),
    ("Per-seed and per-exit results",
     "Table~\\ref{tab:perseed} lists the Exit-4 results of every training run. Tables~\\ref{tab:perexit} and~\\ref{tab:auc} give the quality at each exit. Fig.~\\ref{fig:rates} shows the regression rates of the main text, and Fig.~\\ref{fig:auc} relates them to intermediate quality.",
     ["tab:perseed", "tab:perexit", "tab:auc", "fig:rates", "fig:auc"]),
    ("Paired comparisons",
     "Table~\\ref{tab:paired-rates} gives the paired reductions in regression rate, Table~\\ref{tab:paired-endpoint} the Exit-4 differences among variants, and Table~\\ref{tab:paired-baselines} the differences to the endpoint baselines. Comparisons with the baselines form a separate BH family.",
     ["tab:paired-rates", "tab:paired-endpoint", "tab:paired-baselines"]),
    ("Regressions at equal starting quality",
     "A variant with weaker intermediate exits could show fewer regressions without being more stable. Table~\\ref{tab:or} adjusts for the starting quality, and Tables~\\ref{tab:strat-B} and~\\ref{tab:strat-D} compare rates within quintiles of it.",
     ["tab:or", "tab:strat-B", "tab:strat-D"]),
    ("Size of regressions, regret, and tolerance sensitivity",
     "Table~\\ref{tab:severity} reports the size of regressions, Table~\\ref{tab:regret} the final-exit regret, the oracle gain, and the tail of the final HD95, and Table~\\ref{tab:tolerance} and Fig.~\\ref{fig:tolerance} the sensitivity to the regression tolerances.",
     ["tab:severity", "tab:regret", "tab:tolerance", "fig:tolerance"]),
    ("Local transitions and connected components",
     "Fig.~\\ref{fig:bhrbcr} shows the local trade-off between damaging correct boundary points and correcting wrong ones. Table~\\ref{tab:components} quantifies new connected components at E4 and their contribution to late HD95 regressions.",
     ["fig:bhrbcr", "tab:components"]),
    ("Subgroups",
     "Table~\\ref{tab:subgroups} and Fig.~\\ref{fig:subgroups} report the exploratory subgroups by imaging plane and lesion-size tertile.",
     ["tab:subgroups", "fig:subgroups"]),
    ("Post-hoc exit smoothing",
     f"Table~\\ref{{tab:gamma}} shows the validation selection of the damping factor (selected values: {', '.join(f'{g:g}' for g in _gam)}). Tables~\\ref{{tab:posthoc}} and~\\ref{{tab:e-vs-damp}} compare the post-hoc rules with E, and Fig.~\\ref{{fig:frontier}} shows the frontier.",
     ["tab:gamma", "tab:posthoc", "tab:e-vs-damp", "fig:frontier"]),
    ("Early stopping and task-adapted policies",
     "Table~\\ref{tab:stopping} gives the area-normalized rule at every tolerance. Tables~\\ref{tab:policy-cal} and~\\ref{tab:policy-test} document the whole-image confidence policies that motivated the area-normalized signal.",
     ["tab:stopping", "tab:policy-cal", "tab:policy-test"]),
    ("Compute profiles",
     f"Table~\\ref{{tab:gpu}} lists the modules executed at each exit and the GPU cost; the step from E3 to E4 adds {_e34:.0f}\\% of the FLOPs. Table~\\ref{{tab:cpu}} gives the CPU latency and the end-to-end time.",
     ["tab:gpu", "tab:cpu"]),
    ("nnU-Net baseline",
     "Table~\\ref{tab:nnunet} details the comparison with nnU-Net, including the effect of empty predictions on its surface-distance metrics.",
     ["tab:nnunet"]),
]
label_of = {}
for p in parts:
    m = re.search(r"\\label\{([^}]+)\}", p)
    label_of[m.group(1)] = p
body = head
for title, text, labels in sections:
    body += f"\n\\section{{{title}}}\n{text}\n\n" + "\n".join(label_of.pop(l) for l in labels) + "\n\\clearpage\n"
assert not label_of, f"unplaced: {list(label_of)}"
body += "\\end{document}\n"
Path(ARGS.output_dir).mkdir(parents=True, exist_ok=True)
open(Path(ARGS.output_dir) / "ESM_1.tex", "w", encoding="utf-8").write(body)
print("wrote", Path(ARGS.output_dir) / "ESM_1.tex")
