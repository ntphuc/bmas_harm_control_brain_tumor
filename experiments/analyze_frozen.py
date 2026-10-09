"""Revision analyses that need only the frozen held-out per-case CSVs.

X1  per-exit Dice/HD95 for every variant + area under the GFLOPs-Dice curve
X2  regression rates conditional on exit-k quality (stratified + adjusted logistic model)
X3  severity of regressions (mean size of flagged regressions, CVaR95)
X4  final-exit regret
X6  tolerance sweeps for BMVR (HD95) and MVR (Dice)
X8  subgroups by tumor type / imaging plane / lesion size (needs case metadata)
V2  image-level paired statistics, computed on seed-averaged per-image metrics
V3  number of empty predictions (HD95 fallback = image diagonal)
--  sanity check: recomputed Table 5 versus the manuscript

Pure Python (NumPy used for speed if present).
"""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.common import (
    BASELINES, BMAS_VARIANTS, COMPETITORS, DIAGONAL_256, DICE_TOL, EXITS, HD95_TOL, PAPER_GFLOPS_FALLBACK, PAPER_TABLE5,
    TRANSITIONS, align_case_ids, bh_adjust, bootstrap_ci_mean, cvar_upper, load_case_table, logistic_fit,
    md_table, mean, percentile, pm, read_csv_dicts, read_json, regression_rates, sd, to_float,
    transition_flags, wilcoxon_signed_rank, write_csv, write_json, write_md,
)


# ----------------------------------------------------------------- loading
def load_variants(variant_map: Dict[str, Any], seeds: Sequence[int]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    data: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for variant, info in variant_map["variants"].items():
        per_seed = {}
        for s in seeds:
            d = info["seeds"].get(str(s))
            if d and d.get("per_case_csv"):
                per_seed[s] = load_case_table(d["per_case_csv"])
        if per_seed:
            data[variant] = per_seed
    return data


def gflops_by_exit(path: Optional[str]) -> tuple[Dict[int, float], str]:
    if path and Path(path).is_file():
        rows = read_csv_dicts(path)
        g = {int(float(r["exit"])): float(r["gflops"]) for r in rows if r.get("gflops")}
        if len(g) == 4:
            return g, f"measured ({path})"
    return dict(PAPER_GFLOPS_FALLBACK), "manuscript Table 3 values (exit_compute.csv not available)"


# --------------------------------------------------------------------- X1
def x1_per_exit(data, ids, gflops) -> tuple[List[Dict[str, Any]], str]:
    rows = []
    md_rows = []
    g = [gflops[k] for k in EXITS]
    span = g[-1] - g[0]
    for variant in _ordered(data):
        per_seed_auc, per_seed_inter = [], []
        row: Dict[str, Any] = {"variant": variant, "n_seeds": len(data[variant])}
        for k in EXITS:
            dk = [mean(t[c].get(f"e{k}_dice") for c in ids[variant]) for t in data[variant].values()]
            hk = [mean(t[c].get(f"e{k}_hd95") for c in ids[variant]) for t in data[variant].values()]
            row[f"dice_e{k}_mean"], row[f"dice_e{k}_sd"] = mean(dk), sd(dk)
            row[f"hd95_e{k}_mean"], row[f"hd95_e{k}_sd"] = mean(hk), sd(hk)
            row[f"_dice_e{k}"], row[f"_hd95_e{k}"] = dk, hk
        for si in range(len(data[variant])):
            dice_curve = [row[f"_dice_e{k}"][si] for k in EXITS]
            auc = sum((g[i + 1] - g[i]) * (dice_curve[i] + dice_curve[i + 1]) / 2 for i in range(3)) / span
            per_seed_auc.append(auc)
            per_seed_inter.append(mean(dice_curve[:3]))
        row["auc_gflops_dice_mean"], row["auc_gflops_dice_sd"] = mean(per_seed_auc), sd(per_seed_auc)
        row["mean_dice_e1_e3_mean"], row["mean_dice_e1_e3_sd"] = mean(per_seed_inter), sd(per_seed_inter)
        rates = [regression_rates(t, ids[variant]) for t in data[variant].values()]
        for key in ("mvr", "bmvr"):
            row[f"{key}_mean"], row[f"{key}_sd"] = mean(r[key] for r in rates), sd(r[key] for r in rates)
        md_rows.append([variant] + [pm(row[f"_dice_e{k}"], 4) for k in EXITS]
                       + [pm(row[f"_hd95_e{k}"], 3) for k in EXITS]
                       + [pm(per_seed_inter, 4), pm(per_seed_auc, 4)])
        rows.append({k: v for k, v in row.items() if not k.startswith("_")})
    md = md_table(["Variant"] + [f"Dice E{k}" for k in EXITS] + [f"HD95 E{k} (px)" for k in EXITS]
                  + ["Mean Dice E1–E3", "AUC GFLOPs–Dice (norm.)"], md_rows)
    return rows, md


# --------------------------------------------------------------------- X2
def _transitions(variant_tables, ids, variant):
    """Yield (seed, case, k, dice_k, hd95_k, dice_viol, hd95_viol)."""
    for s, t in variant_tables.items():
        for c in ids:
            rec = t[c]
            for k in TRANSITIONS:
                dv, hv = transition_flags(rec, k)
                dk, hk = rec.get(f"e{k}_dice"), rec.get(f"e{k}_hd95")
                if None in (dv, hv, dk, hk):
                    continue
                yield s, c, k, dk, hk, dv, hv


def x2_stratified(data, ids, compare: Sequence[str], n_bins: int = 5) -> tuple[List[Dict[str, Any]], str]:
    pooled_d, pooled_h = [], []
    for v in compare:
        for _, _, _, dk, hk, _, _ in _transitions(data[v], ids[v], v):
            pooled_d.append(dk)
            pooled_h.append(hk)
    edges_d = [percentile(pooled_d, 100 * i / n_bins) for i in range(n_bins + 1)]
    edges_h = [percentile(pooled_h, 100 * i / n_bins) for i in range(n_bins + 1)]

    def bin_of(x, edges):
        for b in range(n_bins):
            if x <= edges[b + 1] or b == n_bins - 1:
                return b
        return n_bins - 1

    rows, md_rows = [], []
    for kind, edges, key_idx, flag_idx in (("Dice_k", edges_d, 3, 5), ("HD95_k", edges_h, 4, 6)):
        for flag_name, fi in (("dice_regression", 5), ("hd95_regression", 6)):
            for b in range(n_bins):
                r: Dict[str, Any] = {"stratify_by": kind, "outcome": flag_name, "bin": b + 1,
                                     "bin_low": edges[b], "bin_high": edges[b + 1]}
                md = [kind, flag_name, f"Q{b+1} [{edges[b]:.3f}, {edges[b+1]:.3f}]"]
                for v in compare:
                    per_seed = defaultdict(list)
                    for tr in _transitions(data[v], ids[v], v):
                        if bin_of(tr[key_idx], edges) == b:
                            per_seed[tr[0]].append(float(tr[fi]))
                    rates = [100.0 * mean(x) for x in per_seed.values() if x]
                    n = sum(len(x) for x in per_seed.values())
                    r[f"{v}_rate_mean"], r[f"{v}_rate_sd"], r[f"{v}_n"] = mean(rates), sd(rates), n
                    md.append(f"{pm(rates, 2)} (n={n})")
                rows.append(r)
                md_rows.append(md)
    md = md_table(["Stratified by", "Outcome", "Bin"] + [f"{v} rate % (n transitions)" for v in compare], md_rows)
    return rows, md


def x2_adjusted(data, ids, target: str, reference: str, n_boot: int, seed: int = 7) -> Dict[str, Any]:
    """Logistic model: regression ~ is_target + Dice_k + log1p(HD95_k) + transition dummies.

    Images are resampled as clusters for the bootstrap CI because each image
    contributes several transitions and seeds.
    """
    common = sorted(set(ids[target]) & set(ids[reference]))
    by_case: Dict[str, List[tuple]] = defaultdict(list)
    for v, is_t in ((target, 1.0), (reference, 0.0)):
        for s, c, k, dk, hk, dv, hv in _transitions(data[v], common, v):
            x = [1.0, is_t, dk, math.log1p(hk), 1.0 if k == 2 else 0.0, 1.0 if k == 3 else 0.0]
            by_case[c].append((x, float(dv), float(hv)))

    def fit(case_list):
        X, yd, yh = [], [], []
        for c in case_list:
            for x, dv, hv in by_case[c]:
                X.append(x)
                yd.append(dv)
                yh.append(hv)
        return logistic_fit(X, yd)[1], logistic_fit(X, yh)[1]

    cases = list(by_case)
    bd, bh = fit(cases)
    rng = random.Random(seed)
    boots_d, boots_h = [], []
    for _ in range(n_boot):
        sample = [cases[rng.randrange(len(cases))] for _ in cases]
        d, h = fit(sample)
        boots_d.append(d)
        boots_h.append(h)

    def summarize(b, boots):
        lo, hi = percentile(boots, 2.5), percentile(boots, 97.5)
        return {"odds_ratio": math.exp(b), "ci_low": math.exp(lo), "ci_high": math.exp(hi), "log_or": b}

    return {
        "target": target, "reference": reference, "n_images": len(cases), "n_bootstrap": n_boot,
        "covariates": "Dice_k, log1p(HD95_k), transition (E1->E2 reference)",
        "dice_regression": summarize(bd, boots_d),
        "hd95_regression": summarize(bh, boots_h),
    }


# ------------------------------------------------------------------ X3 / X4
def x3_x4(data, ids) -> tuple[List[Dict[str, Any]], str]:
    rows, md_rows = [], []
    for v in _ordered(data):
        acc = defaultdict(list)
        for s, t in data[v].items():
            dh_all, dd_all, dh_flag, dd_flag = [], [], [], []
            e34_dh_flag = []
            reg_d = reg_h = 0
            n = 0
            hd95_e4 = []
            for c in ids[v]:
                rec = t[c]
                for k in TRANSITIONS:
                    d0, d1 = rec.get(f"e{k}_dice"), rec.get(f"e{k+1}_dice")
                    h0, h1 = rec.get(f"e{k}_hd95"), rec.get(f"e{k+1}_hd95")
                    if None in (d0, d1, h0, h1):
                        continue
                    dh, dd = h1 - h0, d0 - d1
                    dh_all.append(dh)
                    dd_all.append(dd)
                    if dh > HD95_TOL:
                        dh_flag.append(dh)
                        if k == 3:
                            e34_dh_flag.append(dh)
                    if dd > DICE_TOL:
                        dd_flag.append(dd)
                dice = [rec.get(f"e{k}_dice") for k in EXITS]
                hd = [rec.get(f"e{k}_hd95") for k in EXITS]
                if None not in dice and None not in hd:
                    n += 1
                    reg_d += dice[3] < max(dice[:3]) - DICE_TOL
                    reg_h += hd[3] > min(hd[:3]) + HD95_TOL
                    hd95_e4.append(hd[3])
            acc["mean_hd95_increase_flagged"].append(mean(dh_flag))
            acc["median_hd95_increase_flagged"].append(percentile(dh_flag, 50))
            acc["cvar95_hd95_change"].append(cvar_upper(dh_all, 0.95))
            acc["mean_e3e4_hd95_increase_flagged"].append(mean(e34_dh_flag))
            acc["mean_dice_drop_flagged"].append(mean(dd_flag))
            acc["cvar95_dice_drop"].append(cvar_upper(dd_all, 0.95))
            acc["regret_dice_pct"].append(100.0 * reg_d / max(n, 1))
            acc["regret_hd95_pct"].append(100.0 * reg_h / max(n, 1))
            acc["p99_hd95_e4"].append(percentile(hd95_e4, 99))
        row = {"variant": v, "n_seeds": len(data[v])}
        for key, vals in acc.items():
            row[f"{key}_mean"], row[f"{key}_sd"] = mean(vals), sd(vals)
        rows.append(row)
        md_rows.append([v, pm(acc["mean_hd95_increase_flagged"], 2), pm(acc["cvar95_hd95_change"], 2),
                        pm(acc["mean_dice_drop_flagged"], 4), pm(acc["cvar95_dice_drop"], 4),
                        pm(acc["regret_dice_pct"], 2), pm(acc["regret_hd95_pct"], 2), pm(acc["p99_hd95_e4"], 2)])
    md = md_table(["Variant", "Mean HD95 increase, flagged (px)", "CVaR95 HD95 change (px)",
                   "Mean Dice drop, flagged", "CVaR95 Dice drop", "R(Dice) %", "R(HD95) %",
                   "P99 HD95 E4 (px, check)"], md_rows)
    return rows, md


# --------------------------------------------------------------------- X6
def x6_tolerance(data, ids, hd_grid, dice_grid) -> tuple[List[Dict[str, Any]], str]:
    rows, md_lines = [], []
    variants = _ordered(data)
    hd_md, dice_md = [], []
    for tol in hd_grid:
        line = [f"{tol:g} px"]
        for v in variants:
            vals = [regression_rates(t, ids[v], DICE_TOL, tol)["bmvr"] for t in data[v].values()]
            rows.append({"metric": "BMVR", "tolerance": tol, "variant": v, "mean": mean(vals), "sd": sd(vals)})
            line.append(pm(vals, 2))
        hd_md.append(line)
    for tol in dice_grid:
        line = [f"{tol:g}"]
        for v in variants:
            vals = [regression_rates(t, ids[v], tol, HD95_TOL)["mvr"] for t in data[v].values()]
            rows.append({"metric": "MVR", "tolerance": tol, "variant": v, "mean": mean(vals), "sd": sd(vals)})
            line.append(pm(vals, 2))
        dice_md.append(line)
    md_lines.append("BMVR (%) by HD95 tolerance\n\n" + md_table(["τ_H"] + variants, hd_md))
    md_lines.append("MVR (%) by Dice tolerance\n\n" + md_table(["τ_D"] + variants, dice_md))
    return rows, "\n\n".join(md_lines)


# --------------------------------------------------------------------- X8
def load_metadata(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path or not Path(path).is_file():
        return {}
    return {r["case_id"]: r for r in read_csv_dicts(path)}


def x8_subgroups(data, ids, meta) -> tuple[List[Dict[str, Any]], str, str]:
    def attr(variant, c, key):
        m = meta.get(c, {})
        if m.get(key) not in (None, "", "unknown"):
            return m[key]
        for t in data[variant].values():
            if t[c].get(key) not in (None, "", "unknown"):
                return str(t[c][key])
        return None

    groups_spec = [("tumor_code", "Tumor type"), ("plane_code", "Imaging plane")]
    # lesion-size tertiles from ground-truth lesion pixels if available
    lesion = {}
    for v in data:
        for c in ids[v]:
            lp = to_float(attr(v, c, "lesion_pixels"))
            if lp is not None:
                lesion[c] = lp
    size_edges = None
    if lesion:
        vals = list(lesion.values())
        size_edges = (percentile(vals, 100 / 3), percentile(vals, 200 / 3))

    rows, md_rows = [], []
    status = []
    for key, label in groups_spec + ([("lesion_size", "Lesion size (tertiles)")] if size_edges else []):
        for v in _ordered(data):
            buckets: Dict[str, List[str]] = defaultdict(list)
            for c in ids[v]:
                if key == "lesion_size":
                    lp = lesion.get(c)
                    if lp is None:
                        continue
                    g = "small" if lp <= size_edges[0] else ("medium" if lp <= size_edges[1] else "large")
                else:
                    g = attr(v, c, key)
                    if g is None:
                        continue
                buckets[g].append(c)
            if not buckets:
                status.append(f"{label}: no metadata for variant {v}")
                continue
            for g, cs in sorted(buckets.items()):
                dice = [mean(t[c].get("e4_dice") for c in cs) for t in data[v].values()]
                hd = [mean(t[c].get("e4_hd95") for c in cs) for t in data[v].values()]
                mvr = [regression_rates(t, cs)["mvr"] for t in data[v].values()]
                bmvr = [regression_rates(t, cs)["bmvr"] for t in data[v].values()]
                rows.append({"grouping": label, "group": g, "variant": v, "n_images": len(cs),
                             "dice_e4_mean": mean(dice), "dice_e4_sd": sd(dice),
                             "hd95_e4_mean": mean(hd), "hd95_e4_sd": sd(hd),
                             "mvr_mean": mean(mvr), "bmvr_mean": mean(bmvr)})
                md_rows.append([label, g, v, len(cs), pm(dice, 4), pm(hd, 3), pm(mvr, 2), pm(bmvr, 2)])
    md = md_table(["Grouping", "Group", "Variant", "n", "Dice E4", "HD95 E4 (px)", "MVR %", "BMVR %"], md_rows)
    if size_edges:
        md += f"\n\nLesion-size tertile edges (GT pixels at 256×256): {size_edges[0]:.0f}, {size_edges[1]:.0f}."
    return rows, md, "\n".join(status)


# --------------------------------------------------------------------- V2
def per_image_seed_avg(tables: Dict[int, Dict[str, Any]], c: str, metric: str) -> Optional[float]:
    vals = []
    for t in tables.values():
        rec = t[c]
        if metric == "mvr_i":
            flags = [transition_flags(rec, k)[0] for k in TRANSITIONS]
            flags = [f for f in flags if f is not None]
            vals.append(100.0 * mean([float(f) for f in flags]) if flags else None)
        elif metric == "bmvr_i":
            flags = [transition_flags(rec, k)[1] for k in TRANSITIONS]
            flags = [f for f in flags if f is not None]
            vals.append(100.0 * mean([float(f) for f in flags]) if flags else None)
        else:
            vals.append(rec.get(metric))
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


V2_METRICS = [  # (name, key, higher_is_better)
    ("Dice E4", "e4_dice", True), ("IoU E4", "e4_iou", True), ("HD95 E4 (px)", "e4_hd95", False),
    ("ASSD E4 (px)", "e4_assd", False), ("B-IoU E4", "e4_boundary_iou", True),
    ("MVR (%)", "mvr_i", False), ("BMVR (%)", "bmvr_i", False),
]


def v2_paired(data, ids, target: str, references: Sequence[str], n_boot: int) -> tuple[List[Dict[str, Any]], str]:
    rows = []
    for name, key, higher in V2_METRICS:
        fam = []
        for ref in references:
            if ref not in data or target not in data:
                continue
            if key in ("mvr_i", "bmvr_i") and ref in BASELINES:
                continue
            common = sorted(set(ids[target]) & set(ids[ref]))
            diffs = []
            for c in common:
                a = per_image_seed_avg(data[target], c, key)
                b = per_image_seed_avg(data[ref], c, key)
                if a is None or b is None:
                    continue
                diffs.append((a - b) if higher else (b - a))
            if not diffs:
                continue
            # seed-level difference (the quantity printed in Tables 1/5)
            def seed_level(v):
                out = []
                for t in data[v].values():
                    if key == "mvr_i":
                        out.append(regression_rates(t, common)["mvr"])
                    elif key == "bmvr_i":
                        out.append(regression_rates(t, common)["bmvr"])
                    else:
                        out.append(mean(t[c].get(key) for c in common))
                return mean(out)
            sl = (seed_level(target) - seed_level(ref)) if higher else (seed_level(ref) - seed_level(target))
            lo, hi = bootstrap_ci_mean(diffs, n_boot)
            _, p, n_nz = wilcoxon_signed_rank(diffs)
            fam.append({"metric": name, "target": target, "reference": ref, "n_images": len(diffs),
                        "n_nonzero": n_nz, "image_level_mean_improvement": mean(diffs),
                        "seed_mean_improvement": sl, "ci95_low": lo, "ci95_high": hi, "wilcoxon_p": p})
        adj = bh_adjust([r["wilcoxon_p"] for r in fam]) if fam else []
        for r, a in zip(fam, adj):
            r["bh_adjusted_p"] = a
            r["bh_family"] = f"{name}: {target} vs {len(fam)} references"
        rows.extend(fam)
    md_rows = [[r["metric"], f"{r['target']} vs {r['reference']}", r["n_images"],
                f"{r['image_level_mean_improvement']:.5g}", f"{r['seed_mean_improvement']:.5g}",
                f"[{r['ci95_low']:.5g}, {r['ci95_high']:.5g}]", f"{r['wilcoxon_p']:.3g}", f"{r['bh_adjusted_p']:.3g}"]
               for r in rows]
    md = md_table(["Metric", "Comparison", "n", "Image-level mean improvement", "Seed-mean improvement",
                   "95% bootstrap CI", "Wilcoxon p", "BH-adjusted p"], md_rows)
    md += ("\n\nPositive = target better. Per-image values are averaged over seeds before pairing "
           "(metric averaging, not prediction averaging). For linear metrics the image-level and seed-mean "
           "improvements coincide by construction; a mismatch with the manuscript means its CIs were computed "
           "differently (e.g. on averaged predictions). BH is applied within each metric across references.")
    return rows, md


# --------------------------------------------------------------------- V3
def v3_empty(data, ids) -> tuple[List[Dict[str, Any]], str]:
    rows, md_rows = [], []
    for v in _ordered(data):
        for k in EXITS:
            counts = []
            for t in data[v].values():
                counts.append(sum(1 for c in ids[v]
                                  if t[c].get(f"e{k}_hd95") is not None
                                  and abs(t[c][f"e{k}_hd95"] - DIAGONAL_256) < 1e-3))
            rows.append({"variant": v, "exit": k, "per_seed_counts": counts, "mean": mean(counts)})
            md_rows.append([v, f"E{k}", ", ".join(str(c) for c in counts)])
    md = md_table(["Variant", "Exit", "Images with HD95 = diagonal (per seed)"], md_rows)
    md += (f"\n\nRule in bmas/metrics.py: if exactly one of prediction/reference is empty, HD95 = ASSD = "
           f"image diagonal ({DIAGONAL_256:.2f} px at 256×256); if both are empty, 0. Counts above detect the "
           "first case from the stored HD95 values.")
    return rows, md


# ------------------------------------------------------------ sanity check
def table5_check(data, ids) -> str:
    md_rows = []
    for v in _ordered(data):
        if v not in PAPER_TABLE5:
            continue
        rates = [regression_rates(t, ids[v]) for t in data[v].values()]
        rec = [mean(r["mvr"] for r in rates), mean(r["bmvr"] for r in rates),
               mean(r["dice_viol_e3e4"] for r in rates), mean(r["hd95_viol_e3e4"] for r in rates)]
        paper = PAPER_TABLE5[v]
        md_rows.append([v] + [f"{a:.2f} / {b:.2f}" for a, b in zip(rec, paper)]
                       + ["OK" if all(abs(a - b) < 0.02 for a, b in zip(rec, paper)) else "MISMATCH"])
    return md_table(["Variant", "MVR recomputed / paper", "BMVR", "E3→E4 Dice viol.", "E3→E4 HD95 viol.",
                     "Status"], md_rows)


def _ordered(data) -> List[str]:
    order = list(BASELINES) + list(BMAS_VARIANTS)
    return [v for v in order if v in data] + sorted(v for v in data if v not in order)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-map", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--exit-compute", default="")
    p.add_argument("--metadata", default="", help="case_metadata.csv from v1_split_match.py")
    p.add_argument("--target", default="E")
    p.add_argument("--x2-references", nargs="+", default=["B", "D"])
    p.add_argument("--x2-bootstrap", type=int, default=500)
    p.add_argument("--v2-bootstrap", type=int, default=5000)
    p.add_argument("--hd95-grid", nargs="+", type=float, default=[0, 0.25, 0.5, 1, 2, 3, 4, 5])
    p.add_argument("--dice-grid", nargs="+", type=float, default=[0, 0.0005, 0.001, 0.005, 0.01])
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    vmap = read_json(args.variant_map)
    seeds = [int(s) for s in vmap["seeds"]]
    data = load_variants(vmap, seeds)
    if not data:
        raise SystemExit("No per-case CSVs found for any variant; nothing to analyse.")
    ids = {v: align_case_ids(list(t.values())) for v, t in data.items()}
    for v in _ordered(data):
        print(f"[load] {v}: seeds={sorted(data[v])} cases={len(ids[v])}")

    gflops, gsrc = gflops_by_exit(args.exit_compute)
    sections = [f"# Frozen-output analyses\n\nGFLOPs per exit: {gsrc}: " +
                ", ".join(f"E{k}={gflops[k]:.3f}" for k in EXITS)]

    sections.append("## Sanity check — Table 5 recomputed from per-case CSVs\n\n" + table5_check(data, ids))

    bmas = {v: data[v] for v in data if v in BMAS_VARIANTS}
    if bmas:
        rows, md = x1_per_exit(bmas, ids, gflops)
        write_csv(out / "x1_per_exit.csv", rows)
        sections.append("## X1 — Quality at each exit (Table 2)\n\n" + md)

        rows, md = x3_x4(bmas, ids)
        write_csv(out / "x3_x4_severity_regret.csv", rows)
        sections.append("## X3 / X4 — Severity and final-exit regret (Table 6)\n\n" + md)

        rows, md = x6_tolerance(bmas, ids, args.hd95_grid, args.dice_grid)
        write_csv(out / "x6_tolerance.csv", rows)
        sections.append("## X6 — Tolerance sensitivity (Section 7.5)\n\n" + md)

        if args.target in bmas:
            x2_md, x2_json = [], {}
            for ref in args.x2_references:
                if ref not in bmas:
                    x2_md.append(f"Reference {ref} not available; skipped.")
                    continue
                rows, md = x2_stratified(bmas, ids, [args.target, ref])
                write_csv(out / f"x2_stratified_{args.target}_vs_{ref}.csv", rows)
                x2_md.append(f"### Stratified: {args.target} vs {ref}\n\n" + md)
                adj = x2_adjusted(bmas, ids, args.target, ref, args.x2_bootstrap)
                x2_json[ref] = adj
                x2_md.append(
                    f"### Adjusted odds ratio, {args.target} vs {ref}\n\n" + md_table(
                        ["Outcome", "OR", "95% CI (image-cluster bootstrap)"],
                        [[o, f"{adj[o]['odds_ratio']:.3f}", f"{adj[o]['ci_low']:.3f}–{adj[o]['ci_high']:.3f}"]
                         for o in ("dice_regression", "hd95_regression")])
                    + f"\n\nCovariates: {adj['covariates']}; n images = {adj['n_images']}; "
                      f"bootstrap = {adj['n_bootstrap']}. OR < 1 with CI below 1 means {args.target} regresses "
                      f"less often than {ref} at equal starting quality.")
            write_json(x2_json, out / "x2_adjusted.json")
            sections.append("## X2 — Conditioning on intermediate-exit quality (Section 7.2)\n\n" + "\n\n".join(x2_md))

        meta = load_metadata(args.metadata)
        rows, md, status = x8_subgroups(bmas, ids, meta)
        write_csv(out / "x8_subgroups.csv", rows)
        sections.append("## X8 — Subgroups (Section 7.8)\n\n" + (md if rows else "No metadata available.")
                        + (f"\n\n{status}" if status else ""))

        rows, md = v3_empty(bmas, ids)
        write_csv(out / "v3_empty_predictions.csv", rows)
        sections.append("## V3 — Empty predictions and the HD95 fallback\n\n" + md)

    comp = {v: data[v] for v in data if v in COMPETITORS}
    if comp:
        md_rows = []
        for v in comp:
            rates = [regression_rates(t, ids[v]) for t in comp[v].values()]
            dice = {k: [mean(t[c].get(f"e{k}_dice") for c in ids[v]) for t in comp[v].values()] for k in EXITS}
            md_rows.append([v, len(comp[v])] + [pm(dice[k], 4) for k in EXITS]
                           + [pm([r["mvr"] for r in rates], 2), pm([r["bmvr"] for r in rates], 2)])
        sections.append("## Independent-exit comparators — raw exits (verifies Online Resource Table S5)\n\n"
                        + md_table(["Model", "Seeds", "Dice E1", "Dice E2", "Dice E3", "Dice E4", "MVR %", "BMVR %"],
                                   md_rows)
                        + "\n\nRates are computed on the comparators' own exits (no stopping policy).")

    if args.target in data:
        families = [("BMAS variants", [v for v in BMAS_VARIANTS if v != args.target and v in data]),
                    ("endpoint baselines", [v for v in BASELINES if v in data]),
                    ("independent-exit comparators", [v for v in COMPETITORS if v in data])]
        all_rows, mds = [], []
        for fam_name, refs in families:
            if not refs:
                continue
            rows, md = v2_paired(data, ids, args.target, refs, args.v2_bootstrap)
            for r in rows:
                r["family"] = fam_name
            all_rows += rows
            mds.append(f"### {args.target} vs {fam_name} (BH within each metric of this family)\n\n" + md)
        write_csv(out / "v2_paired_stats.csv", all_rows)
        sections.append("## V2 — Image-level paired statistics (Sections 5.5, 6.1, 7.1)\n\n" + "\n\n".join(mds))

    write_md(out / "FROZEN_ANALYSES.md", "\n\n".join(sections))
    print(f"Wrote {out / 'FROZEN_ANALYSES.md'}")


if __name__ == "__main__":
    main()
