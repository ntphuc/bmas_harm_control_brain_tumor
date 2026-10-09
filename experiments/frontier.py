"""Regression-rate / final-quality frontier under post-hoc logit damping.

For every variant with inference outputs, the raw exits (gamma = 1) and every
damped transform (``per_case_damp<g>.csv``) give one operating point:
final Dice, final HD95, Boundary-IoU, MVR, BMVR (mean ± SD over seeds).
If E's frontier lies below B's (lower MVR at the same final Dice), the training
objective adds something post-hoc damping cannot; if the frontiers coincide, it
does not. A matched comparison picks, for each variant and seed, the gamma whose
validation final Dice is closest to a common target and compares them on test
with paired image-level statistics.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from experiments.common import (
    bh_adjust, bootstrap_ci_mean, load_case_table, md_table, mean, pm, regression_rates, transition_flags,
    wilcoxon_signed_rank, write_csv, write_json, write_md,
)


def transforms_for(root: Path, variant: str, seed: int, split: str):
    d = root / variant / f"seed{seed}" / split
    out = {}
    for p in sorted(d.glob("per_case_*.csv")):
        name = p.stem.replace("per_case_", "")
        if name == "raw":
            out[1.0] = p
        else:
            m = re.fullmatch(r"damp([0-9.]+)", name)
            if m:
                out[float(m.group(1))] = p
    return out


def summarize(table):
    ids = list(table)
    r = regression_rates(table, ids)
    return {"dice": mean(table[c]["e4_dice"] for c in ids), "hd95": mean(table[c]["e4_hd95"] for c in ids),
            "biou": mean(table[c].get("e4_boundary_iou") for c in ids), "mvr": r["mvr"], "bmvr": r["bmvr"]}


def per_image(tables, c, metric):
    vals = []
    for t in tables:
        rec = t[c]
        if metric in ("mvr", "bmvr"):
            f = [transition_flags(rec, k)[0 if metric == "mvr" else 1] for k in (1, 2, 3)]
            vals.append(100.0 * sum(bool(x) for x in f) / 3.0)
        else:
            vals.append(rec[metric])
    return sum(vals) / len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-roots", nargs="+", required=True, help="one or more inference output roots")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E", "MESS", "ADPC"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    ap.add_argument("--reference", default="B", help="model whose raw validation Dice defines the targets")
    ap.add_argument("--comparisons", nargs="+", default=["E:B", "E:A", "B:A", "MESS:B"],
                    help="target:reference pairs compared at matched final Dice")
    ap.add_argument("--target-drops", nargs="+", type=float, default=[0.0005, 0.001, 0.002, 0.005],
                    help="final validation Dice drops (vs the reference's raw model) at which models are matched")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    roots = [Path(r) for r in args.infer_roots]

    def find(variant, seed, split):
        merged = {}
        for r in roots:
            merged.update(transforms_for(r, variant, seed, split))
        return merged

    rows, md_rows, curves = [], [], {}
    for v in args.variants:
        gammas = None
        for s in args.seeds:
            g = set(find(v, s, "test"))
            gammas = g if gammas is None else gammas & g
        for g in sorted(gammas or [], reverse=True):
            sums = [summarize(load_case_table(find(v, s, "test")[g])) for s in args.seeds]
            row = {"variant": v, "gamma": g}
            for k in ("dice", "hd95", "biou", "mvr", "bmvr"):
                row[f"{k}_mean"] = mean(x[k] for x in sums)
                row[f"{k}_sd"] = (lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / max(len(xs) - 1, 1)) ** 0.5)(
                    [x[k] for x in sums])
            rows.append(row)
            curves.setdefault(v, []).append(row)
            md_rows.append([v, f"{g:g}", pm([x["dice"] for x in sums], 4), pm([x["hd95"] for x in sums], 3),
                            pm([x["biou"] for x in sums], 4), pm([x["mvr"] for x in sums], 2),
                            pm([x["bmvr"] for x in sums], 2)])
    write_csv(out / "frontier.csv", rows)
    text = ["# Frontier: post-hoc damping strength vs regression rate (test)\n",
            md_table(["Variant", "γ (1 = raw)", "Final Dice", "Final HD95 (px)", "B-IoU", "MVR %", "BMVR %"], md_rows)]

    # matched comparisons: for each target Dice drop, gamma is chosen per model and seed on
    # validation so that final Dice is closest to (reference raw Dice - drop); test is paired.
    selection = {}
    cmp_rows = []
    for drop in args.target_drops:
        sel = {}
        for s in args.seeds:
            ref_raw = find(args.reference, s, "val").get(1.0)
            if ref_raw is None:
                continue
            goal = summarize(load_case_table(ref_raw))["dice"] - drop
            for v in args.variants:
                cands = find(v, s, "val")
                if not cands:
                    continue
                vals = {g: summarize(load_case_table(cands[g]))["dice"] for g in cands}
                sel.setdefault(v, {})[s] = min(vals, key=lambda g: abs(vals[g] - goal))
        selection[f"{drop:g}"] = {v: {str(s): g for s, g in d.items()} for v, d in sel.items()}
        for pair in args.comparisons:
            tgt, ref = pair.split(":")
            if tgt not in sel or ref not in sel or len(sel[tgt]) != len(sel[ref]):
                continue
            seeds = sorted(sel[tgt])
            t_tabs = [load_case_table(find(tgt, s, "test")[sel[tgt][s]]) for s in seeds]
            r_tabs = [load_case_table(find(ref, s, "test")[sel[ref][s]]) for s in seeds]
            ids = sorted(set(t_tabs[0]) & set(r_tabs[0]))
            for label, metric, higher in (("Final Dice", "e4_dice", True), ("Final HD95", "e4_hd95", False),
                                          ("B-IoU", "e4_boundary_iou", True), ("MVR", "mvr", False),
                                          ("BMVR", "bmvr", False)):
                d = []
                for c in ids:
                    a, b = per_image(t_tabs, c, metric), per_image(r_tabs, c, metric)
                    d.append(a - b if higher else b - a)
                lo, hi = bootstrap_ci_mean(d)
                _, pval, _ = wilcoxon_signed_rank(d)
                cmp_rows.append({"target_drop": drop, "target": tgt, "reference": ref,
                                 "gamma_target": "/".join(f"{sel[tgt][s]:g}" for s in seeds),
                                 "gamma_reference": "/".join(f"{sel[ref][s]:g}" for s in seeds),
                                 "metric": label, "mean_improvement": mean(d), "ci_low": lo, "ci_high": hi,
                                 "wilcoxon_p": pval})
    # BH within each metric across all (drop, pair) comparisons
    for metric in {r["metric"] for r in cmp_rows}:
        fam = [r for r in cmp_rows if r["metric"] == metric]
        for r, adj in zip(fam, bh_adjust([r["wilcoxon_p"] for r in fam])):
            r["bh_p"] = adj
    write_csv(out / "frontier_matched.csv", cmp_rows)
    write_json(selection, out / "frontier_selection.json")
    if cmp_rows:
        text.append("\n## Matched comparisons (γ chosen on validation; positive = target better)\n\n" + md_table(
            ["Dice drop target", "Comparison", "γ target / ref (per seed)", "Metric", "Mean improvement",
             "95% CI", "Wilcoxon p", "BH p"],
            [[f"{r['target_drop']:g}", f"{r['target']} vs {r['reference']}",
              f"{r['gamma_target']} / {r['gamma_reference']}", r["metric"], f"{r['mean_improvement']:+.4f}",
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", f"{r['wilcoxon_p']:.3g}", f"{r['bh_p']:.3g}"]
             for r in cmp_rows]))
        text.append("\nReading guide: if E vs B is positive for MVR/BMVR with CI above zero at matched Dice, E's "
                    "objective adds something damping alone cannot; if CIs straddle zero at every target, the two "
                    "frontiers coincide.")
    write_md(out / "FRONTIER.md", "\n".join(text))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, key, lab in ((axes[0], "mvr", "MVR (%)"), (axes[1], "bmvr", "BMVR (%)")):
            for v, pts in curves.items():
                pts = sorted(pts, key=lambda r: r["dice_mean"])
                xs = [r["dice_mean"] for r in pts]
                ys = [r[f"{key}_mean"] for r in pts]
                style = "-o" if len(pts) > 1 else "s"
                ax.plot(xs, ys, style, ms=4, label=v)
                for r in pts:
                    if len(pts) > 1:
                        ax.annotate(f"{r['gamma']:g}", (r["dice_mean"], r[f"{key}_mean"]), fontsize=6,
                                    textcoords="offset points", xytext=(3, 3))
            ax.set_xlabel("Final Dice (E4)")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"fig_frontier.{ext}", dpi=300)
    except Exception as exc:  # pragma: no cover
        print("plot skipped:", exc)
    print("\n".join(text))


if __name__ == "__main__":
    main()
