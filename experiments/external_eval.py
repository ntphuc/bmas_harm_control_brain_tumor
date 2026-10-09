"""X14: do the regression findings survive distribution shift?

Reads zero-shot inference outputs on the external set and reports, per model:
final Dice/HD95/B-IoU, MVR, BMVR and final-exit regret, for the raw exits and for
post-hoc damping with the gamma chosen on the BRISC validation split (no tuning on
the external data). Paired image-level comparisons with BH correction and the
adjusted odds ratio of Section 7.2 are repeated on the external images.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.analyze_frozen import x2_adjusted
from experiments.common import (
    DICE_TOL, HD95_TOL, bh_adjust, bootstrap_ci_mean, load_case_table, md_table, mean, pm, read_json,
    regression_rates, transition_flags, wilcoxon_signed_rank, write_csv, write_md,
)


def regret(table, ids):
    rd = rh = 0
    for c in ids:
        r = table[c]
        d = [r[f"e{k}_dice"] for k in (1, 2, 3, 4)]
        h = [r[f"e{k}_hd95"] for k in (1, 2, 3, 4)]
        rd += d[3] < max(d[:3]) - DICE_TOL
        rh += h[3] > min(h[:3]) + HD95_TOL
    return 100.0 * rd / len(ids), 100.0 * rh / len(ids)


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
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--selection", default="", help="frontier_selection.json from the BRISC frontier stage")
    ap.add_argument("--selection-drop", default="0.001")
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E", "MESS"])
    ap.add_argument("--damped", nargs="+", default=["B", "E"], help="models also reported with transferred damping")
    ap.add_argument("--comparisons", nargs="+",
                    default=["E:A", "E:B", "E:D", "E:B+damp", "E+damp:B+damp", "B+damp:B"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    ap.add_argument("--x2-bootstrap", type=int, default=300)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    root, out = Path(args.infer_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sel = {}
    if args.selection and Path(args.selection).is_file():
        sel = read_json(args.selection).get(args.selection_drop, {})

    def tables(model):
        base, damped = (model[:-5], True) if model.endswith("+damp") else (model, False)
        res = []
        for s in args.seeds:
            name = "raw"
            if damped:
                g = sel.get(base, {}).get(str(s))
                if g is None:
                    return [], None
                name = "raw" if float(g) == 1.0 else f"damp{float(g):g}"
            p = root / base / f"seed{s}" / "test" / f"per_case_{name}.csv"
            if not p.is_file():
                return [], None
            res.append(load_case_table(p))
        gam = "/".join(str(sel.get(base, {}).get(str(s), "")) for s in args.seeds) if damped else "1"
        return res, gam

    models = list(args.variants) + [f"{v}+damp" for v in args.damped]
    loaded = {}
    rows, md_rows = [], []
    for m in models:
        tabs, gam = tables(m)
        if not tabs:
            continue
        loaded[m] = tabs
        ids = list(tabs[0])
        stats = []
        for t in tabs:
            r = regression_rates(t, ids)
            rd, rh = regret(t, ids)
            stats.append({"dice": mean(t[c]["e4_dice"] for c in ids), "hd95": mean(t[c]["e4_hd95"] for c in ids),
                          "biou": mean(t[c].get("e4_boundary_iou") for c in ids), "mvr": r["mvr"], "bmvr": r["bmvr"],
                          "r_dice": rd, "r_hd95": rh})
        row = {"model": m, "gamma": gam, "n_images": len(ids)}
        for k in stats[0]:
            row[f"{k}_mean"] = mean(x[k] for x in stats)
        rows.append(row)
        md_rows.append([m, gam, len(ids)] + [pm([x[k] for x in stats], 4 if k in ("dice", "biou") else 2)
                                             for k in ("dice", "hd95", "biou", "mvr", "bmvr", "r_dice", "r_hd95")])
    if not rows:
        raise SystemExit(f"No external inference outputs under {root}")
    write_csv(out / "external_summary.csv", rows)
    text = ["# X14 — External set (zero-shot, distribution shift)\n",
            "Damping uses the γ chosen on BRISC validation (no tuning on the external data).\n",
            md_table(["Model", "γ (per seed)", "n", "Final Dice", "Final HD95 (px)", "B-IoU", "MVR %", "BMVR %",
                      "R(Dice) %", "R(HD95) %"], md_rows)]

    cmp_rows = []
    for pair in args.comparisons:
        tgt, ref = pair.split(":")
        if tgt not in loaded or ref not in loaded:
            continue
        ids = sorted(set(loaded[tgt][0]) & set(loaded[ref][0]))
        for label, metric, higher in (("Final Dice", "e4_dice", True), ("Final HD95", "e4_hd95", False),
                                      ("B-IoU", "e4_boundary_iou", True), ("MVR", "mvr", False),
                                      ("BMVR", "bmvr", False)):
            d = []
            for c in ids:
                a, b = per_image(loaded[tgt], c, metric), per_image(loaded[ref], c, metric)
                d.append(a - b if higher else b - a)
            lo, hi = bootstrap_ci_mean(d)
            _, pval, _ = wilcoxon_signed_rank(d)
            cmp_rows.append({"comparison": f"{tgt} vs {ref}", "metric": label, "mean_improvement": mean(d),
                             "ci_low": lo, "ci_high": hi, "wilcoxon_p": pval})
    for metric in {r["metric"] for r in cmp_rows}:
        fam = [r for r in cmp_rows if r["metric"] == metric]
        for r, adj in zip(fam, bh_adjust([r["wilcoxon_p"] for r in fam])):
            r["bh_p"] = adj
    write_csv(out / "external_paired.csv", cmp_rows)
    if cmp_rows:
        text.append("\n## Paired comparisons (positive = first model better; BH within each metric)\n\n" + md_table(
            ["Comparison", "Metric", "Mean improvement", "95% CI", "Wilcoxon p", "BH p"],
            [[r["comparison"], r["metric"], f"{r['mean_improvement']:+.4f}",
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]", f"{r['wilcoxon_p']:.3g}", f"{r['bh_p']:.3g}"]
             for r in cmp_rows]))

    if "E" in loaded and "B" in loaded:
        data = {"E": dict(zip(args.seeds, loaded["E"])), "B": dict(zip(args.seeds, loaded["B"]))}
        ids = {v: sorted(data[v][args.seeds[0]]) for v in data}
        adj = x2_adjusted(data, ids, "E", "B", args.x2_bootstrap)
        text.append("\n## Adjusted odds ratio, E vs B (as in Section 7.2)\n\n" + md_table(
            ["Outcome", "OR", "95% CI"],
            [[o, f"{adj[o]['odds_ratio']:.3f}", f"{adj[o]['ci_low']:.3f}–{adj[o]['ci_high']:.3f}"]
             for o in ("dice_regression", "hd95_regression")]))
    write_md(out / "EXTERNAL.md", "\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
