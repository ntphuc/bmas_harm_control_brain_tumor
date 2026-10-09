"""X9 (post-hoc exit smoothing) and X10 (early stopping) from inference outputs.

X9  For each seed, gamma of the damping rule is chosen on validation: lowest
    validation MVR among gammas whose final Dice is within --max-dice-drop of
    the undamped model. Cumulative averaging has no parameter. Test results are
    compared with Variant E.
X10 One stopping rule for every model: stop at the first exit k in {1,2,3}
    whose uncertain-pixel ratio is below delta, otherwise run E4. delta is
    calibrated on validation (largest compute saving with mean Dice within
    epsilon of that model's E4) and then applied once to the test split.

Pure Python.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from experiments.common import (
    EXITS, PAPER_GFLOPS_FALLBACK, load_case_table, md_table, mean, percentile, pm, read_csv_dicts,
    regression_rates, sd, to_float, write_csv, write_json, write_md,
)


def load(infer_root: Path, variant: str, seed: int, split: str, transform: str):
    p = infer_root / variant / f"seed{seed}" / split / f"per_case_{transform}.csv"
    return load_case_table(p) if p.is_file() else None


def summary(table, ids=None) -> Dict[str, float]:
    ids = ids or list(table)
    r = regression_rates(table, ids)
    return {
        "dice_e4": mean(table[c]["e4_dice"] for c in ids),
        "hd95_e4": mean(table[c]["e4_hd95"] for c in ids),
        "mvr": r["mvr"], "bmvr": r["bmvr"],
        "mean_dice_e1_e3": mean(mean(table[c][f"e{k}_dice"] for c in ids) for k in (1, 2, 3)),
    }


def per_exit_costs(exit_compute: str, gpu_profile: str, cpu_profile: str) -> Dict[str, Dict[int, float]]:
    costs: Dict[str, Dict[int, float]] = {"gflops": dict(PAPER_GFLOPS_FALLBACK)}
    if exit_compute and Path(exit_compute).is_file():
        costs["gflops"] = {int(float(r["exit"])): float(r["gflops"]) for r in read_csv_dicts(exit_compute)}
    if gpu_profile and Path(gpu_profile).is_file():
        rows = read_csv_dicts(gpu_profile)
        costs["gpu_ms"] = {int(float(r["exit"])): float(r["latency_mean_ms_mean"]) for r in rows
                           if to_float(r.get("latency_mean_ms_mean")) is not None}
    if cpu_profile and Path(cpu_profile).is_file():
        rows = read_csv_dicts(cpu_profile)
        best_threads = max(int(float(r["threads"])) for r in rows)
        costs["cpu_ms"] = {int(float(r["exit"])): float(r["latency_mean_ms_mean"]) for r in rows
                           if int(float(r["threads"])) == best_threads}
        costs["cpu_threads"] = {0: best_threads}
    return costs


# ----------------------------------------------------------------------- X9
def x9(infer_root: Path, seeds: List[int], base: str, target: str, gammas: List[float], max_drop: float):
    selection, per_model = {}, {}
    for s in seeds:
        val_raw = load(infer_root, base, s, "val", "raw")
        if val_raw is None:
            continue
        ref = summary(val_raw)["dice_e4"]
        cands = []
        for g in gammas:
            t = load(infer_root, base, s, "val", f"damp{g:g}")
            if t is None:
                continue
            sm = summary(t)
            cands.append((g, sm, ref - sm["dice_e4"] <= max_drop))
        feas = [c for c in cands if c[2]]
        if feas:
            g_sel = sorted(feas, key=lambda c: (c[1]["mvr"], -c[0]))[0][0]
            ok = True
        elif cands:
            g_sel = sorted(cands, key=lambda c: (ref - c[1]["dice_e4"], c[1]["mvr"]))[0][0]
            ok = False
        else:
            continue
        selection[s] = {"gamma": g_sel, "constraint_met": ok, "val_raw_dice_e4": ref,
                        "val_candidates": {f"{c[0]:g}": c[1] for c in cands}}
        for name, var, tr in ((base, base, "raw"), (f"{base} + cumulative averaging", base, "cumavg"),
                              (f"{base} + damping", base, f"damp{g_sel:g}"), (target, target, "raw")):
            t = load(infer_root, var, s, "test", tr)
            if t is not None:
                per_model.setdefault(name, []).append(summary(t))
    rows, md_rows = [], []
    for name, sums in per_model.items():
        row = {"model": name, "n_seeds": len(sums)}
        for k in ("dice_e4", "hd95_e4", "mvr", "bmvr", "mean_dice_e1_e3"):
            row[f"{k}_mean"], row[f"{k}_sd"] = mean(x[k] for x in sums), sd(x[k] for x in sums)
        rows.append(row)
        md_rows.append([name, pm([x["dice_e4"] for x in sums], 4), pm([x["hd95_e4"] for x in sums], 3),
                        pm([x["mvr"] for x in sums], 2), pm([x["bmvr"] for x in sums], 2),
                        pm([x["mean_dice_e1_e3"] for x in sums], 4)])
    md = md_table(["Model", "Final Dice", "Final HD95 (px)", "MVR (%)", "BMVR (%)", "Mean Dice E1–E3"], md_rows)
    md += "\n\nSelected γ per seed: " + ", ".join(
        f"seed {s}: γ={v['gamma']:g}{'' if v['constraint_met'] else ' (constraint not met; closest)'}"
        for s, v in selection.items())
    return rows, md, selection


# ---------------------------------------------------------------------- X10
def dynamic(table, delta: float, ids=None):
    ids = ids or list(table)
    exits, dice, hd = [], [], []
    for c in ids:
        rec = table[c]
        k_sel = 4
        for k in (1, 2, 3):
            u = rec.get(f"e{k}_uncert")
            if u is not None and u < delta:
                k_sel = k
                break
        exits.append(k_sel)
        dice.append(rec[f"e{k_sel}_dice"])
        hd.append(rec[f"e{k_sel}_hd95"])
    return exits, mean(dice), mean(hd)


def calibrate(val_table, eps: float, gflops: Dict[int, float]):
    e4 = mean(val_table[c]["e4_dice"] for c in val_table)
    us = [val_table[c][f"e{k}_uncert"] for c in val_table for k in (1, 2, 3)
          if val_table[c].get(f"e{k}_uncert") is not None]
    grid = sorted({0.0} | {percentile(us, q / 2) for q in range(0, 201)})
    grid = [d for d in grid if d <= 1.0]
    best = None
    for d in grid:
        exits, dsc, _ = dynamic(val_table, d)
        if dsc >= e4 - eps:
            cost = mean(gflops[k] for k in exits)
            key = (cost, -dsc)
            if best is None or key < best[0]:
                best = (key, d, dsc)
    return (best[1], best[2], e4) if best else (0.0, e4, e4)


def x10(infer_root: Path, seeds: List[int], models: List[tuple], epsilons: List[float], costs):
    rows, md_rows = [], []
    for name, variant, transform_fn in models:
        for eps in epsilons:
            acc = []
            for s in seeds:
                tr = transform_fn(s)
                if tr is None:
                    continue
                val = load(infer_root, variant, s, "val", tr)
                test = load(infer_root, variant, s, "test", tr)
                if val is None or test is None or "e1_uncert" not in next(iter(test.values())):
                    continue
                delta, vdice, ve4 = calibrate(val, eps, costs["gflops"])
                exits, dsc, hd = dynamic(test, delta)
                r = {"delta": delta, "dice": dsc, "hd95": hd, "mean_exit": mean(exits),
                     "gflops": mean(costs["gflops"][k] for k in exits)}
                for key in ("gpu_ms", "cpu_ms"):
                    if key in costs:
                        r[key] = mean(costs[key][k] for k in exits)
                for k in EXITS:
                    r[f"pct_exit{k}"] = 100.0 * sum(1 for e in exits if e == k) / len(exits)
                acc.append(r)
            if not acc:
                continue
            row = {"model": name, "epsilon": eps, "n_seeds": len(acc)}
            for k in acc[0]:
                row[f"{k}_mean"], row[f"{k}_sd"] = mean(a[k] for a in acc), sd(a[k] for a in acc)
            rows.append(row)
            md_rows.append([name, eps, pm([a["dice"] for a in acc], 4), pm([a["hd95"] for a in acc], 3),
                            pm([a["mean_exit"] for a in acc], 2), pm([a["gflops"] for a in acc], 3),
                            pm([a.get("gpu_ms") for a in acc], 3) if "gpu_ms" in acc[0] else "n/a",
                            pm([a.get("cpu_ms") for a in acc], 2) if "cpu_ms" in acc[0] else "n/a",
                            " / ".join(f"{mean(a[f'pct_exit{k}'] for a in acc):.1f}" for k in EXITS)])
    hdr = ["Model", "ε", "Dice", "HD95 (px)", "Mean exit", "Mean GFLOPs", "Mean GPU latency (ms)",
           f"Mean CPU latency (ms, {costs.get('cpu_threads', {0: '?'})[0]} threads)", "% stopping at E1/E2/E3/E4"]
    return rows, md_table(hdr, md_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    ap.add_argument("--base", default="B")
    ap.add_argument("--target", default="E")
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--max-dice-drop", type=float, default=0.001)
    ap.add_argument("--epsilons", nargs="+", type=float, default=[0.005, 0.01, 0.02])
    ap.add_argument("--exit-compute", default="")
    ap.add_argument("--gpu-profile", default="")
    ap.add_argument("--cpu-profile", default="")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    root, out = Path(args.infer_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sections = []
    rows9, md9, sel = x9(root, args.seeds, args.base, args.target, args.gammas, args.max_dice_drop)
    write_csv(out / "x9_posthoc.csv", rows9)
    write_json(sel, out / "x9_gamma_selection.json")
    sections.append("## X9 — Post-hoc exit smoothing vs Variant E (Table 7)\n\n" +
                    (md9 if rows9 else "Not available: needs inference outputs for the base variant (val+test)."))

    costs = per_exit_costs(args.exit_compute, args.gpu_profile, args.cpu_profile)
    models = [
        (args.base, args.base, lambda s: "raw"),
        (f"{args.base} + cumulative averaging", args.base, lambda s: "cumavg"),
        (f"{args.base} + damping (selected γ)", args.base,
         lambda s: f"damp{sel[s]['gamma']:g}" if s in sel else None),
        (args.target, args.target, lambda s: "raw"),
    ]
    rows10, md10 = x10(root, args.seeds, models, args.epsilons, costs)
    write_csv(out / "x10_stopping.csv", rows10)
    sections.append(
        "## X10 — Early stopping with one confidence rule (Table 8)\n\n"
        "Rule: stop at the first exit k ≤ 3 whose ratio (uncertain pixels + 1)/(predicted tumour pixels + 1) "
        "is below δ; uncertain = probability in [0.4, 0.6]. δ is calibrated per model and seed on validation "
        "(minimum mean GFLOPs subject to mean Dice ≥ E4 Dice − ε), then applied once to the test split.\n\n"
        + (md10 if rows10 else "Not available: needs inference outputs (val+test) with uncertainty columns."))
    write_md(out / "POSTHOC_STOPPING.md", "\n\n".join(sections))
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
