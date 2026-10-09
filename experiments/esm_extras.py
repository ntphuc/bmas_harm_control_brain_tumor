"""Statistics for Online Resource 1 that are not produced by the other stages.

Reads the frozen per-case CSVs (variant map), the inference outputs, and the
nnU-Net evaluation, and writes esm_extra.json with:
  per_seed      Exit-4 Dice/HD95/ASSD/B-IoU, MVR, BMVR for every run (Table S3)
  components    pooled new-component statistics at E3->E4 (Table S14)
  oracle        mean Dice gain of an oracle exit choice (Table S12)
  e_vs_damp     paired E vs B damped with the validation-selected gamma (Table S18)
  baselines     paired E vs endpoint baselines and nnU-Net, BH per metric (Table S8)
  nnunet, E_hd  empty-prediction and median-HD95 details (Table S22)
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

from experiments.common import (
    BASELINES, DIAGONAL_256, bh_adjust, bootstrap_ci_mean, load_case_table, mean, percentile, read_json,
    regression_rates, transition_flags, wilcoxon_signed_rank, write_json,
)

SEEDS = [42, 2026, 3407]


def per_image(tables, c, metric):
    vals = []
    for t in tables:
        if c not in t:
            continue
        rec = t[c]
        if metric in ("mvr", "bmvr"):
            f = [transition_flags(rec, k)[0 if metric == "mvr" else 1] for k in (1, 2, 3)]
            vals.append(100.0 * sum(bool(x) for x in f) / 3.0)
        else:
            vals.append(rec.get(metric))
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def paired(target_tabs, ref_tabs, metric, higher):
    ids = sorted(set(target_tabs[0]) & set(ref_tabs[0]))
    d = []
    for c in ids:
        a, b = per_image(target_tabs, c, metric), per_image(ref_tabs, c, metric)
        if a is not None and b is not None:
            d.append(a - b if higher else b - a)
    lo, hi = bootstrap_ci_mean(d)
    _, p, _ = wilcoxon_signed_rank(d)
    return mean(d), lo, hi, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-out", required=True, help="revision pipeline output folder")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root = Path(args.exp_out)
    vmap = read_json(root / "discover" / "variant_map.json")
    seeds = [int(s) for s in vmap.get("seeds", SEEDS)]
    out = {}

    frozen = {}
    for v, info in vmap["variants"].items():
        tabs = [load_case_table(d["per_case_csv"]) for d in info["seeds"].values() if d.get("per_case_csv")]
        if tabs:
            frozen[v] = tabs

    rows = []
    for v in "ABCDE":
        info = vmap["variants"].get(v, {}).get("seeds", {})
        for s in seeds:
            d = info.get(str(s))
            if not d or not d.get("per_case_csv"):
                continue
            t = load_case_table(d["per_case_csv"])
            ids = list(t)
            r = regression_rates(t, ids)
            rows.append({"variant": v, "seed": s, "dice": mean(t[c]["e4_dice"] for c in ids),
                         "hd95": mean(t[c]["e4_hd95"] for c in ids), "assd": mean(t[c].get("e4_assd") for c in ids),
                         "biou": mean(t[c].get("e4_boundary_iou") for c in ids), "mvr": r["mvr"], "bmvr": r["bmvr"]})
    out["per_seed"] = rows

    infer = root / "infer"

    def itab(v, s, tr="raw", split="test"):
        p = infer / v / f"seed{s}" / split / f"per_case_{tr}.csv"
        return load_case_table(p) if p.is_file() else None

    comp, oracle = {}, {}
    for v in "ABCDE":
        w, wo, nreg, newpi, far, nonreg_new, dists, gains = [], [], [], [], [], [], [], []
        for s in seeds:
            p = infer / v / f"seed{s}" / "test" / "per_case_raw.csv"
            if not p.is_file():
                continue
            raw = {r["case_id"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
            reg = fr = nn = nr = 0
            for r in raw.values():
                dh = float(r["e4_hd95"]) - float(r["e3_hd95"])
                nc = float(r.get("e34_new_cc") or 0)
                dists += [float(x) for x in (r.get("e34_new_cc_dists") or "").split(";") if x]
                if dh > 0.5:
                    reg += 1
                    (w if nc > 0 else wo).append(dh)
                    fr += float(r.get("e34_new_cc_far") or 0)
                else:
                    nr += 1
                    nn += nc > 0
            nreg.append(reg)
            newpi.append(mean(float(r.get("e34_new_cc") or 0) for r in raw.values()))
            far.append(100 * fr / max(reg, 1))
            nonreg_new.append(100 * nn / max(nr, 1))
            t = load_case_table(p)
            gains.append(mean(max(t[c][f"e{k}_dice"] for k in (1, 2, 3, 4)) - t[c]["e4_dice"] for c in t))
        if nreg:
            comp[v] = dict(reg_per_seed=st.fmean(nreg), pct_with=100 * len(w) / max(len(w) + len(wo), 1),
                           pct_far=st.fmean(far), pct_nonreg_new=st.fmean(nonreg_new), new_per_image=st.fmean(newpi),
                           mean_dh_with=mean(w), mean_dh_without=mean(wo),
                           share_increase=100 * sum(w) / max(sum(w) + sum(wo), 1e-9), median_dist=percentile(dists, 50))
            oracle[v] = (st.fmean(gains), st.stdev(gains) if len(gains) > 1 else 0.0)
    out["components"], out["oracle"] = comp, oracle

    sel_path = root / "posthoc" / "x9_gamma_selection.json"
    if sel_path.is_file():
        sel = read_json(sel_path)
        e = [itab("E", s) for s in seeds]
        bd = [itab("B", s, f"damp{float(sel[str(s)]['gamma']):g}") for s in seeds if str(s) in sel]
        if all(x is not None for x in e) and bd and all(x is not None for x in bd):
            out["e_vs_damp"] = {m: paired(e, bd, m, h) for m, h in (("e4_dice", True), ("e4_hd95", False),
                                                                   ("e4_boundary_iou", True), ("mvr", False), ("bmvr", False))}
            out["damp_gamma"] = {s: sel[s]["gamma"] for s in sel}

    nn_csv = root / "nnunet" / "eval" / "per_case_metrics.csv"
    refs = {b: frozen[b] for b in BASELINES if b in frozen}
    if nn_csv.is_file():
        refs["nnU-Net 2-D"] = [load_case_table(nn_csv)]
    if "E" in frozen and refs:
        base = {}
        for m, h, lab in (("e4_dice", True, "Dice"), ("e4_hd95", False, "HD95 (px)"), ("e4_assd", False, "ASSD (px)")):
            res = {name: paired(frozen["E"], tabs, m, h) for name, tabs in refs.items()}
            adj = bh_adjust([r[3] for r in res.values()])
            base[lab] = {name: list(r) + [a] for (name, r), a in zip(res.items(), adj)}
        out["baselines"] = base

    if nn_csv.is_file():
        rows = list(csv.DictReader(open(nn_csv, encoding="utf-8")))
        h = [float(r["e4_hd95"]) for r in rows]
        a = [float(r["e4_assd"]) for r in rows]
        keep = [i for i, x in enumerate(h) if abs(x - DIAGONAL_256) > 1e-3]
        out["nnunet"] = dict(dice=mean(float(r["e4_dice"]) for r in rows), hd95=mean(h), assd=mean(a),
                             empty=len(h) - len(keep), hd_mean_nonempty=mean(h[i] for i in keep),
                             assd_nonempty=mean(a[i] for i in keep), hd_median=st.median(h))
    if "E" in frozen:
        eh = {"median": [], "empty": [], "mean_nonempty": []}
        for t in frozen["E"]:
            hh = [t[c]["e4_hd95"] for c in t]
            eh["median"].append(st.median(hh))
            eh["empty"].append(sum(abs(x - DIAGONAL_256) < 1e-3 for x in hh))
            eh["mean_nonempty"].append(mean(x for x in hh if abs(x - DIAGONAL_256) > 1e-3))
        out["E_hd"] = eh
    write_json(out, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
