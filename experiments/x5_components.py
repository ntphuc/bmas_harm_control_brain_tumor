"""X5: are late HD95 regressions caused by new, remote false-positive components?

Reads inference outputs (per_case_raw.csv, test split) and reports per variant:
  * share of E3->E4 HD95 regressions accompanied by >=1 new connected component at E4,
  * share accompanied by a new component farther than 10 px from the reference,
  * median distance of new components to the reference (pooled),
  * mean number of new components per image.
A "new" component does not overlap the E3 prediction (8-connectivity).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from experiments.common import HD95_TOL, load_case_table, md_table, mean, percentile, pm, sd, write_csv, write_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E", "MESS", "ADPC"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    root, out = Path(args.infer_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows, md_rows = [], []
    for v in args.variants:
        acc = {"share_new": [], "share_far": [], "share_new_nonreg": [], "new_per_image": [], "n_reg": [],
               "dhd95_with_new": [], "dhd95_without_new": [], "share_increase_from_new": []}
        dists_all = []
        for s in args.seeds:
            p = root / v / f"seed{s}" / "test" / "per_case_raw.csv"
            if not p.is_file():
                continue
            t = load_case_table(p)
            reg = [c for c in t if t[c]["e4_hd95"] is not None and t[c]["e3_hd95"] is not None
                   and t[c]["e4_hd95"] > t[c]["e3_hd95"] + HD95_TOL]
            reg_set = set(reg)
            nonreg = [c for c in t if c not in reg_set]
            # component columns (e34_*) are read as raw strings because distances are ';'-joined lists
            with p.open(newline="", encoding="utf-8") as f:
                extra = {r["case_id"]: r for r in csv.DictReader(f)}

            def newcc(c):
                return float(extra[c].get("e34_new_cc") or 0)

            def far(c):
                return float(extra[c].get("e34_new_cc_far") or 0)

            acc["n_reg"].append(len(reg))
            acc["share_new"].append(100.0 * mean(float(newcc(c) > 0) for c in reg) if reg else float("nan"))
            acc["share_far"].append(100.0 * mean(far(c) for c in reg) if reg else float("nan"))
            acc["share_new_nonreg"].append(100.0 * mean(float(newcc(c) > 0) for c in nonreg) if nonreg else float("nan"))
            acc["new_per_image"].append(mean(newcc(c) for c in t))
            inc_w = [t[c]["e4_hd95"] - t[c]["e3_hd95"] for c in reg if newcc(c) > 0]
            inc_wo = [t[c]["e4_hd95"] - t[c]["e3_hd95"] for c in reg if newcc(c) == 0]
            acc["dhd95_with_new"].append(mean(inc_w))
            acc["dhd95_without_new"].append(mean(inc_wo))
            tot = sum(inc_w) + sum(inc_wo)
            acc["share_increase_from_new"].append(100.0 * sum(inc_w) / tot if tot > 0 else float("nan"))
            for c in t:
                d = extra[c].get("e34_new_cc_dists") or ""
                dists_all += [float(x) for x in d.split(";") if x]
        if not acc["n_reg"]:
            continue
        row = {"variant": v, "n_seeds": len(acc["n_reg"]), "median_new_component_distance_px": percentile(dists_all, 50),
               "n_new_components_pooled": len(dists_all)}
        for k, vals in acc.items():
            row[f"{k}_mean"], row[f"{k}_sd"] = mean(vals), sd(vals)
        rows.append(row)
        md_rows.append([v, pm(acc["n_reg"], 1), pm(acc["share_new"], 1), pm(acc["share_far"], 1),
                        pm(acc["share_new_nonreg"], 1), f"{percentile(dists_all, 50):.1f}" if dists_all else "—",
                        pm(acc["new_per_image"], 3), pm(acc["dhd95_with_new"], 2), pm(acc["dhd95_without_new"], 2),
                        pm(acc["share_increase_from_new"], 1)])
    write_csv(out / "x5_components.csv", rows)
    text = ("# X5 — New connected components at E4 (Section 7.4)\n\n" + md_table(
        ["Variant", "E3→E4 HD95 regressions (n)", "% with new component", "% with new component >10 px",
         "% new component among non-regressing images", "Median distance of new components (px)",
         "New components per image", "Mean ΔHD95 with new component (px)", "Mean ΔHD95 without (px)",
         "% of total late HD95 increase from new-component cases"], md_rows) +
        "\n\nIf the mechanism in Section 7.4 holds, D should show a higher share of regressions with a new remote "
        "component than B and E, and this share should exceed the rate among non-regressing images.")
    write_md(out / "X5_COMPONENTS.md", text)
    print(text)


if __name__ == "__main__":
    main()
