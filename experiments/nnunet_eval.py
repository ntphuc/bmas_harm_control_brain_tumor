"""X12: score nnU-Net test predictions with the BMAS metric code and compare with E.

Writes per_case_metrics.csv (single exit, copied into e1..e4 columns like the
static baselines in bmas.model), summary.json, and NNUNET.md with the Table 1
row and paired image-level statistics against seed-averaged Variant E.
Needs NumPy + SciPy + OpenCV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from bmas.metrics import numpy_case_metrics
from experiments.common import (
    bootstrap_ci_mean, load_case_table, md_table, mean, read_csv_dicts, read_json, wilcoxon_signed_rank,
    write_csv, write_json, write_md,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--name-map", required=True)
    ap.add_argument("--variant-map", default="")
    ap.add_argument("--target", default="E")
    ap.add_argument("--trainer", default="nnUNetTrainer")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    nm = {r["nnunet_name"]: r for r in read_csv_dicts(args.name_map) if r["split"] == "test"}
    rows = []
    for name, r in nm.items():
        pred = cv2.imread(str(Path(args.pred_dir) / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(str(Path(args.labels_dir) / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if pred is None or gt is None:
            continue
        if pred.ndim == 3:
            pred = pred[..., 0]
        m = numpy_case_metrics((pred > 0).astype(np.float32), (gt > 0).astype(np.float32), threshold=0.5)
        row = {"case_id": r["case_id"], "nnunet_name": name}
        for k in (1, 2, 3, 4):
            for key, val in m.items():
                row[f"e{k}_{key}"] = val
        rows.append(row)
    if not rows:
        raise SystemExit(f"No predictions found in {args.pred_dir}")
    write_csv(out / "per_case_metrics.csv", rows)
    summ = {f"exit4_{k}_mean": float(np.mean([r[f"e4_{k}"] for r in rows]))
            for k in ("dice", "iou", "hd95", "assd", "boundary_iou")}
    summ.update({"n_cases": len(rows), "trainer": args.trainer, "exit4_dice_mean": summ["exit4_dice_mean"]})
    write_json(summ, out / "summary.json")

    text = ["# X12 — nnU-Net 2-D baseline (Table 1 row, single training run)\n",
            md_table(["Method", "Dice", "IoU", "HD95 (px)", "ASSD (px)", "B-IoU"],
                     [[f"nnU-Net 2-D ({args.trainer})", f"{summ['exit4_dice_mean']:.4f}",
                       f"{summ['exit4_iou_mean']:.4f}", f"{summ['exit4_hd95_mean']:.3f}",
                       f"{summ['exit4_assd_mean']:.3f}", f"{summ['exit4_boundary_iou_mean']:.4f}"]])]

    if args.variant_map and Path(args.variant_map).is_file():
        vmap = read_json(args.variant_map)
        info = vmap["variants"].get(args.target)
        tabs = [load_case_table(d["per_case_csv"]) for d in (info or {}).get("seeds", {}).values()
                if d.get("per_case_csv")]
        if tabs:
            nn = {r["case_id"]: r for r in rows}
            md_rows = []
            for label, key, higher in (("Dice", "dice", True), ("HD95 (px)", "hd95", False),
                                       ("ASSD (px)", "assd", False)):
                diffs = []
                for c, r in nn.items():
                    vals = [t[c].get(f"e4_{key}") for t in tabs if c in t and t[c].get(f"e4_{key}") is not None]
                    if not vals:
                        continue
                    e = sum(vals) / len(vals)
                    diffs.append((e - r[f"e4_{key}"]) if higher else (r[f"e4_{key}"] - e))
                if diffs:
                    lo, hi = bootstrap_ci_mean(diffs)
                    _, p, _ = wilcoxon_signed_rank(diffs)
                    md_rows.append([label, len(diffs), f"{mean(diffs):.5g}", f"[{lo:.5g}, {hi:.5g}]", f"{p:.3g}"])
            text.append(f"\nPaired comparison, {args.target} (seed-averaged per image) vs nnU-Net; positive = "
                        f"{args.target} better. Add to the BH family of Section 5.5 before reporting.\n\n" +
                        md_table(["Metric", "n", "Mean improvement", "95% bootstrap CI", "Wilcoxon p"], md_rows))
    write_md(out / "NNUNET.md", "\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
