"""Held-out evaluation of one trained run (all four exits).

    python -m scripts.evaluate --run-dir outputs/runs/bmas_E_preserve_dominant_seed42 \
        --manifest data/brisc2025_manifest.csv --data-root datasets/brisc2025

Writes <run-dir>/eval_test_official/per_case_metrics.csv (one row per image:
Dice, IoU, HD95, ASSD, Boundary-IoU at E1-E4, plus local boundary-transition
counts) and summary.json (means, MVR/BMVR, E3->E4 rates, pooled BHR/BCR).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bmas.data import BRISCSegmentationDataset
from bmas.local_metrics import pooled_rates, transition_counts
from experiments.case_processing import process_case
from experiments.common import load_case_table, regression_rates
from experiments.infer_exits import build_model


def _collate(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out-name", default="eval_test_official")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--amp", action="store_true", help="mixed precision (the paper's frozen evaluation used FP32)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / args.out_name
    if args.skip_existing and (out / "summary.json").is_file():
        print(f"[skip] {out}")
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((run / "config_resolved.json").read_text()) if (run / "config_resolved.json").is_file() else {}
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    model, cfg = build_model(str(run / args.checkpoint), cfg)
    model.to(device).eval()
    thr = float(cfg.get("threshold", 0.5))

    ds = BRISCSegmentationDataset(args.manifest, args.split, int(cfg.get("image_size", 256)), train=False,
                                  imagenet_norm=bool(cfg.get("imagenet_norm", True)), data_root=args.data_root)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        collate_fn=_collate, pin_memory=device.type == "cuda")
    rows = []
    with torch.inference_mode():
        for x, y, metas in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
                outs = model(x.to(device), max_exit=4)
            z = torch.stack([o.float() for o in outs], 1)[:, :, 0].cpu().numpy()
            gts = y[:, 0].numpy()
            for b, meta in enumerate(metas):
                metrics = process_case(z[b], gts[b], ["raw"], threshold=thr, with_components=False)["raw"]
                masks = [z[b, k] >= np.log(thr / (1 - thr)) for k in range(z.shape[1])]
                row = {"case_id": meta["case_id"], "plane_code": meta.get("plane_code", ""),
                       "tumor_code": meta.get("tumor_code", ""), "lesion_pixels": meta.get("lesion_pixels", "")}
                row.update({k: v for k, v in metrics.items() if not k.endswith(("_uncert", "_fg"))})
                row.update(transition_counts(masks, gts[b] >= 0.5, tol=2.0))
                rows.append(row)
    with (out / "per_case_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    table = load_case_table(out / "per_case_metrics.csv")
    ids = list(table)
    summary = {"split": args.split, "n_cases": len(ids), "checkpoint": str(run / args.checkpoint), "amp": args.amp}
    for k in range(1, 5):
        for m in ("dice", "iou", "hd95", "assd", "boundary_iou"):
            summary[f"exit{k}_{m}_mean"] = float(np.mean([table[c][f"e{k}_{m}"] for c in ids]))
    summary.update(regression_rates(table, ids))
    summary.update(pooled_rates(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] {run.name}: Dice E1-E4 " + "/".join(f"{summary[f'exit{k}_dice_mean']:.4f}" for k in range(1, 5))
          + f" | MVR {summary['mvr']:.2f}% BMVR {summary['bmvr']:.2f}% BHR {summary['bhr']:.2f}% BCR {summary['bcr']:.2f}%")


if __name__ == "__main__":
    main()
