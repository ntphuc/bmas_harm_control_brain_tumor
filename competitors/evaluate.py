from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bmas.data import BRISCSegmentationDataset
from bmas.metrics import numpy_case_metrics
from bmas.utils import get_device
from competitors.models import ADPCStyleNet, MESSStyleNet
from competitors.policies import adpc_spatial_fusion, image_confidence, mess_image_exit


def _progression_rates(df: pd.DataFrame, dice_tol: float = 0.001, hd95_tol: float = 0.5) -> dict[str, float]:
    """Case-transition reliability metrics used by the BMAS paper."""
    dice_viol = []
    hd95_viol = []
    for k in (1, 2, 3):
        d0 = df[f"e{k}_dice"].to_numpy(dtype=float)
        d1 = df[f"e{k+1}_dice"].to_numpy(dtype=float)
        h0 = df[f"e{k}_hd95"].to_numpy(dtype=float)
        h1 = df[f"e{k+1}_hd95"].to_numpy(dtype=float)
        dice_viol.append(d1 < (d0 - dice_tol))
        hd95_viol.append(h1 > (h0 + hd95_tol))
    dice_mvr = float(np.concatenate(dice_viol).mean() * 100.0)
    hd95_bmvr = float(np.concatenate(hd95_viol).mean() * 100.0)
    e34 = float(hd95_viol[-1].mean() * 100.0)
    return {
        "dice_mvr_percent": dice_mvr,
        "hd95_bmvr_percent": hd95_bmvr,
        "e3_to_e4_hd95_violation_percent": e34,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate MESS-style or ADP-C-style comparator on BRISC.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default="datasets/brisc2025")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--mess-thresholds", type=float, nargs=3, default=[0.98, 0.98, 0.98])
    p.add_argument("--adpc-threshold", type=float, default=0.98)
    args = p.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    method = ckpt["method"]
    cls = MESSStyleNet if method == "mess_style" else ADPCStyleNet
    model = cls(pretrained_encoder=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    ds = BRISCSegmentationDataset(
        args.manifest,
        args.split,
        256,
        train=False,
        imagenet_norm=True,
        data_root=args.data_root,
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    rows: list[dict[str, float | int | str]] = []
    active_all: list[np.ndarray] = []
    with torch.inference_mode():
        for images, masks, meta in tqdm(loader, desc=f"eval {method} {args.split}"):
            images = images.to(device, non_blocking=device.type == "cuda")
            outputs = model(images, max_exit=4)
            gt = masks.numpy()[0, 0].astype(np.float32)
            row: dict[str, float | int | str] = {"case_id": str(meta["case_id"][0])}

            for i, logits in enumerate(outputs, start=1):
                prob_t = torch.sigmoid(logits)
                prob = prob_t.cpu().numpy()[0, 0]
                for name, value in numpy_case_metrics(prob, gt, threshold=0.5).items():
                    row[f"e{i}_{name}"] = float(value)
                row[f"e{i}_image_confidence"] = float(image_confidence(prob_t).item())

            if method == "mess_style":
                dyn_logits, exit_ids = mess_image_exit(outputs, args.mess_thresholds)
                dyn_prob = torch.sigmoid(dyn_logits).cpu().numpy()[0, 0]
                for name, value in numpy_case_metrics(dyn_prob, gt, threshold=0.5).items():
                    row[f"dynamic_{name}"] = float(value)
                row["dynamic_exit"] = int(exit_ids.item())
            else:
                dyn_logits, active = adpc_spatial_fusion(outputs, args.adpc_threshold)
                dyn_prob = torch.sigmoid(dyn_logits).cpu().numpy()[0, 0]
                for name, value in numpy_case_metrics(dyn_prob, gt, threshold=0.5).items():
                    row[f"dynamic_{name}"] = float(value)
                active_all.append(active)
                for i, value in enumerate(active, start=2):
                    row[f"active_fraction_e{i}"] = float(value)

            rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_case_metrics.csv", index=False)

    summary: dict[str, object] = {
        "method": method,
        "split": args.split,
        "n_cases": len(df),
        "checkpoint": str(args.checkpoint),
    }
    for col in df.columns:
        if col != "case_id" and np.issubdtype(df[col].dtype, np.number):
            summary[f"{col}_mean"] = float(df[col].mean())
    summary.update(_progression_rates(df))

    if method == "mess_style":
        summary["mess_thresholds"] = [float(x) for x in args.mess_thresholds]
        counts = df["dynamic_exit"].value_counts(normalize=True).sort_index()
        summary["dynamic_exit_distribution"] = {str(int(k)): float(v) for k, v in counts.items()}
    else:
        summary["adpc_threshold"] = float(args.adpc_threshold)
        if active_all:
            summary["active_fraction_mean_e2_e3_e4"] = np.mean(np.stack(active_all), axis=0).tolist()

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
