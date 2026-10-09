"""Run archived BMAS checkpoints and store per-case, per-exit outputs.

For every (variant, seed) with a checkpoint and for each split:
  out_root/<variant>/seed<seed>/<split>/per_case_<transform>.csv
  out_root/<variant>/seed<seed>/test/masks_test.npz      (figure variants, figure seed only)
  out_root/<variant>/seed<seed>/<split>/check.json       (agreement with the frozen evaluation)

Transforms: ``raw`` for every variant; ``cumavg`` and ``damp<g>`` additionally
for the post-hoc variants (default B). Needs torch + NumPy + SciPy + OpenCV + pandas.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from bmas.data import IMAGENET_MEAN, IMAGENET_STD, BRISCSegmentationDataset
from bmas.model import build_model_from_config
from experiments.case_processing import process_case
from experiments.common import read_json, write_json
from scripts.profile_anytime_e import extract_state_dict

META_COLS = ["case_id", "tumor_code", "plane_code", "lesion_pixels", "image_path"]


def load_cfg(path: str) -> dict:
    if path.endswith((".yaml", ".yml")):
        from bmas.utils import load_yaml
        return load_yaml(path)
    return read_json(path)


def build_model(ckpt_path: str, cfg: dict):
    """BMAS model from its config, or a same-backbone competitor recorded in the checkpoint."""
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    method = raw.get("method") if isinstance(raw, dict) else None
    if method in ("mess_style", "adpc_style"):
        from competitors.models import ADPCStyleNet, MESSStyleNet
        model = (MESSStyleNet if method == "mess_style" else ADPCStyleNet)(pretrained_encoder=False)
        cfg = {**cfg, "image_size": int(raw.get("image_size", 256)), "imagenet_norm": True}
    else:
        model = build_model_from_config(cfg, pretrained_encoder=False)
    model.load_state_dict(extract_state_dict(raw), strict=True)
    return model, cfg


def _collate(batch):
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def run_one(variant: str, seed: int, info: dict, args, transforms: List[str], save_masks: bool) -> None:
    ckpt = info.get("checkpoint")
    if not ckpt:
        print(f"[infer] {variant} seed{seed}: no checkpoint -> skipped")
        return
    cfg = load_cfg(info["config"]) if info.get("config") else {}
    base = Path(args.out_root) / variant / f"seed{seed}"
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    model = None
    for split in args.splits:
        split_dir = base / split
        files = [split_dir / f"per_case_{t}.csv" for t in transforms]
        need_masks = save_masks and split == "test"
        if args.skip_existing and all(f.is_file() for f in files) and (
                not need_masks or (split_dir / "masks_test.npz").is_file()):
            print(f"[infer] {variant} seed{seed} {split}: exists -> skipped")
            continue
        if model is None:
            model, cfg = build_model(ckpt, cfg)
            model.to(device).eval()
        ds = BRISCSegmentationDataset(args.manifest, split, int(cfg.get("image_size", 256)), train=False,
                                      imagenet_norm=bool(cfg.get("imagenet_norm", True)), data_root=args.data_root)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=_collate, pin_memory=device.type == "cuda")
        split_dir.mkdir(parents=True, exist_ok=True)
        rows_buffer: Dict[str, List[dict]] = {t: [] for t in transforms}
        mask_store = {"case_id": [], "image": [], "gt": [], "e3": [], "e4": []}
        t0 = time.time()
        with torch.inference_mode():
            for x, y, metas in loader:
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=bool(args.amp and device.type == "cuda")):
                    outs = model(x, max_exit=4)
                z = torch.stack([o.float() for o in outs], dim=1)[:, :, 0].cpu().numpy()  # [B,4,H,W]
                gts = y[:, 0].numpy()
                for b, meta in enumerate(metas):
                    res = process_case(z[b], gts[b], transforms, threshold=float(cfg.get("threshold", 0.5)))
                    for t in transforms:
                        rows_buffer[t].append({**{c: meta.get(c, "") for c in META_COLS}, **res[t]})
                    if need_masks:
                        img = x[b, 0].float().cpu().numpy()
                        if bool(cfg.get("imagenet_norm", True)):
                            img = img * IMAGENET_STD[0] + IMAGENET_MEAN[0]
                        mask_store["case_id"].append(meta["case_id"])
                        mask_store["image"].append(np.clip(img * 255.0, 0, 255).astype(np.uint8))
                        mask_store["gt"].append(np.packbits(gts[b] >= 0.5))
                        mask_store["e3"].append(np.packbits(z[b, 2] >= 0.0))
                        mask_store["e4"].append(np.packbits(z[b, 3] >= 0.0))
        for t in transforms:
            rows = rows_buffer[t]
            with (split_dir / f"per_case_{t}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        if need_masks:
            np.savez_compressed(split_dir / "masks_test.npz", case_id=np.array(mask_store["case_id"]),
                                image=np.stack(mask_store["image"]), gt=np.stack(mask_store["gt"]),
                                e3=np.stack(mask_store["e3"]), e4=np.stack(mask_store["e4"]),
                                shape=np.array(z.shape[-2:]))
        raw = rows_buffer.get("raw", [])
        check = {"variant": variant, "seed": seed, "split": split, "n": len(raw), "checkpoint": ckpt,
                 "seconds": time.time() - t0, "amp": bool(args.amp)}
        if raw:
            for k in (1, 2, 3, 4):
                check[f"exit{k}_dice_mean_reinfer"] = float(np.mean([r[f"e{k}_dice"] for r in raw]))
                check[f"exit{k}_hd95_mean_reinfer"] = float(np.mean([r[f"e{k}_hd95"] for r in raw]))
            if split == "test" and info.get("summary") and args.compare_frozen:
                try:
                    s = read_json(info["summary"])
                    for k in (1, 2, 3, 4):
                        if f"exit{k}_dice_mean" in s:
                            check[f"exit{k}_dice_mean_frozen"] = s[f"exit{k}_dice_mean"]
                    if "exit4_dice_mean" in s:
                        check["exit4_dice_abs_diff"] = abs(check["exit4_dice_mean_reinfer"] - s["exit4_dice_mean"])
                except Exception as exc:
                    check["frozen_summary_error"] = repr(exc)
        write_json(check, split_dir / "check.json")
        print(f"[infer] {variant} seed{seed} {split}: n={len(raw)} "
              f"E4 Dice={check.get('exit4_dice_mean_reinfer', float('nan')):.4f} "
              f"(frozen {check.get('exit4_dice_mean_frozen', 'n/a')}) in {check['seconds']:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-map", required=True)
    ap.add_argument("--variants", nargs="+", default=["A", "B", "C", "D", "E", "MESS", "ADPC"])
    ap.add_argument("--posthoc-variants", nargs="+", default=["B"])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    ap.add_argument("--mask-variants", nargs="+", default=["B", "D", "E"])
    ap.add_argument("--mask-seed", type=int, default=42)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["test", "val"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--compare-frozen", action=argparse.BooleanOptionalAction, default=True,
                    help="compare re-inferred test Dice with the frozen BRISC evaluation (off for external data)")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    vmap = read_json(args.variant_map)
    seeds = [int(s) for s in vmap["seeds"]]
    any_run = False
    for variant in args.variants:
        info_v = vmap["variants"].get(variant)
        if not info_v:
            print(f"[infer] variant {variant} not in variant map -> skipped")
            continue
        transforms = ["raw"]
        if variant in args.posthoc_variants:
            transforms += ["cumavg"] + [f"damp{g:g}" for g in args.gammas]
        for s in seeds:
            info = info_v["seeds"].get(str(s))
            if not info:
                continue
            any_run = any_run or bool(info.get("checkpoint"))
            run_one(variant, s, info, args, transforms,
                    save_masks=(variant in args.mask_variants and s == args.mask_seed))
    if not any_run:
        raise SystemExit("No checkpoints found for the requested variants: X5/X7/X9/X10 cannot run. "
                         "Restore best.pt files or retrain (see README_REVISION.md).")
    (Path(args.out_root) / "INFER_DONE").write_text(json.dumps({"variants": args.variants}), encoding="utf-8")


if __name__ == "__main__":
    main()
