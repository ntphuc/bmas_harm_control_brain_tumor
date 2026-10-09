"""Train one BMAS variant (A-E) or endpoint baseline for one seed.

    python -m scripts.train --config configs/variant_E.yaml --seed 42 \
        --manifest data/brisc2025_manifest.csv --data-root datasets/brisc2025 --output outputs/runs

Writes <output>/<experiment_name>_seed<seed>/ with config_resolved.json,
history.json, last.pt, and best.pt (best validation Exit-4 Dice, patience 10).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bmas.data import BRISCSegmentationDataset
from bmas.losses import BMASLoss
from bmas.metrics import dice_iou_from_logits
from bmas.model import build_model_from_config
from bmas.transition_losses import TransitionLoss
from bmas.utils import get_device, load_yaml, save_json, seed_everything, validate_training_config


def build_loss(cfg: dict) -> torch.nn.Module:
    common = dict(exit_weights=cfg["exit_weights"], dice_weight=cfg["dice_weight"], bce_weight=cfg["bce_weight"],
                  boundary_kernel=cfg["boundary_kernel"])
    if cfg.get("objective", "bmas") == "transition":
        t = cfg["transition"]
        loss = TransitionLoss(cfg["paper_variant"], boundary_weight=cfg["boundary_weight"], band=t["band"],
                              power=t["power"], margin=t["margin"], tail_fraction=t["tail_fraction"],
                              d_weight=t["d_weight"], d_coeffs=t["d_coeffs"], e_coeffs=t["e_coeffs"],
                              e_beta=t["e_beta"], lambda_max=t["lambda_max"], warmup_epochs=t["warmup_epochs"],
                              ramp_epochs=t["ramp_epochs"])
        loss.base = BMASLoss(boundary_weight=cfg["boundary_weight"], monotonic_weight=0.0, **common)
        return loss
    return BMASLoss(boundary_weight=cfg["boundary_weight"], monotonic_weight=cfg["monotonic_weight"],
                    monotonic_margin=cfg["monotonic_margin"], **common)


def run_epoch(model, loader, loss_fn, device, optimizer, scaler, amp, grad_clip, threshold):
    train = optimizer is not None
    model.train(train)
    loss_sum, n = 0.0, 0
    dice_sum = [0.0] * 4
    for images, masks, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train), autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images, max_exit=4)
            loss = loss_fn([o.float() for o in outputs], masks).total
        if train:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        bs = images.size(0)
        n += bs
        loss_sum += float(loss.detach()) * bs
        for i, logits in enumerate(outputs):
            dice, _ = dice_iou_from_logits(logits.detach().float(), masks, threshold=threshold)
            dice_sum[i] += float(dice.sum().cpu())
    out = {"loss": loss_sum / max(n, 1)}
    out.update({f"exit{i+1}_dice": dice_sum[i] / max(n, 1) for i in range(4)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="datasets/brisc2025")
    ap.add_argument("--output", default="outputs/runs")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override (for smoke tests)")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    validate_training_config(cfg)
    cfg.update({"seed": args.seed, "manifest": str(args.manifest), "data_root": str(args.data_root),
                "config_file": str(args.config)})

    run_dir = Path(args.output) / f"{cfg['experiment_name']}_seed{args.seed}"
    if args.skip_existing and (run_dir / "best.pt").is_file() and (run_dir / "DONE").is_file():
        print(f"[skip] {run_dir} already trained")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(cfg, run_dir / "config_resolved.json")

    seed_everything(args.seed)
    device = get_device(args.device)
    model = build_model_from_config(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)

    size = int(cfg["image_size"])
    train_ds = BRISCSegmentationDataset(args.manifest, "train", size, train=True, imagenet_norm=True, data_root=args.data_root)
    val_ds = BRISCSegmentationDataset(args.manifest, "val", size, train=False, imagenet_norm=True, data_root=args.data_root)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, generator=g,
                              num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False,
                            num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda")

    optimizer = AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    epochs, warm = int(cfg["epochs"]), int(cfg.get("warmup_epochs", 0))
    cosine = CosineAnnealingLR(optimizer, T_max=max(epochs - warm, 1), eta_min=float(cfg.get("min_lr", 0.0)))
    if warm > 0:
        scheduler = SequentialLR(optimizer, [LinearLR(optimizer, start_factor=float(cfg.get("warmup_start_factor", 0.1)),
                                                      total_iters=warm), cosine], milestones=[warm])
    else:
        scheduler = cosine
    amp = bool(cfg.get("amp", True))
    scaler = GradScaler("cuda", enabled=amp and device.type == "cuda")

    best, bad, history = -1.0, 0, []
    for epoch in range(epochs):
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        t0 = time.time()
        tr = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler, amp, float(cfg["grad_clip"]),
                       float(cfg["threshold"]))
        with torch.no_grad():
            va = run_epoch(model, val_loader, loss_fn, device, None, scaler, amp, 0.0, float(cfg["threshold"]))
        scheduler.step()
        row = {"epoch": epoch + 1, "seconds": time.time() - t0, "lr": optimizer.param_groups[0]["lr"],
               "train": tr, "val": va}
        if hasattr(loss_fn, "schedule"):
            row["transition_weight"] = loss_fn.schedule()
        history.append(row)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        ckpt = {"model": model.state_dict(), "config": cfg, "epoch": epoch + 1, "val": va}
        torch.save(ckpt, run_dir / "last.pt")
        print(f"[{cfg['experiment_name']} s{args.seed}] epoch {epoch+1:02d} val E1-E4 Dice "
              + "/".join(f"{va[f'exit{i}_dice']:.4f}" for i in range(1, 5)), flush=True)
        if va["exit4_dice"] > best:
            best, bad = va["exit4_dice"], 0
            torch.save(ckpt, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= int(cfg["patience"]):
                print(f"early stop after epoch {epoch+1}")
                break
    (run_dir / "DONE").write_text(json.dumps({"best_val_exit4_dice": best}))


if __name__ == "__main__":
    main()
