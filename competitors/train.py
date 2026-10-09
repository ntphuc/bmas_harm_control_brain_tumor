from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bmas.data import BRISCSegmentationDataset
from bmas.metrics import dice_iou_from_logits
from bmas.utils import get_device, seed_everything
from competitors.losses import mess_stage2_loss, multiexit_seg_loss
from competitors.models import ADPCStyleNet, MESSStyleNet


def _save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))


def run_epoch(model, loader, device, optimizer, scaler, stage, amp, grad_clip, teacher_conf, distill_weight):
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    dice_sum = [0.0] * 4
    n = 0
    for images, masks, _ in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=device.type == "cuda")
        masks = masks.to(device, non_blocking=device.type == "cuda")
        if train:
            optimizer.zero_grad(set_to_none=True)
        use_amp = amp and device.type == "cuda"
        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images, max_exit=4)
            if stage == "mess_stage2":
                loss_out = mess_stage2_loss(outputs, masks, distill_weight, teacher_conf)
                loss = loss_out.total
            else:
                loss = multiexit_seg_loss(outputs, masks)
        if train:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        bs = images.size(0)
        n += bs
        loss_sum += float(loss.detach()) * bs
        for i, logits in enumerate(outputs):
            dice, _ = dice_iou_from_logits(logits.detach(), masks, threshold=0.5)
            dice_sum[i] += float(dice.sum().cpu())
    out = {"loss": loss_sum / max(n, 1)}
    out.update({f"exit{i+1}_dice": dice_sum[i] / max(n, 1) for i in range(4)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["mess_style", "adpc_style"], required=True)
    p.add_argument("--stage", choices=["joint", "mess_stage2"], default="joint")
    p.add_argument("--stage1-checkpoint", default=None, help="required for mess_stage2")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", default="datasets/brisc2025")
    p.add_argument("--output", default="outputs/competitors")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--distill-weight", type=float, default=0.5)
    p.add_argument("--teacher-confidence", type=float, default=0.90)
    p.add_argument("--no-pretrained", action="store_true")
    args = p.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    cls = MESSStyleNet if args.method == "mess_style" else ADPCStyleNet
    model = cls(pretrained_encoder=not args.no_pretrained).to(device)

    if args.stage == "mess_stage2":
        if args.method != "mess_style":
            raise ValueError("mess_stage2 is only valid for --method mess_style")
        if not args.stage1_checkpoint:
            raise ValueError("--stage1-checkpoint is required for mess_stage2")
        ckpt = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.freeze_trunk_for_stage2()

    train_ds = BRISCSegmentationDataset(args.manifest, "train", 256, train=True, imagenet_norm=True, data_root=args.data_root)
    val_ds = BRISCSegmentationDataset(args.manifest, "val", 256, train=False, imagenet_norm=True, data_root=args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    run_dir = Path(args.output) / f"{args.method}_{args.stage}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, optimizer, scaler, args.stage, True, args.grad_clip, args.teacher_confidence, args.distill_weight)
        with torch.no_grad():
            va = run_epoch(model, val_loader, device, None, scaler, args.stage, True, args.grad_clip, args.teacher_confidence, args.distill_weight)
        scheduler.step()
        row = {"epoch": epoch, "seconds": time.time() - t0, "train": tr, "val": va}
        history.append(row)
        _save_json(history, run_dir / "history.json")
        checkpoint = {
            "model": model.state_dict(),
            "method": args.method,
            "stage": args.stage,
            "seed": args.seed,
            "image_size": 256,
            "config": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        score = va["exit4_dice"] if args.stage == "joint" else sum(va[f"exit{i}_dice"] for i in (1,2,3)) / 3.0
        print(f"epoch={epoch:03d} val E1/E2/E3/E4=" + "/".join(f"{va[f'exit{i}_dice']:.4f}" for i in range(1,5)))
        if score > best:
            best = score
            bad = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            bad += 1
        if bad >= args.patience:
            break


if __name__ == "__main__":
    main()
