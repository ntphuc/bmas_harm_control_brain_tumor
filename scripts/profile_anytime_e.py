from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
from pathlib import Path
from typing import Any

import csv
import math
import statistics

import torch

from bmas.model import build_model_from_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Standardized BMAS-E fixed-exit profiling: batch-1 latency, p50/p95, "
            "GFLOPs, executed/total parameters, CUDA peak memory, plus Dice/HD95 "
            "read from a frozen evaluation summary. A trained checkpoint is optional."
        )
    )
    p.add_argument("--config", required=True)
    p.add_argument("--metrics-summary", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--memory-runs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_state_dict(checkpoint_obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "model_state_dict", "state_dict", "net", "network"):
            value = checkpoint_obj.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
                state = value
                break
        else:
            if checkpoint_obj and all(torch.is_tensor(v) for v in checkpoint_obj.values()):
                state = checkpoint_obj
            else:
                raise ValueError("Checkpoint does not contain a recognizable state dict.")
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint_obj)!r}")

    for prefix in ("module.", "model."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items()}
    return state


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=enabled, dtype=torch.float16)
    return contextlib.nullcontext()


def percentile(values: list[float], q: float) -> float:
    """NumPy-free linear percentile for latency summaries."""
    if not values:
        return float("nan")
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (float(q) / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def profile_latency(model, x, max_exit: int, warmup: int, runs: int, amp: bool) -> dict[str, float]:
    if x.device.type != "cuda":
        raise RuntimeError("Latency profiling is intended for CUDA. Use --device cuda.")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast_context(x.device, amp):
                _ = model(x, max_exit=max_exit)
        torch.cuda.synchronize()
        times_ms = []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_context(x.device, amp):
                _ = model(x, max_exit=max_exit)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
    return {
        "latency_mean_ms": float(statistics.fmean(times_ms)),
        "latency_sd_ms": float(statistics.stdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "latency_p50_ms": percentile(times_ms, 50),
        "latency_p95_ms": percentile(times_ms, 95),
        "latency_min_ms": float(min(times_ms)),
        "latency_max_ms": float(max(times_ms)),
    }


def profile_peak_memory(model, x, max_exit: int, runs: int, amp: bool) -> dict[str, float]:
    if x.device.type != "cuda":
        return {"peak_allocated_mb": float("nan"), "peak_reserved_mb": float("nan"),
                "incremental_peak_allocated_mb": float("nan")}
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    with torch.inference_mode():
        for _ in range(max(1, runs)):
            with autocast_context(x.device, amp):
                _ = model(x, max_exit=max_exit)
        torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    mb = 1024.0 ** 2
    return {
        "peak_allocated_mb": float(peak_alloc / mb),
        "peak_reserved_mb": float(peak_reserved / mb),
        "incremental_peak_allocated_mb": float(max(0, peak_alloc - before) / mb),
    }


def profile_flops(model, x, max_exit: int, amp: bool) -> tuple[float, str]:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.inference_mode():
            with FlopCounterMode(display=False) as mode:
                with autocast_context(x.device, amp):
                    _ = model(x, max_exit=max_exit)
        return float(mode.get_total_flops() / 1e9), "torch.utils.flop_counter.FlopCounterMode"
    except Exception as exc:
        total = 0
        hooks = []
        def conv_hook(module, inputs, output):
            nonlocal total
            if not torch.is_tensor(output):
                return
            batch = int(output.shape[0])
            out_h, out_w = int(output.shape[-2]), int(output.shape[-1])
            out_ch = int(output.shape[1])
            if isinstance(module, torch.nn.Conv2d):
                kh, kw = module.kernel_size
                in_per_group = module.in_channels // module.groups
                total += 2 * batch * out_h * out_w * out_ch * in_per_group * kh * kw
            elif isinstance(module, torch.nn.ConvTranspose2d):
                kh, kw = module.kernel_size
                in_ch = module.in_channels
                out_per_group = module.out_channels // module.groups
                in_h, in_w = int(inputs[0].shape[-2]), int(inputs[0].shape[-1])
                total += 2 * batch * in_h * in_w * in_ch * out_per_group * kh * kw
        def linear_hook(module, inputs, output):
            nonlocal total
            n = int(inputs[0].numel() // module.in_features)
            total += 2 * n * module.in_features * module.out_features
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                hooks.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, torch.nn.Linear):
                hooks.append(m.register_forward_hook(linear_hook))
        try:
            with torch.inference_mode(), autocast_context(x.device, amp):
                _ = model(x, max_exit=max_exit)
        finally:
            for h in hooks:
                h.remove()
        return float(total / 1e9), f"fallback_conv_linear_hooks ({type(exc).__name__})"


def executed_parameter_count(model, x, max_exit: int, amp: bool) -> int:
    used: set[int] = set()
    hooks = []
    def mark(module, _inputs, _output):
        for p in module.parameters(recurse=False):
            used.add(id(p))
    for m in model.modules():
        if any(True for _ in m.parameters(recurse=False)):
            hooks.append(m.register_forward_hook(mark))
    try:
        with torch.inference_mode(), autocast_context(x.device, amp):
            _ = model(x, max_exit=max_exit)
    finally:
        for h in hooks:
            h.remove()
    id_to_numel = {id(p): int(p.numel()) for p in model.parameters()}
    return int(sum(id_to_numel[i] for i in used if i in id_to_numel))


def hardware_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": str(device),
    }
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu_name": prop.name,
            "gpu_total_memory_gb": float(prop.total_memory / 1024**3),
            "gpu_compute_capability": f"{prop.major}.{prop.minor}",
            "cuda_device_index": int(idx),
        })
    return info


def metrics_from_frozen_summary(summary: dict[str, Any], exit_index: int) -> dict[str, float]:
    return {
        "n_cases": int(summary.get("n_cases", 0)),
        "dice_mean": float(summary[f"exit{exit_index}_dice_mean"]),
        "hd95_mean": float(summary[f"exit{exit_index}_hd95_mean"]),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("For reviewer-facing fixed-exit profiling, use --batch-size 1.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config).resolve()
    metrics_path = Path(args.metrics_summary).resolve()
    cfg = load_json(cfg_path)
    frozen_metrics = load_json(metrics_path)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    # Avoid downloading pretrained encoder weights. Learned weights, if available,
    # are loaded below. If they are not available, random initialization is valid for
    # architecture-only compute profiling because the executed graph is unchanged.
    torch.manual_seed(args.seed)
    model = build_model_from_config(cfg, pretrained_encoder=False)
    checkpoint_path = None
    profiling_mode = "architecture_only_no_checkpoint"
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        if checkpoint_path.exists():
            checkpoint_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            state = extract_state_dict(checkpoint_obj)
            model.load_state_dict(state, strict=True)
            profiling_mode = "trained_checkpoint"
    model.to(device).eval()

    image_size = int(cfg.get("image_size", 256))
    # Use one synthetic input so I/O and preprocessing are excluded from model-only
    # latency. This is the reviewer-facing inference-compute profile.
    sample_x = torch.randn(1, 3, image_size, image_size, device=device)

    hw = hardware_info(device)
    hw.update({
        "seed": args.seed,
        "profiling_mode": profiling_mode,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "config": str(cfg_path),
        "metrics_summary": str(metrics_path),
        "batch_size": 1,
        "image_size": image_size,
        "synthetic_input": True,
        "amp": bool(args.amp),
        "latency_warmup": args.warmup,
        "latency_runs": args.runs,
        "memory_runs": args.memory_runs,
        "note": (
            "Latency/FLOPs/VRAM/parameter counts are architecture-path measurements and do not require trained weights. "
            "Dice/HD95 are copied from the frozen held-out evaluation summary."
        ),
    })
    (output_dir / "hardware.json").write_text(json.dumps(hw, indent=2), encoding="utf-8")

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    fixed_exit: dict[int, dict[str, Any]] = {}
    for e in range(1, 5):
        print(f"[profile] seed={args.seed} exit=E{e}: latency", flush=True)
        lat = profile_latency(model, sample_x, e, args.warmup, args.runs, args.amp)
        print(f"[profile] seed={args.seed} exit=E{e}: peak memory", flush=True)
        mem = profile_peak_memory(model, sample_x, e, args.memory_runs, args.amp)
        print(f"[profile] seed={args.seed} exit=E{e}: FLOPs / executed params", flush=True)
        gflops, flop_method = profile_flops(model, sample_x, e, args.amp)
        used_params = executed_parameter_count(model, sample_x, e, args.amp)
        fixed_exit[e] = {
            "exit": e, **lat, **mem,
            "gflops": gflops, "flop_method": flop_method,
            "executed_params": used_params, "executed_params_m": used_params / 1e6,
            "total_params": total_params, "total_params_m": total_params / 1e6,
            "trainable_params": trainable_params, "trainable_params_m": trainable_params / 1e6,
            "profiling_mode": profiling_mode,
        }

    rows = []
    e4_latency = fixed_exit[4]["latency_mean_ms"]
    for e in range(1, 5):
        r = {"seed": args.seed, **fixed_exit[e], **metrics_from_frozen_summary(frozen_metrics, e)}
        r["latency_reduction_vs_e4_pct"] = float(100.0 * (e4_latency - r["latency_mean_ms"]) / e4_latency)
        r["speedup_vs_e4_x"] = float(e4_latency / r["latency_mean_ms"])
        rows.append(r)

    csv_path = output_dir / "profile_by_exit.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "profile_summary.json").write_text(
        json.dumps({"seed": args.seed, "hardware": hw, "rows": rows}, indent=2), encoding="utf-8"
    )

    cols = [
        "exit", "dice_mean", "hd95_mean", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
        "gflops", "peak_allocated_mb", "executed_params_m", "latency_reduction_vs_e4_pct"
    ]
    print(" | ".join(cols), flush=True)
    for row in rows:
        vals = []
        for c in cols:
            v = row[c]
            vals.append(f"{v:.6g}" if isinstance(v, float) else str(v))
        print(" | ".join(vals), flush=True)
    print(f"Wrote: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
