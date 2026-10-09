"""X11: fixed-exit latency of BMAS-E on CPU (FP32, batch 1) plus end-to-end time.

Forward latency uses a synthetic 1x3x256x256 input, so it is valid with or
without trained weights (the executed graph is identical). End-to-end time,
measured only when the data stack (NumPy/OpenCV/pandas) is importable, covers
reading one held-out image, preprocessing, the forward pass and thresholding.
"""
from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from pathlib import Path

import torch

from bmas.model import build_model_from_config
from experiments.common import percentile, read_json, write_csv, write_json, write_md, md_table, pm
from scripts.profile_anytime_e import extract_state_dict


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def time_forward(model, x, exit_k: int, warmup: int, runs: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, max_exit=exit_k)
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x, max_exit=exit_k)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return ts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-map", required=True)
    ap.add_argument("--variant", default="E")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--e2e-images", type=int, default=100)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vmap = read_json(args.variant_map)
    info_v = vmap["variants"].get(args.variant, {"seeds": {}})
    seeds = [int(s) for s in vmap["seeds"]]
    hw = {"cpu": cpu_name(), "os_cpu_count": os.cpu_count(), "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
          "torch": torch.__version__, "python": platform.python_version(), "dtype": "float32", "batch_size": 1,
          "warmup": args.warmup, "runs": args.runs, "threads": args.threads,
          "parallel_info": torch.__config__.parallel_info()}
    rows, e2e_rows = [], []
    for s in seeds:
        info = info_v["seeds"].get(str(s), {})
        cfg = read_json(info["config"]) if info.get("config", "").endswith(".json") else {}
        torch.manual_seed(s)
        model = build_model_from_config(cfg, pretrained_encoder=False)
        mode = "architecture_only"
        if info.get("checkpoint"):
            model.load_state_dict(extract_state_dict(torch.load(info["checkpoint"], map_location="cpu",
                                                                weights_only=False)), strict=True)
            mode = "trained_checkpoint"
        model.eval()
        size = int(cfg.get("image_size", 256))
        x = torch.randn(1, 3, size, size)
        for t in args.threads:
            torch.set_num_threads(t)
            e4 = None
            per_exit = {}
            for k in (4, 1, 2, 3):
                ts = time_forward(model, x, k, args.warmup, args.runs)
                per_exit[k] = ts
            e4 = statistics.fmean(per_exit[4])
            for k in (1, 2, 3, 4):
                ts = per_exit[k]
                m = statistics.fmean(ts)
                rows.append({"seed": s, "threads": t, "exit": k, "mode": mode, "latency_mean_ms": m,
                             "latency_sd_ms": statistics.stdev(ts), "latency_p50_ms": percentile(ts, 50),
                             "latency_p95_ms": percentile(ts, 95), "speedup_vs_e4": e4 / m})
            print(f"[cpu] seed={s} threads={t} " + " ".join(
                f"E{k}={statistics.fmean(per_exit[k]):.2f}ms" for k in (1, 2, 3, 4)), flush=True)

        if args.manifest and args.e2e_images > 0:
            try:
                from bmas.data import BRISCSegmentationDataset
                ds = BRISCSegmentationDataset(args.manifest, "test", size, train=False,
                                              imagenet_norm=bool(cfg.get("imagenet_norm", True)),
                                              data_root=args.data_root)
                torch.set_num_threads(max(args.threads))
                n = min(args.e2e_images, len(ds))
                for k in (1, 4):
                    ts = []
                    with torch.inference_mode():
                        for i in range(n):
                            t0 = time.perf_counter()
                            img, _, _ = ds[i]
                            logits = model(img[None], max_exit=k)[-1]
                            _ = (torch.sigmoid(logits) >= 0.5)
                            ts.append((time.perf_counter() - t0) * 1000.0)
                    e2e_rows.append({"seed": s, "threads": max(args.threads), "exit": k, "n_images": n,
                                     "e2e_mean_ms": statistics.fmean(ts[5:] if n > 10 else ts),
                                     "e2e_p95_ms": percentile(ts, 95)})
            except Exception as exc:
                hw["e2e_error"] = repr(exc)
                print(f"[cpu] end-to-end timing skipped: {exc!r}")

    write_csv(out / "cpu_profile_by_seed.csv", rows)
    if e2e_rows:
        write_csv(out / "cpu_end_to_end.csv", e2e_rows)
    write_json(hw, out / "cpu_hardware.json")

    summ, md_rows = [], []
    for t in args.threads:
        for k in (1, 2, 3, 4):
            sel = [r for r in rows if r["threads"] == t and r["exit"] == k]
            ms = [r["latency_mean_ms"] for r in sel]
            sp = [r["speedup_vs_e4"] for r in sel]
            e2e = [r["e2e_mean_ms"] for r in e2e_rows if r["exit"] == k and r["threads"] == t]
            summ.append({"exit": k, "threads": t, "latency_mean_ms_mean": statistics.fmean(ms),
                         "latency_mean_ms_sd": statistics.stdev(ms) if len(ms) > 1 else 0.0,
                         "speedup_vs_e4_mean": statistics.fmean(sp),
                         "e2e_mean_ms": statistics.fmean(e2e) if e2e else ""})
            md_rows.append([f"E{k}", t, pm(ms, 2), f"{statistics.fmean(sp):.2f}×", pm(e2e, 2) if e2e else "—"])
    write_csv(out / "cpu_profile_summary.csv", summ)
    text = (f"# X11 — CPU fixed-exit latency (Table 4)\n\nCPU: {hw['cpu']}; torch {hw['torch']}; FP32; batch 1; "
            f"{args.warmup} warm-up + {args.runs} timed passes; synthetic input; mean ± SD over seeds.\n\n"
            + md_table(["Exit", "Threads", "Forward latency (ms)", "Speed-up vs E4", "End-to-end (ms)"], md_rows))
    write_md(out / "CPU_PROFILE.md", text)
    print(text)


if __name__ == "__main__":
    main()
