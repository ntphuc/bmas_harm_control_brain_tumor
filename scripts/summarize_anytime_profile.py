from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    return p.parse_args()


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def as_float(value):
    try:
        if value is None or value == "":
            return None
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def mean_sd(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return float("nan"), float("nan")
    return statistics.fmean(vals), statistics.stdev(vals) if len(vals) > 1 else 0.0


def main():
    args = parse_args()
    root = Path(args.root)
    paths = sorted(root.glob("seed*/profile_by_exit.csv"))
    if not paths:
        raise FileNotFoundError(f"No seed*/profile_by_exit.csv under {root}")

    all_rows = []
    fieldnames = None
    for path in paths:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames or []
            all_rows.extend(dict(r) for r in reader)

    all_csv = root / "profile_all_seeds.csv"
    with all_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(all_rows)

    numeric = [
        "dice_mean", "hd95_mean", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms",
        "gflops", "peak_allocated_mb", "peak_reserved_mb", "incremental_peak_allocated_mb",
        "executed_params_m", "total_params_m", "latency_reduction_vs_e4_pct", "speedup_vs_e4_x",
    ]
    groups = defaultdict(list)
    for r in all_rows:
        groups[int(float(r["exit"]))].append(r)

    out_rows = []
    for e in sorted(groups):
        g = groups[e]
        seed_vals = {r.get("seed", "") for r in g}
        row = {"exit": e, "n_seeds": len(seed_vals)}
        for col in numeric:
            mean, sd = mean_sd([as_float(r.get(col)) for r in g])
            row[f"{col}_mean"] = mean
            row[f"{col}_sd"] = sd
        out_rows.append(row)

    out_fields = list(out_rows[0].keys())
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    md = []
    md.append("# BMAS-E standardized fixed-exit compute profile\n")
    md.append("Values are mean ± sample SD across the three BMAS-E seed run configurations. Compute measurements use batch size 1 on the allocated GPU after warm-up; if archived checkpoints are absent, they are architecture-only measurements. Dice/HD95 are read from the frozen held-out evaluation summaries.\n")
    md.append("| Exit | Dice | HD95 (px) | Latency mean (ms) | p50 (ms) | p95 (ms) | GFLOPs | Peak alloc. VRAM (MB) | Executed params (M) | Latency reduction vs E4 | Speed-up vs E4 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in out_rows:
        md.append(
            f"| E{int(r['exit'])} | "
            f"{fmt(r['dice_mean_mean'], r['dice_mean_sd'], 4)} | "
            f"{fmt(r['hd95_mean_mean'], r['hd95_mean_sd'], 3)} | "
            f"{fmt(r['latency_mean_ms_mean'], r['latency_mean_ms_sd'], 3)} | "
            f"{fmt(r['latency_p50_ms_mean'], r['latency_p50_ms_sd'], 3)} | "
            f"{fmt(r['latency_p95_ms_mean'], r['latency_p95_ms_sd'], 3)} | "
            f"{fmt(r['gflops_mean'], r['gflops_sd'], 3)} | "
            f"{fmt(r['peak_allocated_mb_mean'], r['peak_allocated_mb_sd'], 1)} | "
            f"{fmt(r['executed_params_m_mean'], r['executed_params_m_sd'], 3)} | "
            f"{r['latency_reduction_vs_e4_pct_mean']:.1f}% | "
            f"{r['speedup_vs_e4_x_mean']:.2f}× |"
        )
    md_path = root / "PAPER_TABLE_ANYTIME_PROFILE.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("exit,n_seeds,latency_mean_ms_mean,gflops_mean,peak_allocated_mb_mean")
    for r in out_rows:
        print(f"{r['exit']},{r['n_seeds']},{r['latency_mean_ms_mean']:.6g},{r['gflops_mean']:.6g},{r['peak_allocated_mb_mean']:.6g}")
    print(f"\nWrote: {out_path}")
    print(f"Wrote: {all_csv}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
