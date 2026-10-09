"""V4: which modules each BMAS exit executes, with FLOPs and parameters.

Writes exit_compute.csv (exit, gflops, executed_params, modules) used by the
other stages, and v4_exit_graph.md for the notation in Section 4.1.
Architecture-only (random weights, CPU, FP32): no checkpoint or data needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bmas.model import BMASNet
from experiments.common import md_table, write_csv, write_md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = BMASNet(pretrained_encoder=False).eval()
    x = torch.randn(1, 3, args.image_size, args.image_size)
    top = dict(model.named_children())

    rows = []
    prev = set()
    for e in range(1, 5):
        used = set()
        hooks = [m.register_forward_hook(lambda mod, i, o, n=n: used.add(n)) for n, m in top.items()]
        with torch.inference_mode():
            model(x, max_exit=e)
        for h in hooks:
            h.remove()
        try:
            from torch.utils.flop_counter import FlopCounterMode
            with torch.inference_mode(), FlopCounterMode(display=False) as fc:
                model(x, max_exit=e)
            gflops = fc.get_total_flops() / 1e9
        except Exception as exc:  # pragma: no cover
            print(f"FlopCounterMode unavailable: {exc}")
            gflops = float("nan")
        params = sum(p.numel() for n in used for p in top[n].parameters())
        new = [n for n in top if n in used and n not in prev]
        rows.append({"exit": e, "gflops": gflops, "executed_params": params,
                     "modules": " ".join(n for n in top if n in used), "new_modules": " ".join(new)})
        prev = used
    write_csv(out / "exit_compute.csv", rows)

    g = {r["exit"]: r["gflops"] for r in rows}
    md = md_table(["Exit", "GFLOPs", "Executed params (M)", "Modules added at this exit"],
                  [[f"E{r['exit']}", f"{r['gflops']:.3f}", f"{r['executed_params']/1e6:.3f}", r["new_modules"]]
                   for r in rows])
    text = (
        "# V4 — Exit paths in bmas/model.py\n\n" + md + "\n\n"
        f"E3→E4 adds {g[4]-g[3]:.3f} GFLOPs ({100*(g[4]-g[3])/g[4]:.1f}% of E4).\n\n"
        "Mapping for the manuscript (Fig. 1 names → code):\n\n"
        "- Exit 1: `exit1_head(bottleneck(s5))` → z1 = up(h1(b)).\n"
        "- Exit 2: `dec4(b, s4)` = Fig. 1 block D1 → z2 = z1 + up(h2(d4)), with d4 the output of `dec4`.\n"
        "- Exit 3: `dec3(d4, s3)` = D2 → z3 = z2 + up(h3(d3)).\n"
        "- Exit 4: `dec2` (D3, skip S2) → `dec1` (skip S1) → upsample to input size → `full_refine` → "
        "`exit4_residual` → z4 = z3 + h4(f).\n\n"
        "So the code names (d4, d3) match the original equations; in the revised text write "
        "\"d4 and d3 are the outputs of the decoder blocks that fuse S4 and S3 (D1 and D2 in Fig. 1), and f is "
        "the output of D3, the S1 fusion block, and the full-resolution refinement\". The refinement in Fig. 1 "
        "therefore corresponds to `dec1` + `full_refine`."
    )
    write_md(out / "v4_exit_graph.md", text)
    print(text)


if __name__ == "__main__":
    main()
