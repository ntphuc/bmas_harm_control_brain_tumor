"""Figures of the main paper that come from the pipeline outputs.

Fig. 3 (frontier): regression rate against final Dice for B under damping and the raw
variants, drawn at single-column width. The qualitative figure (Fig. 2) is produced by
experiments.make_figures (stage `figures` of run/05_revision_experiments.sh).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#d62728", "E": "#9467bd"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-out", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-gamma", type=float, default=0.5, help="smallest damping factor shown")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(Path(args.exp_out) / "frontier" / "frontier.csv", encoding="utf-8")))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.7), sharex=True)
    xs_all = []
    for ax, key, lab in ((axes[0], "mvr", "MVR (%)"), (axes[1], "bmvr", "BMVR (%)")):
        b = sorted([r for r in rows if r["variant"] == "B" and float(r["gamma"]) >= args.min_gamma],
                   key=lambda r: float(r["dice_mean"]))
        ax.plot([float(r["dice_mean"]) for r in b], [float(r[f"{key}_mean"]) for r in b], "-o", ms=3,
                color=COLORS["B"], lw=1, label="B, damping γ")
        for r in b:
            g = float(r["gamma"])
            ax.annotate(f"γ={g:g}" if g < 1 else "γ=1", (float(r["dice_mean"]), float(r[f"{key}_mean"])),
                        fontsize=6, textcoords="offset points", xytext=(-30, 2) if g < 1 else (-6, -11))
            xs_all.append(float(r["dice_mean"]))
        for v in "ACDE":
            raw = [x for x in rows if x["variant"] == v and float(x["gamma"]) == 1.0]
            if not raw:
                continue
            r = raw[0]
            ax.plot(float(r["dice_mean"]), float(r[f"{key}_mean"]), "s", ms=4, color=COLORS[v], label=v)
            xs_all.append(float(r["dice_mean"]))
            if v == "E":
                ax.annotate("E", (float(r["dice_mean"]), float(r[f"{key}_mean"])), fontsize=6.5,
                            textcoords="offset points", xytext=(4, -3))
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, lw=0.5)
    if xs_all:
        pad = 0.15 * (max(xs_all) - min(xs_all) + 1e-6)
        axes[1].set_xlim(min(xs_all) - pad, max(xs_all) + pad)
    axes[1].set_xlabel("Final Dice (E4)")
    axes[0].legend(loc="lower right", frameon=False, ncol=3, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "fig_frontier_col.pdf")
    fig.savefig(out / "fig_frontier_col.png", dpi=300)
    print(f"wrote {out / 'fig_frontier_col.pdf'}")


if __name__ == "__main__":
    main()
