"""Figures for the revision (NumPy + Matplotlib).

Fig. 4 (X7): two held-out cases at E3 and E4 for B, D and E
    (a) D adds a late HD95 regression that E avoids,
    (b) E leaves an error at E4 that D corrects.
    Candidate lists are written so the authors can pick other cases (--cases).
Fig. 5 (X1/X2): regression rate versus normalised area under the GFLOPs-Dice curve.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import HD95_TOL, load_case_table, read_csv_dicts, to_float, write_csv  # noqa: E402

COLS = ["B", "D", "E"]


def rss_mb() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_masks(path: Path, wanted: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Unpack only the requested cases; the full arrays are released before returning."""
    out: Dict[str, Dict[str, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as z:
        h, w = (int(v) for v in z["shape"])
        index = {str(c): i for i, c in enumerate(z["case_id"].tolist())}
        rows = {cid: index[cid] for cid in wanted if cid in index}
        for key in ("image", "gt", "e3", "e4"):
            arr = z[key]
            for cid, i in rows.items():
                item = arr[i]
                if key != "image":
                    item = np.unpackbits(item)[: h * w].reshape(h, w).astype(bool)
                out.setdefault(cid, {})[key] = np.array(item, copy=True)
            del arr
    gc.collect()
    return out


def select_cases(tabs: Dict[str, dict], top: int = 10):
    common = set.intersection(*(set(t) for t in tabs.values()))
    a, b = [], []
    for c in common:
        dB, dD, dE = (tabs[v][c] for v in COLS)
        dh_D = dD["e4_hd95"] - dD["e3_hd95"]
        dh_E = dE["e4_hd95"] - dE["e3_hd95"]
        if dh_D > HD95_TOL and dh_E <= HD95_TOL:
            a.append({"case_id": c, "score": dh_D - max(dh_E, 0.0), "D_dHD95": dh_D, "E_dHD95": dh_E,
                      "D_new_cc": to_float(dD.get("e34_new_cc")) or 0.0})
        gain_D = dD["e4_dice"] - dD["e3_dice"]
        gain_E = dE["e4_dice"] - dE["e3_dice"]
        if dD["e4_dice"] > dE["e4_dice"] + 0.02 and gain_D > gain_E:
            b.append({"case_id": c, "score": (dD["e4_dice"] - dE["e4_dice"]) + (gain_D - gain_E),
                      "D_e4_dice": dD["e4_dice"], "E_e4_dice": dE["e4_dice"]})
    a.sort(key=lambda r: (-(r["D_new_cc"] > 0), -r["score"]))
    b.sort(key=lambda r: -r["score"])
    return a[:top], b[:top]


def draw_fig4(cases: List[str], tabs, masks, out: Path, labels: List[str]) -> None:
    fig, axes = plt.subplots(len(cases), 6, figsize=(13.5, 2.5 * len(cases) + 0.4), squeeze=False)
    for r, cid in enumerate(cases):
        for j, v in enumerate(COLS):
            m = masks[v][cid]
            for e_i, ex in enumerate(("e3", "e4")):
                ax = axes[r][2 * j + e_i]
                ax.imshow(m["image"], cmap="gray", interpolation="nearest")
                if m["gt"].any():
                    ax.contour(m["gt"], levels=[0.5], colors="#2ca02c", linewidths=1.6)
                if m[ex].any():
                    ax.contour(m[ex], levels=[0.5], colors="#d62728", linewidths=0.8)
                rec = tabs[v][cid]
                k = 3 if ex == "e3" else 4
                ax.set_title(f"{v} E{k}  Dice {rec[f'e{k}_dice']:.3f}  HD95 {rec[f'e{k}_hd95']:.1f}", fontsize=7.5)
                ax.set_xticks([])
                ax.set_yticks([])
        axes[r][0].set_ylabel(labels[r], fontsize=9)
    fig.suptitle("Reference contour green, prediction red", fontsize=9, y=0.995)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig4_qualitative.{ext}", dpi=300)
    plt.close(fig)


def draw_fig5(x1_csv: Path, out: Path) -> bool:
    if not x1_csv.is_file():
        return False
    rows = [r for r in read_csv_dicts(x1_csv) if r["variant"] in "ABCDE" and r.get("mvr_mean")]
    if not rows:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, key, title in ((axes[0], "mvr", "MVR (%)"), (axes[1], "bmvr", "BMVR (%)")):
        for r in rows:
            x, xe = float(r["auc_gflops_dice_mean"]), float(r["auc_gflops_dice_sd"] or 0)
            y, ye = float(r[f"{key}_mean"]), float(r[f"{key}_sd"] or 0)
            ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", ms=5, capsize=2, color="#1f77b4")
            ax.annotate(r["variant"], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=9)
        ax.set_xlabel("Normalized area under GFLOPs–Dice curve")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig5_regression_vs_anytime_quality.{ext}", dpi=300)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-root", required=True)
    ap.add_argument("--analysis-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cases", nargs="*", default=[], help="override: case ids for rows (a) and (b)")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.infer_root)

    made5 = draw_fig5(Path(args.analysis_dir) / "x1_per_exit.csv", out)
    plt.close("all")
    gc.collect()
    print("[fig] Fig. 5", "written" if made5 else "skipped (x1_per_exit.csv missing)",
          f"(peak RSS {rss_mb():.0f} MB)", flush=True)

    tabs = {}
    for v in COLS:
        base = root / v / f"seed{args.seed}" / "test"
        if not (base / "per_case_raw.csv").is_file() or not (base / "masks_test.npz").is_file():
            print(f"[fig] Fig. 4 skipped: missing inference outputs for {v} seed{args.seed}")
            return
        tabs[v] = load_case_table(base / "per_case_raw.csv")
    print(f"[fig] per-case tables loaded (peak RSS {rss_mb():.0f} MB)", flush=True)
    cand_a, cand_b = select_cases(tabs)
    write_csv(out / "fig4_candidates_a_D_regresses_E_does_not.csv", cand_a)
    write_csv(out / "fig4_candidates_b_D_corrects_E_does_not.csv", cand_b)
    names = ["(a) late regression", "(b) uncorrected error"]
    if args.cases:
        pairs = list(zip(names, args.cases[:2]))
    else:
        pairs = [(n, c[0]["case_id"]) for n, c in zip(names, (cand_a, cand_b)) if c]
    if not pairs:
        print("[fig] Fig. 4 skipped: no case satisfies the selection rules")
        return
    labels = [p[0] for p in pairs]
    cases = [p[1] for p in pairs]
    masks = {}
    for v in COLS:
        masks[v] = load_masks(root / v / f"seed{args.seed}" / "test" / "masks_test.npz", cases)
        missing = [c for c in cases if c not in masks[v]]
        if missing:
            print(f"[fig] Fig. 4 skipped: cases {missing} not in the saved masks of {v}")
            return
    print(f"[fig] masks for {cases} loaded (peak RSS {rss_mb():.0f} MB)", flush=True)
    draw_fig4(cases, tabs, masks, out, labels)
    print(f"[fig] Fig. 4 written with cases {cases}")


if __name__ == "__main__":
    main()
