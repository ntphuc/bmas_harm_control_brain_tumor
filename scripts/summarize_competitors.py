from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = [
    "dynamic_dice_mean",
    "dynamic_iou_mean",
    "dynamic_hd95_mean",
    "dynamic_assd_mean",
    "e4_dice_mean",
    "e4_hd95_mean",
    "e4_assd_mean",
    "dice_mvr_percent",
    "hd95_bmvr_percent",
    "e3_to_e4_hd95_violation_percent",
]


def _fmt(vals: list[float]) -> str:
    a = np.asarray(vals, dtype=float)
    if len(a) == 1:
        return f"{a[0]:.4f}"
    return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="outputs/competitors")
    p.add_argument("--output-csv", default="outputs/competitors/competitor_summary_3seeds.csv")
    args = p.parse_args()

    root = Path(args.out_root)
    methods = {
        "MESS-style": sorted(root.glob("mess_style_mess_stage2_seed*/eval_test_calibrated/summary.json")),
        "ADP-C-style": sorted(root.glob("adpc_style_joint_seed*/eval_test_calibrated/summary.json")),
    }

    rows = []
    for name, files in methods.items():
        if not files:
            print(f"warning: no test summaries found for {name}")
            continue
        summaries = [json.loads(f.read_text()) for f in files]
        row = {"method": name, "n_seeds": len(summaries)}
        for key in KEYS:
            vals = [float(s[key]) for s in summaries if key in s]
            if vals:
                row[key] = _fmt(vals)
                row[f"{key}_raw"] = ";".join(f"{v:.8f}" for v in vals)
        if name == "MESS-style":
            exits = [float(s.get("dynamic_exit_mean", np.nan)) for s in summaries]
            exits = [x for x in exits if np.isfinite(x)]
            if exits:
                row["dynamic_exit_mean"] = _fmt(exits)
        if name == "ADP-C-style":
            actives = [s.get("active_fraction_mean_e2_e3_e4") for s in summaries]
            actives = [a for a in actives if a is not None]
            if actives:
                arr = np.asarray(actives, dtype=float)
                for i, exit_id in enumerate((2, 3, 4)):
                    row[f"active_fraction_e{exit_id}"] = _fmt(arr[:, i].tolist())
        rows.append(row)

    df = pd.DataFrame(rows)
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
