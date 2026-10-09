from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _choose_mess(df: pd.DataFrame, max_drop: float):
    """
    Validation-only threshold calibration.

    Original version failed when no threshold could satisfy the very strict
    Dice constraint. This version keeps the constraint as the preferred rule,
    but provides a safe fallback: select the threshold with the smallest Dice
    degradation and then the lowest average exit depth.

    This avoids stopping the whole experiment pipeline while still reporting
    whether the requested constraint was achievable.
    """
    e4 = float(df["e4_dice"].mean())

    best_feasible = None
    best_fallback = None

    # Include very high confidence values because segmentation confidence can
    # be saturated by dominant background pixels.
    thresholds = np.concatenate([
        np.linspace(0.50, 0.999, 250),
        np.linspace(0.999, 0.99999, 50)
    ])

    conf_cache = {
        k: df[f"e{k}_image_confidence"].to_numpy()
        for k in (1, 2, 3)
    }

    for t in thresholds:
        exits = np.full(len(df), 4, dtype=int)

        # earlier exit is accepted only when confidence condition is satisfied
        for k in (1, 2, 3):
            take = (exits == 4) & (conf_cache[k] >= t)
            exits[take] = k

        dice = np.array(
            [df.iloc[i][f"e{exits[i]}_dice"] for i in range(len(df))],
            dtype=float
        )

        mean_dice = float(dice.mean())
        mean_exit = float(exits.mean())
        drop = float(e4 - mean_dice)

        # Preferred candidate: satisfy quality constraint, minimize compute
        if drop <= max_drop:
            candidate = (mean_exit, -mean_dice, t, mean_dice, drop)
            if best_feasible is None or candidate < best_feasible:
                best_feasible = candidate

        # Fallback: minimize Dice degradation first, then compute
        fallback = (drop, mean_exit, -mean_dice, t, mean_dice)
        if best_fallback is None or fallback < best_fallback:
            best_fallback = fallback

    if best_feasible is not None:
        _, _, t, mean_dice, drop = best_feasible
        feasible = True
    else:
        drop, _, _, t, mean_dice = best_fallback
        feasible = False

    return {
        "thresholds": [float(t), float(t), float(t)],
        "val_dynamic_dice": float(mean_dice),
        "val_e4_dice": float(e4),
        "dice_drop": float(e4 - mean_dice),
        "requested_max_drop": float(max_drop),
        "constraint_satisfied": feasible,
        "note": (
            "Found feasible threshold."
            if feasible
            else "No threshold satisfied max Dice drop constraint; "
                 "reported closest validation operating point."
        )
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-case-csv", required=True)
    p.add_argument("--method", choices=["mess_style"], required=True)
    p.add_argument(
        "--max-dice-drop",
        type=float,
        default=0.001,
        help="absolute Dice, e.g. 0.001 = 0.1 pp"
    )
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.per_case_csv)
    result = _choose_mess(df, args.max_dice_drop)

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
