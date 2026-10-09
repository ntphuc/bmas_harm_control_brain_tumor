from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Select ADP-C-style threshold using validation results only.")
    p.add_argument("--summary", action="append", required=True, help="path to an eval summary.json; repeat for each threshold")
    p.add_argument("--max-dice-drop", type=float, default=0.001, help="allowed absolute Dice drop from E4")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    candidates = []
    for path_str in args.summary:
        path = Path(path_str)
        s = json.loads(path.read_text())
        threshold = float(s["adpc_threshold"])
        dynamic_dice = float(s["dynamic_dice_mean"])
        e4_dice = float(s["e4_dice_mean"])
        active = s.get("active_fraction_mean_e2_e3_e4", [1.0, 1.0, 1.0])
        mean_active = float(sum(active) / len(active))
        feasible = dynamic_dice >= e4_dice - args.max_dice_drop
        candidates.append({
            "threshold": threshold,
            "dynamic_dice": dynamic_dice,
            "e4_dice": e4_dice,
            "mean_active_fraction": mean_active,
            "feasible": feasible,
            "source": str(path),
        })

    feasible = [c for c in candidates if c["feasible"]]
    if feasible:
        # Minimum active fraction among settings that stay within the validation Dice tolerance.
        best = min(feasible, key=lambda c: (c["mean_active_fraction"], -c["dynamic_dice"], c["threshold"]))
        selection_rule = "minimum mean active fraction subject to validation Dice >= E4 - max_dice_drop"
    else:
        # Conservative fallback: best validation dynamic Dice, then lower active fraction.
        best = max(candidates, key=lambda c: (c["dynamic_dice"], -c["mean_active_fraction"]))
        selection_rule = "fallback: best validation dynamic Dice; no threshold met the Dice-drop constraint"

    result = {
        "selected_threshold": best["threshold"],
        "max_dice_drop": args.max_dice_drop,
        "selection_rule": selection_rule,
        "selected": best,
        "candidates": candidates,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
