"""Locate the archived run directories of variants A-E and the endpoint baselines.

Output: ``variant_map.json`` + ``variant_map.tsv`` describing, per variant and
seed, the run directory, config, frozen held-out per-case CSV, summary JSON and
checkpoint (if any).

Resolution order
  1. ``--override`` TSV (``variant<TAB>run_dir_template``; ``{seed}`` is substituted).
  2. Automatic scan of ``--roots`` for directories named ``<prefix>_seed<seed>``.
     Each prefix is labelled from its config (experiment_name, model_type,
     boundary/monotonic weights). If a label has several candidate prefixes, the
     one whose frozen Exit-4 Dice is closest to the submitted manuscript is kept
     and the choice is printed so it can be audited.

Pure Python: no NumPy needed.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.common import (
    BASELINES, BMAS_VARIANTS, COMPETITORS, PAPER_EXIT4_DICE, is_per_case_exit_table, read_json, to_float,
    write_csv, write_json,
)

CKPT_NAMES = ("best.pt", "best_model.pt", "checkpoint_best.pt", "model_best.pt", "best.pth", "last.pt", "last.pth")
CONFIG_NAMES = ("config_resolved.json", "config.json", "config_resolved.yaml", "config.yaml")


def find_checkpoint(run_dir: Path) -> Optional[Path]:
    for n in CKPT_NAMES:
        if (run_dir / n).is_file():
            return run_dir / n
    hits = sorted(list(run_dir.rglob("*.pt")) + list(run_dir.rglob("*.pth")))
    hits = [h for h in hits if len(h.relative_to(run_dir).parts) <= 3]
    return hits[0] if hits else None


def find_config(run_dir: Path) -> Optional[Path]:
    for n in CONFIG_NAMES:
        if (run_dir / n).is_file():
            return run_dir / n
    return None


def _eval_dir_rank(d: Path) -> int:
    n = d.name.lower()
    if "val" in n and "test" not in n:
        return 99
    if "official" in n:
        return 0
    if "paper" in n:
        return 1
    if n.startswith("eval_test"):
        return 2
    if "test" in n:
        return 3
    return 50


def find_frozen_eval(run_dir: Path) -> tuple[Optional[Path], Optional[Path]]:
    """Return (per_case_csv, summary_json) of the frozen held-out evaluation."""
    eval_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.lower().startswith("eval")]
    eval_dirs.sort(key=lambda d: (_eval_dir_rank(d), d.name))
    for d in eval_dirs:
        if _eval_dir_rank(d) >= 99:
            continue
        csvs = sorted(d.rglob("*.csv"), key=lambda p: (0 if "per_case" in p.name else 1, len(p.parts), p.name))
        for c in csvs:
            if is_per_case_exit_table(c):
                summ = d / "summary.json"
                return c, (summ if summ.is_file() else None)
    for d in eval_dirs:
        if _eval_dir_rank(d) < 99 and (d / "summary.json").is_file():
            return None, d / "summary.json"
    return None, None


def summary_exit4_dice(summary: Optional[Path]) -> Optional[float]:
    if not summary:
        return None
    try:
        s = read_json(summary)
    except Exception:
        return None
    for key in ("exit4_dice_mean", "e4_dice_mean", "dice_mean", "exit4_dice"):
        if key in s:
            return to_float(s[key])
    return None


def label_from_config(prefix: str, cfg: Dict[str, Any]) -> Optional[str]:
    for key in ("paper_variant", "variant"):
        v = str(cfg.get(key, "")).strip().upper()
        if v in BMAS_VARIANTS:
            return v
    name = (prefix + " " + str(cfg.get("experiment_name", ""))).lower()
    if "mess_style_mess_stage2" in name:
        return "MESS"
    if "adpc_style_joint" in name:
        return "ADPC"
    if "baseline_effb0" in name or "efficientnet_unet" in name or "effb0_unet" in name:
        return "EffB0UNet"
    if "deeplab" in name:
        return "DeepLabV3"
    if "baseline_unet" in name:
        return "UNet"
    mt = str(cfg.get("model_type", "bmas")).lower()
    if mt in {"unet", "u-net"}:
        return "UNet"
    if mt.startswith("deeplab"):
        return "DeepLabV3"
    if mt in {"efficientnet_unet", "effb0_unet", "efficientnet-b0-unet"}:
        return "EffB0UNet"
    if any(t in name for t in ("competitor", "mess", "adpc")):
        return None
    flat = " ".join(f"{k}={v}" for k, v in cfg.items()).lower()
    if "preserve_dominant" in name or "cdoubleprime" in name or "preserve_dominant" in flat:
        return "E"
    if re.search(r"cprime(?!_?preserve)", name) or "harm" in name or "harm" in flat:
        return "D"
    mono = to_float(cfg.get("monotonic_weight")) or 0.0
    bnd = to_float(cfg.get("boundary_weight")) or 0.0
    if mono > 0:
        return "C"
    if bnd > 0:
        return "B"
    if "boundary_weight" in cfg:
        return "A"
    return None


def scan(roots: List[Path], seeds: List[int]) -> Dict[str, Dict[int, Path]]:
    pat = re.compile(r"^(?P<prefix>.+?)_seed(?P<seed>\d+)$")
    groups: Dict[str, Dict[int, Path]] = defaultdict(dict)
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.rglob("*"):
            if not d.is_dir() or len(d.relative_to(root).parts) > 3:
                continue
            m = pat.match(d.name)
            if m and int(m.group("seed")) in seeds:
                key = str(d.parent / m.group("prefix"))
                groups[key].setdefault(int(m.group("seed")), d)
    return groups


def describe_seed_dir(run_dir: Path) -> Dict[str, Any]:
    cfg_path = find_config(run_dir)
    csv_path, summ = find_frozen_eval(run_dir)
    ckpt = find_checkpoint(run_dir)
    return {
        "run_dir": str(run_dir),
        "config": str(cfg_path) if cfg_path else "",
        "per_case_csv": str(csv_path) if csv_path else "",
        "summary": str(summ) if summ else "",
        "checkpoint": str(ckpt) if ckpt else "",
        "exit4_dice": summary_exit4_dice(summ),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 2026, 3407])
    p.add_argument("--override", default="", help="TSV: variant<TAB>run_dir_template with {seed}")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    result: Dict[str, Any] = {"seeds": seeds, "variants": {}, "notes": []}

    templates: Dict[str, str] = {}
    if args.override and Path(args.override).is_file():
        for line in Path(args.override).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\t+|\s{2,}", line)
            if len(parts) >= 2:
                templates[parts[0].strip()] = parts[1].strip()
        result["notes"].append(f"override file used: {args.override}")

    if templates:
        for variant, tmpl in templates.items():
            per_seed = {}
            for s in seeds:
                d = Path(tmpl.format(seed=s))
                if d.is_dir():
                    per_seed[str(s)] = describe_seed_dir(d)
            result["variants"][variant] = {"prefix": tmpl, "source": "override", "seeds": per_seed}
    else:
        groups = scan([Path(r) for r in args.roots], seeds)
        candidates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        all_rows = []
        for prefix, seed_dirs in sorted(groups.items()):
            any_dir = next(iter(seed_dirs.values()))
            cfg_path = find_config(any_dir)
            cfg: Dict[str, Any] = {}
            if cfg_path and cfg_path.suffix == ".json":
                try:
                    cfg = read_json(cfg_path)
                except Exception:
                    cfg = {}
            label = label_from_config(Path(prefix).name, cfg)
            per_seed = {str(s): describe_seed_dir(d) for s, d in sorted(seed_dirs.items())}
            dices = [v["exit4_dice"] for v in per_seed.values() if v["exit4_dice"] is not None]
            mean_dice = sum(dices) / len(dices) if dices else None
            row = {"prefix": prefix, "label": label or "", "n_seeds": len(seed_dirs),
                   "mean_exit4_dice": mean_dice, "experiment_name": cfg.get("experiment_name", ""),
                   "has_per_case": all(v["per_case_csv"] for v in per_seed.values()),
                   "has_checkpoint": all(v["checkpoint"] for v in per_seed.values())}
            all_rows.append(row)
            if label:
                candidates[label].append({"prefix": prefix, "seeds": per_seed, "mean_dice": mean_dice,
                                          "n_seeds": len(seed_dirs)})
        write_csv(out / "all_candidate_runs.csv", all_rows)

        for label, cands in candidates.items():
            ref = PAPER_EXIT4_DICE.get(label)

            def score(c):
                complete = -c["n_seeds"]
                diff = abs(c["mean_dice"] - ref) if (ref is not None and c["mean_dice"] is not None) else 9.0
                return (complete, diff)

            cands.sort(key=score)
            best = cands[0]
            diff = (abs(best["mean_dice"] - ref) if (ref is not None and best["mean_dice"] is not None) else None)
            result["variants"][label] = {
                "prefix": best["prefix"] + "_seed{seed}",
                "source": "auto",
                "n_candidates": len(cands),
                "other_candidates": [c["prefix"] for c in cands[1:]],
                "paper_exit4_dice": ref,
                "found_exit4_dice": best["mean_dice"],
                "abs_diff_vs_paper": diff,
                "seeds": best["seeds"],
            }
            if len(cands) > 1:
                result["notes"].append(
                    f"{label}: {len(cands)} candidate prefixes; chose {best['prefix']} "
                    f"(|Dice-paper|={diff}). Others: {[c['prefix'] for c in cands[1:]]}")
            if diff is not None and diff > 0.0006:
                result["notes"].append(
                    f"WARNING {label}: chosen run Exit-4 Dice {best['mean_dice']:.4f} differs from manuscript "
                    f"{ref:.4f} by {diff:.4f}. Check, or pin the run with VARIANT_MAP.")

    missing = [v for v in BMAS_VARIANTS if v not in result["variants"]]
    if missing:
        result["notes"].append(f"Variants not found: {missing}. Provide VARIANT_MAP to pin them.")
    for v in list(BASELINES[:3]) + list(COMPETITORS):
        if v not in result["variants"]:
            result["notes"].append(f"Baseline not found (paired endpoint tests will skip it): {v}")

    write_json(result, out / "variant_map.json")
    rows = []
    for variant, info in result["variants"].items():
        for s, d in info["seeds"].items():
            rows.append({"variant": variant, "seed": s, **{k: d[k] for k in
                         ("run_dir", "per_case_csv", "summary", "checkpoint", "exit4_dice")}})
    write_csv(out / "variant_map.tsv".replace(".tsv", ".csv"), rows)
    (out / "variant_map.tsv").write_text(
        "\n".join(f"{v}\t{i['prefix']}" for v, i in result["variants"].items()) + "\n", encoding="utf-8")

    print("variant\tseeds\tper_case\tcheckpoint\texit4_dice\tprefix")
    for variant, info in sorted(result["variants"].items()):
        seeds_found = sorted(info["seeds"])
        pc = sum(1 for d in info["seeds"].values() if d["per_case_csv"])
        ck = sum(1 for d in info["seeds"].values() if d["checkpoint"])
        print(f"{variant}\t{','.join(seeds_found)}\t{pc}/{len(seeds_found)}\t{ck}/{len(seeds_found)}\t"
              f"{info.get('found_exit4_dice')}\t{info['prefix']}")
    for n in result["notes"]:
        print("NOTE:", n)


if __name__ == "__main__":
    main()
