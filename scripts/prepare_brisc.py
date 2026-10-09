"""Build data/brisc2025_manifest.csv from the original BRISC 2025 release.

Expected layout (as distributed): <brisc_root>/segmentation_task/{train,test}/{images,masks}/
with files named brisc2025_<split>_<index>_<tumor>_<plane>_<sequence>.jpg/.png.

Split protocol of the paper: the BRISC test images form the held-out set; the
development images are split once with seed 42 into training and 590
validation images, and this partition is reused for every training seed.
The manifest uses the raw backend of bmas.data (paths relative to --data-root).
"""
from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

PAT = re.compile(r"brisc2025_(train|test)_(\d+)_([a-z]{2})_([a-z]{2})_([A-Za-z0-9]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="folder that contains segmentation_task/")
    ap.add_argument("--output", default="data/brisc2025_manifest.csv")
    ap.add_argument("--val-size", type=int, default=590)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = Path(args.data_root)

    images = {p.stem: p for p in root.rglob("brisc2025_*") if p.suffix.lower() in (".jpg", ".jpeg")}
    masks = {p.stem: p for p in root.rglob("brisc2025_*") if p.suffix.lower() == ".png"}
    pairs = sorted(s for s in images if s in masks)
    if not pairs:
        raise SystemExit(f"No brisc2025_* image/mask pairs found under {root}")

    rows = {"train": [], "test": []}
    for stem in pairs:
        m = PAT.search(stem)
        if not m:
            continue
        rows[m.group(1)].append({"case_id": stem, "image_path": str(images[stem].relative_to(root)),
                                 "mask_path": str(masks[stem].relative_to(root)), "tumor_code": m.group(3),
                                 "plane_code": m.group(4), "data_backend": "raw"})
    dev = rows["train"]
    rng = random.Random(args.seed)
    order = list(range(len(dev)))
    rng.shuffle(order)
    val_idx = set(order[: args.val_size])
    out = []
    for i, r in enumerate(dev):
        out.append({"split": "val" if i in val_idx else "train", **r})
    out += [{"split": "test", **r} for r in rows["test"]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    counts = {s: sum(1 for r in out if r["split"] == s) for s in ("train", "val", "test")}
    print(f"wrote {args.output}: {counts}")
    if counts != {"train": 3343, "val": 590, "test": 860}:
        print("NOTE: counts differ from the paper (3343/590/860). Check the BRISC version, "
              "or pass the authors' original manifest via MANIFEST=... instead.")


if __name__ == "__main__":
    main()
