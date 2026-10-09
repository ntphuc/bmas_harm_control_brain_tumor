"""V1 + X8 metadata: match every manifest sample to an original BRISC file.

The preprocessed NPY export used for training lost the original filenames.
BRISC filenames encode the official split, tumor type and imaging plane
(``brisc2025_<split>_<index>_<tumor>_<plane>_<seq>``). We recover them by
nearest-neighbour matching of 64x64 image signatures (cosine similarity),
verified with the mask Dice of the matched pair.

Outputs (in --output-dir):
  case_metadata.csv        case_id, our split, official split, tumor/plane codes, match quality
  split_contingency.md     our split x official split counts + verdict for V1
  v1_summary.json

Needs NumPy + OpenCV (no torch).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from bmas.data import BRISCSegmentationDataset
from experiments.common import md_table, parse_brisc_filename, write_csv, write_json, write_md

SIG = 64


def _sig_image(img01: np.ndarray) -> np.ndarray:
    small = cv2.resize(img01.astype(np.float32), (SIG, SIG), interpolation=cv2.INTER_AREA).ravel()
    small = small - small.mean()
    n = np.linalg.norm(small)
    return small / n if n > 0 else small


def _sig_mask(mask01: np.ndarray) -> np.ndarray:
    return cv2.resize(mask01.astype(np.float32), (SIG, SIG), interpolation=cv2.INTER_NEAREST).ravel() > 0.5


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else float(2.0 * np.logical_and(a, b).sum() / s)


SKIP_DIRS = {"site-packages", "dist-packages", "__pycache__", ".git", "node_modules", "nnUNet_raw",
             "nnUNet_preprocessed", "nnUNet_results", "outputs", "preprocessed"}


def collect_raw(raw_roots) -> Dict[str, Tuple[Path, Path]]:
    """stem -> (image_path, mask_path) for BRISC segmentation pairs found under any root.

    Virtual environments, caches and pipeline outputs are pruned so that a broad root
    such as the datasets folder can be scanned quickly.
    """
    images: Dict[str, Path] = {}
    masks: Dict[str, Path] = {}
    for root in raw_roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(("venv", ".venv"))]
            for fn in filenames:
                if not fn.startswith("brisc2025_"):
                    continue
                path = Path(dirpath) / fn
                ext = path.suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    images.setdefault(path.stem, path)
                elif ext == ".png":
                    masks.setdefault(path.stem, path)
    return {s: (images[s], masks[s]) for s in images if s in masks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--raw-root", nargs="+", required=True,
                    help="one or more folders searched for original brisc2025_*.jpg/png files")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-similarity", type=float, default=0.97)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = []  # (case_id, our_split, sig, mask_sig, lesion_pixels, locator)
    direct_meta = 0
    for split in args.splits:
        try:
            ds = BRISCSegmentationDataset(args.manifest, split, 256, train=False, imagenet_norm=False,
                                          data_root=args.data_root)
        except ValueError as exc:
            print(f"[v1] split {split!r} skipped: {exc}")
            continue
        for i in range(len(ds)):
            img, mask, meta = ds.load_numpy(i)
            parsed = parse_brisc_filename(meta["image_path"])
            if parsed:
                direct_meta += 1
            samples.append((meta["case_id"], split, _sig_image(img), _sig_mask(mask), float(mask.sum()),
                            meta["image_path"], parsed))
        print(f"[v1] loaded split={split} n={len(ds)}")

    rows: List[Dict] = []
    if samples and direct_meta == len(samples):
        print("[v1] manifest already references original BRISC filenames; no image matching needed")
        for cid, split, _, _, lp, loc, parsed in samples:
            rows.append({"case_id": cid, "our_split": split, "matched_file": Path(loc).name, "similarity": 1.0,
                         "margin": "", "mask_dice": 1.0, "lesion_pixels": lp, "match_ok": True, **parsed})
    else:
        raw = collect_raw(args.raw_root)
        if not raw:
            print(f"[v1] No brisc2025_* image/mask pairs found under {args.raw_root}. Download the original "
                  "BRISC 2025 release (segmentation_task with images/ and masks/) and set RAW_BRISC_ROOT to "
                  "its folder.", flush=True)
            sys.exit(3)
        stems = sorted(raw)
        print(f"[v1] raw BRISC segmentation pairs found: {len(stems)}")
        raw_sig = np.zeros((len(stems), SIG * SIG), dtype=np.float32)
        raw_mask = np.zeros((len(stems), SIG * SIG), dtype=bool)
        for j, s in enumerate(stems):
            img = cv2.imread(str(raw[s][0]), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(raw[s][1]), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            raw_sig[j] = _sig_image(img.astype(np.float32) / 255.0)
            raw_mask[j] = _sig_mask((msk > 127).astype(np.float32))
        man_sig = np.stack([s[2] for s in samples]).astype(np.float32)
        sim = man_sig @ raw_sig.T
        best = sim.argmax(axis=1)
        top2 = np.partition(sim, -2, axis=1)[:, -2:] if sim.shape[1] > 1 else np.concatenate([sim, sim], 1)
        for i, (cid, split, _, msig, lp, loc, _) in enumerate(samples):
            j = int(best[i])
            s_best = float(sim[i, j])
            margin = float(top2[i].max() - top2[i].min())
            parsed = parse_brisc_filename(stems[j]) or {}
            md = _dice(msig, raw_mask[j])
            rows.append({"case_id": cid, "our_split": split, "matched_file": stems[j], "similarity": s_best,
                         "margin": margin, "mask_dice": md, "lesion_pixels": lp,
                         "match_ok": bool(s_best >= args.min_similarity and md >= 0.8), **parsed})
        dup = Counter(r["matched_file"] for r in rows)
        for r in rows:
            r["duplicate_match"] = dup[r["matched_file"]] > 1

    write_csv(out / "case_metadata.csv", rows)

    ok = [r for r in rows if r["match_ok"]]
    cont: Dict[str, Counter] = defaultdict(Counter)
    for r in ok:
        cont[r["our_split"]][r.get("official_split", "?")] += 1
    md_rows = [[s, cont[s].get("train", 0), cont[s].get("test", 0), sum(cont[s].values())] for s in args.splits if s in cont]
    n_dup = sum(1 for r in rows if r.get("duplicate_match"))
    test_ok = cont.get("test", Counter())
    dev = Counter()
    for s in args.splits:
        if s != "test":
            dev.update(cont.get(s, Counter()))
    n_test = sum(1 for r in rows if r["our_split"] == "test")
    if n_test and test_ok.get("test", 0) == n_test and dev.get("test", 0) == 0:
        verdict = ("OFFICIAL: every held-out image matches an official BRISC test file and no development image "
                   "comes from the official test set -> use Option 1 in Section 3.1.")
    else:
        verdict = (f"NOT OFFICIAL (or incomplete match): held-out images matched to official test = "
                   f"{test_ok.get('test', 0)}/{n_test}; development images from official test = {dev.get('test', 0)}. "
                   "Use Option 2 in Section 3.1 and state the reason.")
    tumor = Counter((r["our_split"], r.get("tumor_code", "?")) for r in ok)
    tumor_rows = [[s, t, n] for (s, t), n in sorted(tumor.items())]
    text = (
        "# V1 — Split verification against original BRISC filenames\n\n"
        f"Samples: {len(rows)}; confident matches: {len(ok)} (cosine ≥ {args.min_similarity}, mask Dice ≥ 0.8); "
        f"many-to-one matches: {n_dup}.\n\n"
        + md_table(["Our split", "Official train", "Official test", "Matched"], md_rows)
        + f"\n\n**Verdict:** {verdict}\n\n"
        + "Tumor-type counts among confident matches (for X8):\n\n"
        + md_table(["Our split", "Tumor code", "n"], tumor_rows)
    )
    write_md(out / "split_contingency.md", text)
    write_json({"n_samples": len(rows), "n_confident": len(ok), "n_many_to_one": n_dup, "verdict": verdict,
                "contingency": {k: dict(v) for k, v in cont.items()}}, out / "v1_summary.json")
    print(text)


if __name__ == "__main__":
    main()
