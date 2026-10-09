"""X14: build an external held-out set for zero-shot evaluation under distribution shift.

Mode ``brats``: BraTS-style subject folders with a post-contrast T1 volume
(``*t1ce.nii.gz`` or ``*t1c.nii.gz``) and a label volume (``*seg.nii.gz``).
For each subject the axial slice(s) with the largest tumor area are exported:
    * tumor definition ``tc`` (default) = all labels except edema (label 2),
      i.e. necrotic core + enhancing tumor, which is what is visible on T1c in
      both the 2021 (labels 1/2/4) and 2023 (labels 1/2/3) conventions; ``wt`` = all labels;
    * intensities clipped to the 0.5-99.5 percentiles of brain voxels and scaled to 0-255;
    * radiological display orientation (anterior up, patient right on image left).
One slice per subject (default) keeps the external set subject-independent.

Mode ``folder``: existing 2-D image/mask files paired by file stem.

Output: <out>/images/*.png, <out>/masks/*.png (0/255) and <out>/manifest.csv with
split=test and raw-backend columns, usable directly by BRISCSegmentationDataset.
Needs NumPy + OpenCV (+ nibabel for BraTS).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from experiments.common import write_csv, write_json

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _find(folder: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(folder.glob(pat))
        if hits:
            return hits[0]
    return None


def _to_display(slice2d: np.ndarray) -> np.ndarray:
    # nibabel canonical (RAS) array[x, y]: rows = anterior at top, columns = patient right on image left
    return np.ascontiguousarray(slice2d.T[::-1, ::-1])


def prepare_brats(src: Path, out: Path, label: str, per_subject: int, min_area: int, gap: int,
                  max_subjects: int) -> List[dict]:
    import nibabel as nib

    subjects = sorted(d for d in src.rglob("*") if d.is_dir() and _find(d, ["*seg.nii*"]))
    if max_subjects > 0:
        subjects = subjects[:max_subjects]
    rows = []
    for d in subjects:
        t1c = _find(d, ["*t1ce.nii*", "*t1c.nii*", "*T1c*.nii*", "*t1Gd*.nii*"])
        seg = _find(d, ["*seg.nii*"])
        if t1c is None or seg is None:
            continue
        img = np.asanyarray(nib.as_closest_canonical(nib.load(str(t1c))).dataobj).astype(np.float32)
        lab = np.asanyarray(nib.as_closest_canonical(nib.load(str(seg))).dataobj).astype(np.int16)
        tumor = (lab > 0) if label == "wt" else ((lab > 0) & (lab != 2))
        brain = img > 0
        if not brain.any() or not tumor.any():
            continue
        lo, hi = np.percentile(img[brain], [0.5, 99.5])
        areas = tumor.sum(axis=(0, 1))
        order = [int(z) for z in np.argsort(-areas) if areas[z] >= min_area]
        chosen: List[int] = []
        for z in order:
            if all(abs(z - c) >= gap for c in chosen):
                chosen.append(z)
            if len(chosen) >= per_subject:
                break
        for z in chosen:
            sl = np.clip((img[:, :, z] - lo) / max(hi - lo, 1e-6), 0, 1)
            im8 = (_to_display(sl) * 255.0 + 0.5).astype(np.uint8)
            m8 = (_to_display(tumor[:, :, z].astype(np.uint8)) * 255).astype(np.uint8)
            name = f"{d.name}_z{z:03d}"
            cv2.imwrite(str(out / "images" / f"{name}.png"), im8)
            cv2.imwrite(str(out / "masks" / f"{name}.png"), m8)
            rows.append({"split": "test", "image_path": f"images/{name}.png", "mask_path": f"masks/{name}.png",
                         "case_id": name, "patient_id": d.name, "slice_index": z, "tumor_code": "gl",
                         "tumor_label": "glioma", "plane_code": "ax", "plane_label": "axial",
                         "data_backend": "raw", "tumor_area_px": int(areas[z])})
    return rows


def prepare_folder(images: Path, masks: Path, out: Path) -> List[dict]:
    mask_index = {}
    for p in masks.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            stem = re.sub(r"(_mask|_seg|_label|_gt)$", "", p.stem, flags=re.I)
            mask_index[stem] = p
    rows = []
    for p in sorted(images.rglob("*")):
        if p.suffix.lower() not in IMG_EXT:
            continue
        m = mask_index.get(p.stem)
        if m is None:
            continue
        mk = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
        if mk is None or (mk > 0).sum() == 0:
            continue
        rows.append({"split": "test", "image_path": str(p.resolve()), "mask_path": str(m.resolve()),
                     "case_id": p.stem, "patient_id": p.stem, "slice_index": -1, "data_backend": "raw"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["brats", "folder"], required=True)
    ap.add_argument("--source", required=True, help="BraTS root, or the images folder for --format folder")
    ap.add_argument("--masks", default="", help="masks folder for --format folder")
    ap.add_argument("--label", choices=["tc", "wt"], default="tc")
    ap.add_argument("--slices-per-subject", type=int, default=1)
    ap.add_argument("--min-area", type=int, default=100, help="minimum tumor pixels for a slice to qualify")
    ap.add_argument("--slice-gap", type=int, default=5)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    out = Path(args.output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    if args.format == "brats":
        rows = prepare_brats(Path(args.source), out, args.label, args.slices_per_subject, args.min_area,
                             args.slice_gap, args.max_subjects)
    else:
        if not args.masks:
            raise SystemExit("--masks is required for --format folder")
        rows = prepare_folder(Path(args.source), Path(args.masks), out)
    if not rows:
        raise SystemExit(f"No usable image/mask pairs found in {args.source}")
    write_csv(out / "manifest.csv", rows)
    write_json({"format": args.format, "source": args.source, "label": args.label, "n_images": len(rows),
                "n_subjects": len({r["patient_id"] for r in rows}),
                "slices_per_subject": args.slices_per_subject}, out / "external_info.json")
    print(f"[external] {len(rows)} images from {len({r['patient_id'] for r in rows})} subjects -> {out}")


if __name__ == "__main__":
    main()
