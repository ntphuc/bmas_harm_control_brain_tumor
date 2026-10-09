"""X12: export the study split to nnU-Net v2 raw format (2-D PNG).

Images are written exactly as BMAS sees them before ImageNet normalisation
(256x256, grayscale, [0,1] -> uint8), so nnU-Net and BMAS are scored on the same
grid. The development split is reproduced with a custom splits_final.json
(fold 0 = our train/val partition); the held-out split goes to imagesTs.
Needs NumPy + OpenCV + pandas (no torch).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from bmas.data import BRISCSegmentationDataset
from experiments.common import write_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--nnunet-raw", required=True)
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--dataset-name", default="BRISC")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--output-dir", required=True, help="where splits_final.json and the name map are written")
    args = ap.parse_args()

    ds_dir = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds_dir / sub).mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    names = {"train": [], "val": [], "test": []}
    name_map = []
    for split in ("train", "val", "test"):
        ds = BRISCSegmentationDataset(args.manifest, split, args.image_size, train=False, imagenet_norm=False,
                                      data_root=args.data_root)
        img_dir, lab_dir = ("imagesTs", "labelsTs") if split == "test" else ("imagesTr", "labelsTr")
        for i in range(len(ds)):
            img, mask, meta = ds.load_numpy(i)
            name = f"{args.dataset_name}_{split}_{i:05d}"
            cv2.imwrite(str(ds_dir / img_dir / f"{name}_0000.png"), np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8))
            cv2.imwrite(str(ds_dir / lab_dir / f"{name}.png"), (mask >= 0.5).astype(np.uint8))
            names[split].append(name)
            name_map.append({"nnunet_name": name, "case_id": meta["case_id"], "split": split})
        print(f"[nnunet-export] {split}: {len(ds)} images")

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(names["train"]) + len(names["val"]),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "name": args.dataset_name,
        "description": "BRISC 2-D T1c tumor masks, BMAS study split (fold 0 = train/val)",
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    (out / "splits_final.json").write_text(json.dumps([{"train": names["train"], "val": names["val"]}], indent=2),
                                           encoding="utf-8")
    write_csv(out / "nnunet_name_map.csv", name_map)
    print(f"[nnunet-export] dataset at {ds_dir}")


if __name__ == "__main__":
    main()
