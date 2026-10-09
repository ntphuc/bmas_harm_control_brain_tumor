from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:  # torch is optional so that NumPy-only revision tools can reuse this loader.
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised only on torch-free nodes
    torch = None

    class Dataset:  # minimal stand-in with the same indexing protocol
        pass


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference training augmentation used by this cleaned implementation.

    See REPRODUCIBILITY.md: the paper's archived artifacts did not retain enough
    provenance to claim bitwise identity with the historical training pipeline.
    """
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])

    if random.random() < 0.35:
        angle = random.uniform(-10.0, 10.0)
        h, w = image.shape
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    if random.random() < 0.35:
        alpha = random.uniform(0.90, 1.10)
        beta = random.uniform(-0.05, 0.05)
        image = np.clip(image * alpha + beta, 0.0, 1.0)

    if random.random() < 0.20:
        gamma = random.uniform(0.85, 1.15)
        image = np.power(np.clip(image, 0.0, 1.0), gamma)

    return image, mask


def _image_to_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unsupported image sample shape: {array.shape}")
    if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
        return array.astype(np.float32).mean(axis=0)
    if array.shape[-1] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
        return array.astype(np.float32).mean(axis=-1)
    raise ValueError(f"Cannot infer image channel axis from shape: {array.shape}")


def _mask_to_2d(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Unsupported mask sample shape: {array.shape}")
    if array.shape[0] <= 4 and array.shape[1] > 4 and array.shape[2] > 4:
        return array.max(axis=0)
    if array.shape[-1] <= 4 and array.shape[0] > 4 and array.shape[1] > 4:
        return array.max(axis=-1)
    raise ValueError(f"Cannot infer mask channel axis from shape: {array.shape}")


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    if float(image.max()) > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    if not np.isfinite(mask).all():
        raise ValueError("mask contains non-finite values")
    if float(mask.max()) > 1.5:
        mask = mask / 255.0
    return (mask > 0.5).astype(np.float32)


class BRISCSegmentationDataset(Dataset):
    """BRISC segmentation dataset backed by a CSV manifest.

    Supported manifest backends:
      * raw: image_path + mask_path
      * npy: x_path + y_path + sample_index

    Returned tensors are [3,H,W] ImageNet-normalized images and [1,H,W]
    binary masks, matching the paper implementation.
    """

    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        image_size: int = 256,
        train: bool = False,
        imagenet_norm: bool = True,
        data_root: str | Path = "datasets/brisc2025",
    ) -> None:
        self.manifest_path = Path(manifest_csv).resolve()
        self.data_root = Path(data_root).resolve()
        frame = pd.read_csv(self.manifest_path)
        if "split" not in frame.columns:
            raise ValueError("manifest missing required column: split")
        self.df = frame[frame["split"].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows found for split={split!r}")

        self.image_size = int(image_size)
        self.train = bool(train)
        self.imagenet_norm = bool(imagenet_norm)

        if "data_backend" in self.df.columns:
            values = self.df["data_backend"].dropna().astype(str).str.lower().unique().tolist()
            if len(values) != 1:
                raise ValueError(f"A split cannot mix data backends: {values}")
            self.backend = values[0]
        elif {"image_path", "mask_path"}.issubset(self.df.columns):
            self.backend = "raw"
        elif {"x_path", "y_path", "sample_index"}.issubset(self.df.columns):
            self.backend = "npy"
        else:
            raise ValueError("manifest must describe raw or npy BRISC samples")

        if self.backend not in {"raw", "npy"}:
            raise ValueError(f"Unsupported backend for the paper repository: {self.backend!r}")

        self.X = None
        self.Y = None
        self._x_path: Path | None = None
        self._y_path: Path | None = None
        if self.backend == "npy":
            x_paths = self.df["x_path"].astype(str).unique().tolist()
            y_paths = self.df["y_path"].astype(str).unique().tolist()
            if len(x_paths) != 1 or len(y_paths) != 1:
                raise ValueError("Each split must reference one X array and one Y array")
            self._x_path = self._resolve(x_paths[0])
            self._y_path = self._resolve(y_paths[0])
            self.X = np.load(self._x_path, mmap_mode="r")
            self.Y = np.load(self._y_path, mmap_mode="r")
            if len(self.X) != len(self.Y):
                raise ValueError("X/Y NPY length mismatch")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, value: str) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else self.data_root / path

    def _read_raw(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, str, str]:
        image_path = self._resolve(row.image_path)
        mask_path = self._resolve(row.mask_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(mask_path)
        return (
            _normalize_image(image),
            _normalize_mask(mask),
            str(image_path),
            str(mask_path),
        )

    def _read_npy(self, row: pd.Series) -> tuple[np.ndarray, np.ndarray, str, str]:
        idx = int(row.sample_index)
        if idx < 0 or idx >= len(self.X):
            raise IndexError(f"sample_index={idx} outside NPY array")
        image = _normalize_image(_image_to_2d(self.X[idx]))
        mask = _normalize_mask(_mask_to_2d(self.Y[idx]))
        return image, mask, f"{self._x_path}#{idx}", f"{self._y_path}#{idx}"

    def load_numpy(self, idx: int):
        """Return the preprocessed sample as NumPy arrays (no ImageNet normalisation).

        image: float32 [H,W] in [0,1] after the same resize used by ``__getitem__``;
        mask: float32 [H,W] in {0,1}; meta: same dictionary as ``__getitem__``.
        Augmentation is applied only when ``train=True``, exactly as before.
        """
        row = self.df.iloc[idx]
        if self.backend == "raw":
            image, mask, image_locator, mask_locator = self._read_raw(row)
        else:
            image, mask, image_locator, mask_locator = self._read_npy(row)

        if self.train:
            image, mask = _augment(image, mask)

        image = cv2.resize(
            image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        mask = (mask > 0.5).astype(np.float32)

        default_id = f"{self.backend}_{idx:05d}"
        meta = {
            "case_id": str(row.get("case_id", default_id)),
            "patient_id": str(row.get("patient_id", row.get("case_id", default_id))),
            "slice_index": int(row.get("slice_index", -1)),
            "image_path": image_locator,
            "mask_path": mask_locator,
            "data_backend": self.backend,
            "tumor_code": str(row.get("tumor_code", "unknown")),
            "tumor_label": str(row.get("tumor_label", "unknown")),
            "plane_code": str(row.get("plane_code", "unknown")),
            "plane_label": str(row.get("plane_label", "unknown")),
            "lesion_pixels": float(mask.sum()),
        }
        return image.astype(np.float32), mask, meta

    def __getitem__(self, idx: int):
        if torch is None:
            raise ModuleNotFoundError("torch is required for __getitem__; use load_numpy() instead")
        image, mask, meta = self.load_numpy(idx)
        image3 = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
        if self.imagenet_norm:
            image3 = (image3 - IMAGENET_MEAN) / IMAGENET_STD

        image_t = torch.from_numpy(np.ascontiguousarray(image3.transpose(2, 0, 1))).float()
        mask_t = torch.from_numpy(np.ascontiguousarray(mask[None])).float()
        return image_t, mask_t, meta
