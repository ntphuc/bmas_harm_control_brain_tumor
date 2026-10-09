"""Per-case processing shared by the inference stage (NumPy + SciPy only).

Given the four exit logits of one image, compute
  * post-hoc smoothed exit outputs (X9): cumulative averaging and logit damping,
  * per-exit Dice / IoU / HD95 / ASSD / Boundary-IoU with bmas.metrics,
  * the stopping signal used in X10 (uncertain-pixel ratio),
  * connected-component statistics for the E3->E4 transition (X5).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from bmas.metrics import numpy_case_metrics

FAR_PX = 10.0
STRUCT8 = np.ones((3, 3), dtype=bool)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def transform_logits(z: np.ndarray, name: str) -> np.ndarray:
    """z: [4,H,W] raw exit logits -> [4,H,W] transformed exit logits."""
    if name == "raw":
        return z
    if name == "cumavg":
        return np.cumsum(z, axis=0) / np.arange(1, z.shape[0] + 1, dtype=z.dtype)[:, None, None]
    if name.startswith("damp"):
        g = float(name[len("damp"):])
        out = np.empty_like(z)
        out[0] = z[0]
        for k in range(1, z.shape[0]):
            out[k] = out[k - 1] + g * (z[k] - z[k - 1])
        return out
    raise ValueError(f"unknown transform {name!r}")


def uncertain_ratio(prob: np.ndarray, lo: float = 0.4, hi: float = 0.6) -> float:
    """(uncertain pixels + 1) / (predicted tumour pixels + 1).

    Uncertain = probability in [lo, hi]. The +1 smoothing makes an empty
    prediction score >= 1, so a stopping rule with threshold < 1 never accepts an
    exit that predicts no tumour.
    """
    unc = np.logical_and(prob >= lo, prob <= hi).sum()
    fg = (prob >= 0.5).sum()
    return (float(unc) + 1.0) / (float(fg) + 1.0)


def component_stats(prev_mask: np.ndarray, next_mask: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Components of ``next_mask`` that do not touch ``prev_mask`` and their distance to the reference."""
    lab_prev, n_prev = label(prev_mask, structure=STRUCT8)
    lab_next, n_next = label(next_mask, structure=STRUCT8)
    gt_b = gt.astype(bool)
    if gt_b.any():
        dist_to_gt = distance_transform_edt(~gt_b)
    else:
        dist_to_gt = np.full(gt.shape, float(np.hypot(*gt.shape)))
    new_ids: List[int] = []
    if n_next:
        overlapping = set(np.unique(lab_next[prev_mask.astype(bool)]).tolist()) - {0}
        new_ids = [i for i in range(1, n_next + 1) if i not in overlapping]
    new_pixels = 0
    dists = []
    for i in new_ids:
        comp = lab_next == i
        new_pixels += int(comp.sum())
        dists.append(float(dist_to_gt[comp].min()))
    return {
        "cc_prev": int(n_prev), "cc_next": int(n_next), "new_cc": len(new_ids), "new_cc_pixels": new_pixels,
        "new_cc_max_dist": max(dists) if dists else 0.0,
        "new_cc_far": int(any(d > FAR_PX for d in dists)),
        "new_cc_dists": ";".join(f"{d:.2f}" for d in dists),
    }


def process_case(z: np.ndarray, gt: np.ndarray, transforms: Sequence[str], threshold: float = 0.5,
                 with_components: bool = True) -> Dict[str, Dict[str, float]]:
    """Return ``{transform: row}`` with canonical columns e{k}_{metric}, e{k}_uncert, e{k}_fg."""
    out: Dict[str, Dict[str, float]] = {}
    for t in transforms:
        zt = transform_logits(z, t)
        row: Dict[str, float] = {}
        masks = []
        for k in range(zt.shape[0]):
            prob = sigmoid(zt[k])
            for name, value in numpy_case_metrics(prob, gt, threshold=threshold).items():
                row[f"e{k+1}_{name}"] = float(value)
            row[f"e{k+1}_uncert"] = uncertain_ratio(prob)
            row[f"e{k+1}_fg"] = float((prob >= threshold).sum())
            masks.append(prob >= threshold)
        if with_components and t == "raw":
            for key, val in component_stats(masks[2], masks[3], gt >= 0.5).items():
                row[f"e34_{key}"] = val
        out[t] = row
    return out
