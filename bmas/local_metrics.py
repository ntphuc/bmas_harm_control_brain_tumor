"""Local boundary transitions (paper, Sect. 3.2).

A reference boundary point is *correctly localized* at an exit when the nearest
predicted boundary point lies within ``tol`` pixels (2 px in the paper).
For a transition k -> k+1:
    BHR = #(correct at k and incorrect at k+1) / #(correct at k)
    BCR = #(incorrect at k and correct at k+1) / #(incorrect at k)
Counts are returned per image so that rates can be pooled over a test set.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

STRUCT = np.ones((3, 3), dtype=bool)


def boundary_pixels(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask)
    return mask & ~binary_erosion(mask, structure=STRUCT, border_value=0)


def localized(reference_boundary: np.ndarray, pred_mask: np.ndarray, tol: float = 2.0) -> np.ndarray:
    """Boolean per reference boundary pixel: is a predicted boundary pixel within ``tol``?"""
    pb = boundary_pixels(pred_mask)
    if not pb.any():
        return np.zeros(int(reference_boundary.sum()), dtype=bool)
    dist = distance_transform_edt(~pb)
    return dist[reference_boundary] <= tol


def transition_counts(masks: List[np.ndarray], reference: np.ndarray, tol: float = 2.0) -> Dict[str, int]:
    """Per-image counts for every transition of a list of binary exit masks."""
    ref_b = boundary_pixels(reference)
    out: Dict[str, int] = {}
    if not ref_b.any():
        for k in range(1, len(masks)):
            for key in ("good", "harm", "bad", "corr"):
                out[f"e{k}e{k+1}_{key}"] = 0
        return out
    states = [localized(ref_b, m, tol) for m in masks]
    for k in range(1, len(masks)):
        a, b = states[k - 1], states[k]
        out[f"e{k}e{k+1}_good"] = int(a.sum())
        out[f"e{k}e{k+1}_harm"] = int((a & ~b).sum())
        out[f"e{k}e{k+1}_bad"] = int((~a).sum())
        out[f"e{k}e{k+1}_corr"] = int((~a & b).sum())
    return out


def pooled_rates(rows: List[Dict[str, float]], n_exits: int = 4) -> Dict[str, float]:
    """BHR/BCR pooled over images, per transition and averaged over transitions (in %)."""
    res: Dict[str, float] = {}
    bhr, bcr = [], []
    for k in range(1, n_exits):
        good = sum(r.get(f"e{k}e{k+1}_good", 0) for r in rows)
        harm = sum(r.get(f"e{k}e{k+1}_harm", 0) for r in rows)
        bad = sum(r.get(f"e{k}e{k+1}_bad", 0) for r in rows)
        corr = sum(r.get(f"e{k}e{k+1}_corr", 0) for r in rows)
        res[f"bhr_e{k}e{k+1}"] = 100.0 * harm / good if good else float("nan")
        res[f"bcr_e{k}e{k+1}"] = 100.0 * corr / bad if bad else float("nan")
        bhr.append(res[f"bhr_e{k}e{k+1}"])
        bcr.append(res[f"bcr_e{k}e{k+1}"])
    res["bhr"] = float(np.nanmean(bhr)) if not all(np.isnan(bhr)) else float("nan")
    res["bcr"] = float(np.nanmean(bcr)) if not all(np.isnan(bcr)) else float("nan")
    return res
