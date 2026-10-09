from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

if TYPE_CHECKING:  # torch is imported lazily so NumPy-only tools can use this module
    import torch


def dice_iou_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch  # local import keeps the NumPy metrics usable on torch-free nodes

    pred = (torch.sigmoid(logits) >= threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dims)
    pred_sum = pred.sum(dims)
    target_sum = target.sum(dims)
    dice = (2.0 * inter + eps) / (pred_sum + target_sum + eps)
    union = pred_sum + target_sum - inter
    iou = (inter + eps) / (union + eps)
    return dice, iou


def _boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return mask
    return np.logical_xor(mask, binary_erosion(mask))


def surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return symmetric HD95 and ASSD in pixel units."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    if not pred.any() and not gt.any():
        return 0.0, 0.0

    h, w = pred.shape
    fallback = float(np.hypot(h, w))
    if not pred.any() or not gt.any():
        return fallback, fallback

    pred_boundary = _boundary(pred)
    gt_boundary = _boundary(gt)
    to_gt = distance_transform_edt(~gt_boundary)[pred_boundary]
    to_pred = distance_transform_edt(~pred_boundary)[gt_boundary]
    all_distances = np.concatenate([to_gt, to_pred])
    hd95 = float(np.percentile(all_distances, 95))
    assd = float((to_gt.mean() + to_pred.mean()) / 2.0)
    return hd95, assd


def boundary_iou(
    pred: np.ndarray,
    gt: np.ndarray,
    dilation: int = 2,
    eps: float = 1e-7,
) -> float:
    pred_band = binary_dilation(_boundary(pred), iterations=dilation)
    gt_band = binary_dilation(_boundary(gt), iterations=dilation)
    inter = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    return float((inter + eps) / (union + eps))


def numpy_case_metrics(
    probability: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    pred = np.asarray(probability) >= threshold
    gt = np.asarray(ground_truth) >= 0.5
    inter = np.logical_and(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    dice = float((2.0 * inter + 1e-7) / (pred_sum + gt_sum + 1e-7))
    union = np.logical_or(pred, gt).sum()
    iou = float((inter + 1e-7) / (union + 1e-7))
    hd95, assd = surface_distances(pred, gt)
    return {
        "dice": dice,
        "iou": iou,
        "hd95": hd95,
        "assd": assd,
        "boundary_iou": boundary_iou(pred, gt),
    }
