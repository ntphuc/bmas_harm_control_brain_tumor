from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice loss for binary segmentation, averaged over the batch."""
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def soft_boundary_map(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Differentiable morphological boundary map used by BMAS.

    The map is computed as local dilation minus local erosion. Inputs are
    expected in [0, 1].
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer >= 3")
    pad = kernel_size // 2
    dilation = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    erosion = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilation - erosion).clamp(0.0, 1.0)


def boundary_discrepancy_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Per-case differentiable boundary discrepancy, shape [B]."""
    if logits.shape != target.shape:
        raise ValueError(f"logits/target shape mismatch: {logits.shape} vs {target.shape}")
    prob = torch.sigmoid(logits)
    pred_boundary = soft_boundary_map(prob, kernel_size)
    target_boundary = soft_boundary_map(target, kernel_size)
    return (pred_boundary - target_boundary).abs().flatten(1).mean(1)


def per_case_monotonic_penalty(
    boundary_errors: List[torch.Tensor],
    margin: float = 0.0,
) -> tuple[torch.Tensor, List[torch.Tensor]]:
    """Penalize boundary deterioration from exit k to k+1 case by case.

    The earlier-exit error is stop-gradient detached. This prevents the model
    from reducing the penalty by deliberately worsening the previous exit.
    """
    if len(boundary_errors) < 2:
        if not boundary_errors:
            raise ValueError("boundary_errors must contain at least one exit")
        zero = boundary_errors[0].new_tensor(0.0)
        return zero, []

    terms: List[torch.Tensor] = []
    for previous, later in zip(boundary_errors[:-1], boundary_errors[1:]):
        if previous.ndim != 1 or later.ndim != 1 or previous.shape != later.shape:
            raise ValueError("each boundary error tensor must have the same shape [B]")
        term = F.relu(later - previous.detach() + float(margin))
        terms.append(term)

    stacked = torch.stack(terms, dim=0)
    return stacked.mean(), terms


@dataclass
class BMASLossOutput:
    total: torch.Tensor
    parts: Dict[str, float]


class BMASLoss(nn.Module):
    """Objective used for the paper variants A, B, and C.

    A: L_seg
    B: L_seg + boundary_weight * L_bnd
    C: L_seg + boundary_weight * L_bnd + monotonic_weight * L_mono
    """

    def __init__(
        self,
        exit_weights=(0.10, 0.15, 0.25, 0.50),
        dice_weight: float = 0.60,
        bce_weight: float = 0.40,
        boundary_weight: float = 0.15,
        monotonic_weight: float = 0.20,
        monotonic_margin: float = 0.0,
        boundary_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.exit_weights = [float(x) for x in exit_weights]
        self.dice_weight = float(dice_weight)
        self.bce_weight = float(bce_weight)
        self.boundary_weight = float(boundary_weight)
        self.monotonic_weight = float(monotonic_weight)
        self.monotonic_margin = float(monotonic_margin)
        self.boundary_kernel = int(boundary_kernel)

    def forward(self, outputs: List[torch.Tensor], target: torch.Tensor) -> BMASLossOutput:
        if not outputs:
            raise ValueError("outputs must contain at least one exit")
        if len(outputs) > len(self.exit_weights):
            raise ValueError("more outputs than configured exit_weights")

        weights = target.new_tensor(self.exit_weights[: len(outputs)])
        weights = weights / weights.sum()

        seg_total = target.new_tensor(0.0)
        boundary_total = target.new_tensor(0.0)
        boundary_errors: List[torch.Tensor] = []

        for weight, logits in zip(weights, outputs):
            bce = F.binary_cross_entropy_with_logits(logits, target)
            dice = soft_dice_loss(logits, target)
            seg_total = seg_total + weight * (
                self.dice_weight * dice + self.bce_weight * bce
            )

            per_case = boundary_discrepancy_per_sample(
                logits, target, kernel_size=self.boundary_kernel
            )
            boundary_errors.append(per_case)
            boundary_total = boundary_total + weight * per_case.mean()

        monotonic_total, transition_terms = per_case_monotonic_penalty(
            boundary_errors, margin=self.monotonic_margin
        )

        total = (
            seg_total
            + self.boundary_weight * boundary_total
            + self.monotonic_weight * monotonic_total
        )

        parts: Dict[str, float] = {
            "seg": float(seg_total.detach().cpu()),
            "boundary": float(boundary_total.detach().cpu()),
            "monotonic": float(monotonic_total.detach().cpu()),
            "total": float(total.detach().cpu()),
        }
        for idx, term in enumerate(transition_terms, start=1):
            parts[f"monotonic_e{idx}_to_e{idx+1}"] = float(term.mean().detach().cpu())
            parts[f"monotonic_active_e{idx}_to_e{idx+1}"] = float(
                (term.detach() > 0).float().mean().cpu()
            )

        return BMASLossOutput(total=total, parts=parts)
