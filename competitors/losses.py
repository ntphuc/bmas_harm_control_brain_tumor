from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from bmas.losses import soft_dice_loss


@dataclass
class CompetitorLossOutput:
    total: torch.Tensor
    seg: torch.Tensor
    distill: torch.Tensor


def multiexit_seg_loss(
    outputs: List[torch.Tensor],
    target: torch.Tensor,
    exit_weights=(0.10, 0.15, 0.25, 0.50),
    dice_weight: float = 0.60,
    bce_weight: float = 0.40,
) -> torch.Tensor:
    weights = target.new_tensor(exit_weights[: len(outputs)])
    weights = weights / weights.sum()
    total = target.new_tensor(0.0)
    for w, logits in zip(weights, outputs):
        dice = soft_dice_loss(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        total = total + w * (dice_weight * dice + bce_weight * bce)
    return total


def mess_stage2_loss(
    outputs: List[torch.Tensor],
    target: torch.Tensor,
    distill_weight: float = 0.50,
    teacher_confidence: float = 0.90,
) -> CompetitorLossOutput:
    """Early-exit supervised + positive-filtered teacher distillation.

    The deepest exit is detached and used as teacher.  Only high-confidence
    teacher pixels participate in the soft-target BCE term.
    """
    if len(outputs) != 4:
        raise ValueError("MESS stage-2 expects four exits")
    teacher_prob = torch.sigmoid(outputs[-1]).detach()
    teacher_conf = torch.maximum(teacher_prob, 1.0 - teacher_prob)
    keep = (teacher_conf >= teacher_confidence).float()

    seg = target.new_tensor(0.0)
    distill = target.new_tensor(0.0)
    for logits in outputs[:-1]:
        seg = seg + 0.6 * soft_dice_loss(logits, target) + 0.4 * F.binary_cross_entropy_with_logits(logits, target)
        pixel_bce = F.binary_cross_entropy_with_logits(logits, teacher_prob, reduction="none")
        denom = keep.sum().clamp_min(1.0)
        distill = distill + (pixel_bce * keep).sum() / denom
    seg = seg / 3.0
    distill = distill / 3.0
    total = seg + float(distill_weight) * distill
    return CompetitorLossOutput(total=total, seg=seg, distill=distill)
