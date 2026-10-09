"""Local transition objectives of Variants D and E (paper, Sect. 4.2).

For each transition k -> k+1 the terms are computed inside a local boundary band
around the reference contour, against the *detached* state of exit k:

* preservation  L_pres : new error in band pixels that exit k already got right,
                          weighted by goodness^p  (good regions must not get worse);
* correction    L_corr : remaining error at exit k+1 in pixels that exit k got wrong,
                          weighted by badness^p   (bad regions should improve);
* tail          L_tail : mean of the largest `tail_fraction` new errors per image.

Variant D:  L_seg + 0.15 L_bnd + 0.20 * mean_k (L_pres + 0.50 L_corr + 0.25 L_tail)
Variant E:  L_seg + 0.15 L_bnd + lambda(t) * sum_k beta_k (L_pres + 0.10 L_corr + 0.10 L_tail)
            beta = [0.2, 0.5, 1.0]; lambda(t) = 0 for `warmup_epochs`, then a linear
            ramp over `ramp_epochs` to `lambda_max` (0.10).

NOTE: this module re-implements the objective from the description in the paper
(7x7 band, preserve margin 0, good/bad weighting power 2, top-20% tail). It was
written for the public release; if it differs from the authors' original training
code, the original should take precedence for exact reproduction of the paper's
Variant D/E checkpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bmas.losses import BMASLoss, BMASLossOutput, soft_boundary_map


def boundary_band(target: torch.Tensor, band: int = 7, kernel_size: int = 3) -> torch.Tensor:
    """Binary band of width ``band`` around the reference contour, shape [B,1,H,W]."""
    edge = soft_boundary_map(target, kernel_size)
    pad = band // 2
    return (F.max_pool2d(edge, kernel_size=band, stride=1, padding=pad) > 0).float()


@dataclass
class TransitionTerms:
    pres: torch.Tensor
    corr: torch.Tensor
    tail: torch.Tensor


def transition_terms(prev_logits: torch.Tensor, next_logits: torch.Tensor, target: torch.Tensor,
                     band_mask: torch.Tensor, power: float = 2.0, margin: float = 0.0,
                     tail_fraction: float = 0.20) -> TransitionTerms:
    """Preservation, correction, and tail terms for one transition (scalars)."""
    p_prev = torch.sigmoid(prev_logits).detach()
    p_next = torch.sigmoid(next_logits)
    err_prev = (p_prev - target).abs()
    err_next = (p_next - target).abs()
    good = (1.0 - err_prev).clamp(0, 1) ** power
    bad = err_prev.clamp(0, 1) ** power
    new_err = F.relu(err_next - err_prev - margin)

    denom = band_mask.flatten(1).sum(1).clamp_min(1.0)
    pres = ((good * new_err * band_mask).flatten(1).sum(1) / denom).mean()
    corr = ((bad * err_next * band_mask).flatten(1).sum(1) / denom).mean()

    masked = (new_err * band_mask).flatten(1)
    tails = []
    for i in range(masked.shape[0]):
        n_band = int(band_mask[i].sum().item())
        k = max(1, int(round(tail_fraction * n_band)))
        tails.append(torch.topk(masked[i], k=min(k, masked.shape[1])).values.mean())
    tail = torch.stack(tails).mean()
    return TransitionTerms(pres=pres, corr=corr, tail=tail)


class TransitionLoss(nn.Module):
    """Variant D or E objective: L_seg + 0.15 L_bnd + transition term."""

    def __init__(self, variant: str, boundary_weight: float = 0.15, band: int = 7, power: float = 2.0,
                 margin: float = 0.0, tail_fraction: float = 0.20, d_weight: float = 0.20,
                 d_coeffs: Sequence[float] = (1.0, 0.50, 0.25), e_coeffs: Sequence[float] = (1.0, 0.10, 0.10),
                 e_beta: Sequence[float] = (0.2, 0.5, 1.0), lambda_max: float = 0.10,
                 warmup_epochs: int = 5, ramp_epochs: int = 5) -> None:
        super().__init__()
        variant = variant.upper()
        if variant not in {"D", "E"}:
            raise ValueError("TransitionLoss is defined for variants D and E")
        self.variant = variant
        self.base = BMASLoss(boundary_weight=boundary_weight, monotonic_weight=0.0)
        self.band, self.power, self.margin, self.tail_fraction = band, power, margin, tail_fraction
        self.d_weight, self.d_coeffs = d_weight, tuple(d_coeffs)
        self.e_coeffs, self.e_beta = tuple(e_coeffs), tuple(e_beta)
        self.lambda_max, self.warmup_epochs, self.ramp_epochs = lambda_max, warmup_epochs, ramp_epochs
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Epochs are counted from 0."""
        self.epoch = int(epoch)

    def schedule(self) -> float:
        if self.variant == "D":
            return self.d_weight
        if self.epoch < self.warmup_epochs:
            return 0.0
        ramp = (self.epoch - self.warmup_epochs + 1) / max(self.ramp_epochs, 1)
        return self.lambda_max * min(1.0, ramp)

    def forward(self, outputs: List[torch.Tensor], target: torch.Tensor) -> BMASLossOutput:
        base = self.base(outputs, target)
        weight = self.schedule()
        band = boundary_band(target, self.band)
        parts: Dict[str, float] = dict(base.parts)
        trans_total = target.new_tensor(0.0)
        coeffs = self.d_coeffs if self.variant == "D" else self.e_coeffs
        for k in range(len(outputs) - 1):
            t = transition_terms(outputs[k], outputs[k + 1], target, band, self.power, self.margin,
                                 self.tail_fraction)
            term = coeffs[0] * t.pres + coeffs[1] * t.corr + coeffs[2] * t.tail
            if self.variant == "D":
                term = term / (len(outputs) - 1)
            else:
                term = self.e_beta[k] * term
            trans_total = trans_total + term
            parts[f"pres_e{k+1}e{k+2}"] = float(t.pres.detach())
            parts[f"corr_e{k+1}e{k+2}"] = float(t.corr.detach())
            parts[f"tail_e{k+1}e{k+2}"] = float(t.tail.detach())
        total = base.total + weight * trans_total
        parts.update({"transition": float(trans_total.detach()), "transition_weight": weight,
                      "total": float(total.detach())})
        return BMASLossOutput(total=total, parts=parts)
