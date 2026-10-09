from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


def image_confidence(prob: torch.Tensor) -> torch.Tensor:
    """Mean binary confidence per image, shape [B]."""
    conf = torch.maximum(prob, 1.0 - prob)
    return conf.flatten(1).mean(1)


def mess_image_exit(outputs: List[torch.Tensor], thresholds: Sequence[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Whole-image confidence exit policy.

    thresholds contains three values for E1/E2/E3; samples not exiting early use
    E4. Returns selected logits and 1-based exit ids.
    """
    if len(outputs) != 4 or len(thresholds) != 3:
        raise ValueError("need four outputs and three thresholds")
    batch = outputs[0].shape[0]
    chosen = outputs[-1].clone()
    exit_ids = torch.full((batch,), 4, dtype=torch.long, device=outputs[0].device)
    undecided = torch.ones(batch, dtype=torch.bool, device=outputs[0].device)
    for idx, (logits, threshold) in enumerate(zip(outputs[:3], thresholds), start=1):
        conf = image_confidence(torch.sigmoid(logits))
        take = undecided & (conf >= float(threshold))
        if take.any():
            chosen[take] = logits[take]
            exit_ids[take] = idx
            undecided[take] = False
    return chosen, exit_ids


def adpc_spatial_fusion(outputs: List[torch.Tensor], threshold: float) -> Tuple[torch.Tensor, np.ndarray]:
    """Confidence-adaptive spatial refinement at prediction level.

    Confident pixels are frozen at an early exit; only uncertain pixels are
    updated by the next prediction.  Returns fused logits and the mean active
    pixel fraction before each deeper stage (E2/E3/E4).

    NOTE: this is a prediction-level same-backbone proxy. Dense PyTorch kernels
    still execute the complete deeper stage, so active fractions are *not* true
    wall-clock or GFLOP savings.
    """
    if len(outputs) != 4:
        raise ValueError("need four outputs")
    fused_prob = torch.sigmoid(outputs[0])
    active_fracs = []
    for later in outputs[1:]:
        conf = torch.maximum(fused_prob, 1.0 - fused_prob)
        active = conf < float(threshold)
        active_fracs.append(float(active.float().mean().detach().cpu()))
        later_prob = torch.sigmoid(later)
        fused_prob = torch.where(active, later_prob, fused_prob)
    eps = torch.finfo(fused_prob.dtype).eps
    fused_prob = fused_prob.clamp(eps, 1.0 - eps)
    logits = torch.logit(fused_prob)
    return logits, np.asarray(active_fracs, dtype=float)
