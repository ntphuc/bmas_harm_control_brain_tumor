from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from bmas.model import ConvBNAct, DecoderBlock, EfficientNetB0Encoder


class _FullResHead(nn.Module):
    """Small segmentation head used by the same-backbone competitor baselines."""

    def __init__(self, in_ch: int, hidden_ch: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, 1),
        )

    def forward(self, x: torch.Tensor, out_size) -> torch.Tensor:
        return F.interpolate(self.net(x), size=out_size, mode="bilinear", align_corners=False)


class MESSStyleNet(nn.Module):
    """Same-backbone MESS-style comparator for BRISC.

    IMPORTANT: this is not the original MESS implementation.  It reproduces the
    aspects that can be compared cleanly in the BMAS codebase: independent
    executable exits and two-stage early-exit training/distillation.  The
    original MESS paper additionally searches exit number/placement/head design
    and deployment thresholds.
    """

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=pretrained_encoder)
        self.bottleneck = ConvBNAct(320, 192)
        self.dec4 = DecoderBlock(192, 112, 128)
        self.dec3 = DecoderBlock(128, 40, 96)
        self.dec2 = DecoderBlock(96, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 48)
        self.full_refine = ConvBNAct(48, 32)

        # Independent heads (not residual logit corrections).
        self.exit1_head = _FullResHead(192, 48)
        self.exit2_head = _FullResHead(128, 48)
        self.exit3_head = _FullResHead(96, 48)
        self.exit4_head = _FullResHead(32, 32)

    def trunk_parameters(self):
        modules = [
            self.encoder,
            self.bottleneck,
            self.dec4,
            self.dec3,
            self.dec2,
            self.dec1,
            self.full_refine,
            self.exit4_head,
        ]
        for module in modules:
            yield from module.parameters()

    def early_exit_parameters(self):
        for module in [self.exit1_head, self.exit2_head, self.exit3_head]:
            yield from module.parameters()

    def freeze_trunk_for_stage2(self) -> None:
        for p in self.trunk_parameters():
            p.requires_grad_(False)
        for p in self.early_exit_parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor, max_exit: int = 4) -> List[torch.Tensor]:
        if max_exit not in (1, 2, 3, 4):
            raise ValueError("max_exit must be one of {1,2,3,4}")
        out_size = x.shape[-2:]
        s1, s2, s3, s4, s5 = self.encoder(x)
        b = self.bottleneck(s5)
        e1 = self.exit1_head(b, out_size)
        outputs = [e1]
        if max_exit == 1:
            return outputs

        d4 = self.dec4(b, s4)
        e2 = self.exit2_head(d4, out_size)
        outputs.append(e2)
        if max_exit == 2:
            return outputs

        d3 = self.dec3(d4, s3)
        e3 = self.exit3_head(d3, out_size)
        outputs.append(e3)
        if max_exit == 3:
            return outputs

        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        full = F.interpolate(d1, size=out_size, mode="bilinear", align_corners=False)
        full = self.full_refine(full)
        e4 = self.exit4_head(full, out_size)
        outputs.append(e4)
        return outputs


class ADPCStyleNet(MESSStyleNet):
    """Same-backbone ADP-C-style comparator.

    Raw exits are trained as independent dense predictions.  Confidence-adaptive
    spatial refinement is applied by the evaluator, which freezes confident
    pixels and lets deeper exits update uncertain pixels.  This captures the
    *prediction mechanism* of confidence adaptivity, but standard dense PyTorch
    kernels do not realize the sparse-compute savings of the official HRNet
    ADP-C implementation.  Therefore do not report wall-clock/FLOPs from this
    class as if they were official ADP-C compute numbers.
    """

    pass
