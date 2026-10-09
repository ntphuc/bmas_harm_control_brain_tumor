from __future__ import annotations

import warnings
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights, efficientnet_b0
from torchvision.models.segmentation import deeplabv3_resnet50


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class _SingleExitAdapter(nn.Module):
    """Expose a conventional single-output baseline through BMAS' 4-exit API.

    Training/evaluation code expects ``model(x, max_exit=N) -> list[Tensor]``.
    Static baselines do not have genuine early exits, so their one prediction is
    repeated in the returned list.  This keeps the common metric/loss pipeline
    intact without pretending the baseline has cheaper intermediate exits.
    """

    @staticmethod
    def _as_exit_list(logits: torch.Tensor, max_exit: int) -> List[torch.Tensor]:
        if max_exit not in (1, 2, 3, 4):
            raise ValueError("max_exit must be one of {1,2,3,4}")
        return [logits for _ in range(max_exit)]


class UNetBaseline(_SingleExitAdapter):
    """Compact 2-D U-Net baseline for 3-channel 256x256 inputs."""

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16
        self.enc1 = ConvBNAct(3, c1)
        self.enc2 = ConvBNAct(c1, c2)
        self.enc3 = ConvBNAct(c2, c3)
        self.enc4 = ConvBNAct(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBNAct(c4, c5)
        self.up4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBNAct(c4 + c4, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBNAct(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBNAct(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBNAct(c1 + c1, c1)
        self.head = nn.Conv2d(c1, 1, kernel_size=1)

    @staticmethod
    def _cat(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([up, skip], dim=1)

    def forward(self, x: torch.Tensor, max_exit: int = 4) -> List[torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(self._cat(self.up4(b), e4))
        d3 = self.dec3(self._cat(self.up3(d4), e3))
        d2 = self.dec2(self._cat(self.up2(d3), e2))
        d1 = self.dec1(self._cat(self.up1(d2), e1))
        logits = self.head(d1)
        return self._as_exit_list(logits, max_exit)


class DeepLabV3Baseline(_SingleExitAdapter):
    """Torchvision DeepLabV3-ResNet50 adapted to binary segmentation."""

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        backbone_weights = ResNet50_Weights.DEFAULT if pretrained_encoder else None
        try:
            self.net = deeplabv3_resnet50(
                weights=None,
                weights_backbone=backbone_weights,
                num_classes=1,
                aux_loss=False,
            )
        except Exception as exc:
            warnings.warn(
                f"Could not load pretrained ResNet50 backbone ({exc}); using random initialization."
            )
            self.net = deeplabv3_resnet50(
                weights=None,
                weights_backbone=None,
                num_classes=1,
                aux_loss=False,
            )

    def forward(self, x: torch.Tensor, max_exit: int = 4) -> List[torch.Tensor]:
        logits = self.net(x)["out"]
        return self._as_exit_list(logits, max_exit)


class EfficientNetB0Encoder(nn.Module):
    """EfficientNet-B0 feature pyramid using torchvision only."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        try:
            backbone = efficientnet_b0(weights=weights)
        except Exception as exc:
            warnings.warn(f"Could not load pretrained EfficientNet-B0 weights ({exc}); using random initialization.")
            backbone = efficientnet_b0(weights=None)
        f = backbone.features
        self.stem = nn.Sequential(f[0], f[1])      # 16 @ 1/2
        self.stage2 = f[2]                         # 24 @ 1/4
        self.stage3 = f[3]                         # 40 @ 1/8
        self.stage4 = nn.Sequential(f[4], f[5])    # 112 @ 1/16
        self.stage5 = nn.Sequential(f[6], f[7])    # 320 @ 1/32

    def forward(self, x: torch.Tensor):
        s1 = self.stem(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return s1, s2, s3, s4, s5


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBNAct(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class EfficientNetUNetBaseline(_SingleExitAdapter):
    """Single-output EfficientNet-B0 U-Net baseline using the BMAS decoder widths."""

    def __init__(self, pretrained_encoder: bool = True):
        super().__init__()
        self.encoder = EfficientNetB0Encoder(pretrained=pretrained_encoder)
        self.bottleneck = ConvBNAct(320, 192)
        self.dec4 = DecoderBlock(192, 112, 128)
        self.dec3 = DecoderBlock(128, 40, 96)
        self.dec2 = DecoderBlock(96, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 48)
        self.full_refine = ConvBNAct(48, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, max_exit: int = 4) -> List[torch.Tensor]:
        out_size = x.shape[-2:]
        s1, s2, s3, s4, s5 = self.encoder(x)
        b = self.bottleneck(s5)
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        full = F.interpolate(d1, size=out_size, mode="bilinear", align_corners=False)
        logits = self.head(self.full_refine(full))
        return self._as_exit_list(logits, max_exit)


class BMASNet(nn.Module):
    """BMAS: progressive residual segmentation with four genuine anytime exits.

    Exit 1: bottleneck-only coarse mask.
    Exit 2: after 1 decoder block.
    Exit 3: after 2 decoder blocks.
    Exit 4: full decoder.

    Each later exit predicts a residual correction on top of the previous full-resolution logits.
    `max_exit` makes actual early stopping possible for latency/compute profiling.
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

        self.exit1_head = nn.Conv2d(192, 1, kernel_size=1)
        self.exit2_residual = nn.Conv2d(128, 1, kernel_size=1)
        self.exit3_residual = nn.Conv2d(96, 1, kernel_size=1)
        self.exit4_residual = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    @staticmethod
    def _up(logits: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor, max_exit: int = 4) -> List[torch.Tensor]:
        if max_exit not in (1, 2, 3, 4):
            raise ValueError("max_exit must be one of {1,2,3,4}")
        out_size = x.shape[-2:]
        s1, s2, s3, s4, s5 = self.encoder(x)
        b = self.bottleneck(s5)

        e1 = self._up(self.exit1_head(b), out_size)
        outputs = [e1]
        if max_exit == 1:
            return outputs

        d4 = self.dec4(b, s4)
        e2 = e1 + self._up(self.exit2_residual(d4), out_size)
        outputs.append(e2)
        if max_exit == 2:
            return outputs

        d3 = self.dec3(d4, s3)
        e3 = e2 + self._up(self.exit3_residual(d3), out_size)
        outputs.append(e3)
        if max_exit == 3:
            return outputs

        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        full = F.interpolate(d1, size=out_size, mode="bilinear", align_corners=False)
        full = self.full_refine(full)
        e4 = e3 + self.exit4_residual(full)
        outputs.append(e4)
        return outputs


def build_model_from_config(cfg: dict, pretrained_encoder: bool | None = None) -> nn.Module:
    """Build the model requested by a frozen experiment config.

    ``model_type`` is used by the three static baselines. Paper variants A/B/C
    omit it and therefore use the progressive four-exit BMAS network.
    """
    model_type = str(cfg.get("model_type", "bmas")).lower()
    encoder = str(cfg.get("encoder", "efficientnet_b0")).lower()
    num_exits = int(cfg.get("num_exits", 4))
    if num_exits != 4:
        raise ValueError(f"This training/evaluation pipeline expects num_exits=4, got {num_exits}.")

    if pretrained_encoder is None:
        pretrained_encoder = bool(cfg.get("pretrained_encoder", True))

    if model_type in {"unet", "u-net"}:
        if encoder not in {"none", "unet", "plain"}:
            raise ValueError(f"UNet baseline expects encoder='none', got {encoder!r}.")
        return UNetBaseline()

    if model_type in {"deeplabv3", "deeplab", "deeplabv3_resnet50"}:
        if encoder not in {"resnet50", "resnet_50"}:
            raise ValueError(f"DeepLabV3 baseline expects encoder='resnet50', got {encoder!r}.")
        return DeepLabV3Baseline(pretrained_encoder=bool(pretrained_encoder))

    if model_type in {"efficientnet_unet", "effb0_unet", "efficientnet-b0-unet"}:
        if encoder != "efficientnet_b0":
            raise ValueError(f"EfficientNet U-Net baseline expects encoder='efficientnet_b0', got {encoder!r}.")
        return EfficientNetUNetBaseline(pretrained_encoder=bool(pretrained_encoder))

    if model_type not in {"bmas", "bmasnet", "multi_exit", "multiexit"}:
        raise ValueError(f"Unsupported model_type={model_type!r}.")
    if encoder != "efficientnet_b0":
        raise ValueError(f"BMAS expects encoder='efficientnet_b0', got {encoder!r}.")
    return BMASNet(pretrained_encoder=bool(pretrained_encoder))
