from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return config


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_training_config(cfg: dict) -> None:
    required = [
        "experiment_name", "image_size", "batch_size", "num_workers",
        "encoder", "num_exits", "epochs", "lr", "weight_decay",
        "exit_weights", "dice_weight", "bce_weight", "boundary_weight",
        "monotonic_weight", "monotonic_margin", "boundary_kernel",
        "monitor", "patience", "threshold",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    model_type = str(cfg.get("model_type", "bmas")).lower()
    encoder = str(cfg["encoder"]).lower()
    allowed = {
        "bmas", "bmasnet", "multi_exit", "multiexit",
        "unet", "u-net",
        "deeplabv3", "deeplab", "deeplabv3_resnet50",
        "efficientnet_unet", "effb0_unet", "efficientnet-b0-unet",
    }
    if model_type not in allowed:
        raise ValueError(f"Unsupported model_type={model_type!r}")
    if model_type in {"unet", "u-net"} and encoder not in {"none", "unet", "plain"}:
        raise ValueError("U-Net baseline expects encoder='none'")
    if model_type in {"deeplabv3", "deeplab", "deeplabv3_resnet50"} and encoder not in {"resnet50", "resnet_50"}:
        raise ValueError("DeepLabV3 baseline expects encoder='resnet50'")
    if model_type in {
        "efficientnet_unet", "effb0_unet", "efficientnet-b0-unet",
        "bmas", "bmasnet", "multi_exit", "multiexit",
    } and encoder != "efficientnet_b0":
        raise ValueError(f"{model_type} expects encoder='efficientnet_b0'")

    if int(cfg["num_exits"]) != 4:
        raise ValueError("paper implementation expects num_exits=4")
    if len(cfg["exit_weights"]) != 4 or float(sum(cfg["exit_weights"])) <= 0:
        raise ValueError("exit_weights must contain four positive-sum values")

    epochs = int(cfg["epochs"])
    warmup = int(cfg.get("warmup_epochs", 0))
    if epochs < 1 or warmup < 0 or warmup >= epochs:
        raise ValueError("require epochs >= 1 and 0 <= warmup_epochs < epochs")

    if str(cfg.get("scheduler", "cosine")).lower() != "cosine":
        raise ValueError("this repository implements the paper's cosine schedule")
    if not (0.0 < float(cfg.get("warmup_start_factor", 0.1)) <= 1.0):
        raise ValueError("warmup_start_factor must be in (0, 1]")
    if float(cfg.get("min_lr", 0.0)) < 0:
        raise ValueError("min_lr must be >= 0")

    if not (0.0 <= float(cfg["threshold"]) <= 1.0):
        raise ValueError("threshold must be in [0, 1]")
    if int(cfg["patience"]) < 1:
        raise ValueError("patience must be >= 1")
    if int(cfg["boundary_kernel"]) < 3 or int(cfg["boundary_kernel"]) % 2 == 0:
        raise ValueError("boundary_kernel must be odd and >= 3")
