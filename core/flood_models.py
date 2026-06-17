"""Build flood classifier (ResNet18) and segmenter (DeepLabv3+ MobileNetV3) for PyTorch or export."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def _clean_state_dict(state_dict: dict) -> dict:
    return {k.replace("model.", ""): v for k, v in state_dict.items()}


def build_resnet18_classifier(
    weights_path: Path,
    device: torch.device,
) -> nn.Module:
    from models.flood_classifier.realtime_flood_detection import Net

    model = Net()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(_clean_state_dict(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def build_deeplab_segmenter(
    weights_path: Path,
    device: torch.device,
) -> nn.Module:
    model = deeplabv3_mobilenet_v3_large(weights=None)
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


class DeepLabExportWrapper(nn.Module):
    """ONNX/TensorRT export: logits only (no dict)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]
