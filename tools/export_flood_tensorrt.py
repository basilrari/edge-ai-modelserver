#!/usr/bin/env python3
"""
Export ResNet18 classifier + DeepLabv3+ segmenter to ONNX and TensorRT on Jetson.

  cd ~/python-worker/model_server
  python3 tools/export_flood_tensorrt.py

Outputs:
  models/flood_classifier/flood_resnet18.engine
  models/flood_segmentation/DeepLabv3_plus/flood_segmentation/flood_deeplab.engine
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.flood_models import DeepLabExportWrapper, build_deeplab_segmenter, build_resnet18_classifier
from tools.export_tensorrt import ensure_export_deps

CLF_SIZE = 224
SEG_SIZE = 256
WORKSPACE_GB = 4

CLF_WEIGHTS = ROOT / "models/flood_classifier/flood_resnet18.pth"
SEG_WEIGHTS = (
    ROOT
    / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation/best_model.pth"
)
SEG_ROBUST_WEIGHTS = (
    ROOT
    / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation/best_model_robust_floodnet.pth"
)
CLF_ONNX = ROOT / "models/flood_classifier/flood_resnet18.onnx"
SEG_ONNX = ROOT / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation/flood_deeplab.onnx"
SEG_ROBUST_ONNX = (
    ROOT / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation/flood_deeplab_robust.onnx"
)
CLF_ENGINE = ROOT / "models/flood_classifier/flood_resnet18.engine"
SEG_ENGINE = ROOT / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation/flood_deeplab.engine"
SEG_ROBUST_ENGINE = (
    ROOT / "models/flood_segmentation/DeepLabv3_plus/flood_segmentation/flood_deeplab_robust.engine"
)


def _export_onnx_engine(
    pytorch_model: torch.nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
    engine_path: Path,
    input_name: str = "images",
    output_name: str = "output0",
) -> Path:
    from ultralytics.utils.export.engine import onnx2engine, torch2onnx

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[EXPORT] ONNX → {onnx_path}")
    torch2onnx(
        pytorch_model,
        dummy,
        onnx_path,
        opset=17,
        input_names=[input_name],
        output_names=[output_name],
    )

    print(f"[EXPORT] TensorRT → {engine_path}")
    onnx2engine(
        str(onnx_path),
        engine_path,
        workspace=WORKSPACE_GB,
        half=True,
        dynamic=False,
        shape=tuple(dummy.shape),
        verbose=True,
        prefix="[TRT]",
    )
    return engine_path


def export_classifier(device: torch.device) -> Path:
    if not CLF_WEIGHTS.exists():
        raise FileNotFoundError(CLF_WEIGHTS)

    model = build_resnet18_classifier(CLF_WEIGHTS, device)
    dummy = torch.randn(1, 3, CLF_SIZE, CLF_SIZE, device=device)
    with torch.inference_mode():
        model(dummy)

    return _export_onnx_engine(
        model,
        dummy,
        CLF_ONNX,
        CLF_ENGINE,
        input_name="images",
        output_name="logits",
    )


def export_segmenter(device: torch.device, *, robust: bool = False) -> Path:
    weights = SEG_ROBUST_WEIGHTS if robust else SEG_WEIGHTS
    onnx_path = SEG_ROBUST_ONNX if robust else SEG_ONNX
    engine_path = SEG_ROBUST_ENGINE if robust else SEG_ENGINE
    label = "robust FloodNet" if robust else "lightweight"

    if not weights.exists():
        raise FileNotFoundError(weights)

    core = build_deeplab_segmenter(weights, device)
    model = DeepLabExportWrapper(core).eval()
    dummy = torch.randn(1, 3, SEG_SIZE, SEG_SIZE, device=device)
    with torch.inference_mode():
        model(dummy)

    print(f"[EXPORT] segmenter ({label}) weights={weights.name}")
    return _export_onnx_engine(
        model,
        dummy,
        onnx_path,
        engine_path,
        input_name="images",
        output_name="logits",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Export flood classifier/segmenter to TensorRT")
    parser.add_argument(
        "--segmenter",
        choices=("lightweight", "robust", "both"),
        default="both",
        help="Which segmenter engine(s) to build (default: both)",
    )
    parser.add_argument(
        "--classifier-only",
        action="store_true",
        help="Export classifier only",
    )
    args = parser.parse_args()

    ensure_export_deps()

    if not torch.cuda.is_available():
        print("[EXPORT] CUDA required for TensorRT build on Jetson")
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"[EXPORT] Device: {torch.cuda.get_device_name(0)}")

    export_classifier(device)
    if not args.classifier_only:
        if args.segmenter in ("lightweight", "both"):
            export_segmenter(device, robust=False)
        if args.segmenter in ("robust", "both"):
            export_segmenter(device, robust=True)

    print("[EXPORT] Flood engines ready:")
    print(f"  {CLF_ENGINE}")
    if args.segmenter in ("lightweight", "both") and not args.classifier_only:
        print(f"  {SEG_ENGINE}")
    if args.segmenter in ("robust", "both") and not args.classifier_only:
        print(f"  {SEG_ROBUST_ENGINE}")
    print("[EXPORT] Restart server — USE_TENSORRT=1 loads .engine files automatically")


if __name__ == "__main__":
    main()
