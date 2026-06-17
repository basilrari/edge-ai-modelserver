#!/usr/bin/env python3
"""
Export all detection models to TensorRT on Jetson Orin.

  cd ~/python-worker/model_server
  bash tools/install_export_deps.sh
  python3 tools/export_tensorrt.py              # all models
  python3 tools/export_tensorrt.py --yolo-only
  python3 tools/export_tensorrt.py --flood-only

Outputs:
  yolov8n.engine
  models/flood_classifier/flood_resnet18.engine
  models/flood_segmentation/DeepLabv3_plus/flood_segmentation/flood_deeplab.engine
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.perf_config import YOLO_IMGSZ

JETSON_EXPORT_PACKAGES = [
    "onnx>=1.12.0,<2.0.0",
    "onnxslim>=0.1.71",
    "onnxruntime>=1.16.0",
]


def _pip_install(packages: list[str]) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    print(f"[EXPORT] Installing: {' '.join(packages)}")
    subprocess.run(cmd, check=True)


def ensure_export_deps() -> None:
    try:
        import onnx  # noqa: F401
        import onnxslim  # noqa: F401
        import onnxruntime  # noqa: F401
        print("[EXPORT] ONNX dependencies already installed")
        return
    except ImportError:
        pass
    _pip_install(JETSON_EXPORT_PACKAGES)


def _patch_ultralytics_jetson_checks() -> None:
    import ultralytics.utils.checks as checks

    if getattr(checks, "_jetson_patched", False):
        return

    original = checks.check_requirements

    def patched(requirements, *args, **kwargs):
        if isinstance(requirements, list):
            fixed = []
            for req in requirements:
                if isinstance(req, str) and "onnxruntime-gpu" in req:
                    fixed.append(req.replace("onnxruntime-gpu", "onnxruntime"))
                else:
                    fixed.append(req)
            requirements = fixed
        elif isinstance(requirements, str) and "onnxruntime-gpu" in requirements:
            requirements = requirements.replace("onnxruntime-gpu", "onnxruntime")
        return original(requirements, *args, **kwargs)

    checks.check_requirements = patched
    checks._jetson_patched = True


def export_yolo() -> Path:
    _patch_ultralytics_jetson_checks()
    from ultralytics import YOLO

    pt = ROOT / "yolov8n.pt"
    if not pt.exists():
        pt = Path("yolov8n.pt")
    if not pt.exists():
        raise FileNotFoundError(f"Missing weights: {pt}")

    print(f"[EXPORT] YOLO source: {pt}  imgsz={YOLO_IMGSZ}")
    model = YOLO(str(pt))
    out = model.export(
        format="engine",
        half=True,
        imgsz=YOLO_IMGSZ,
        device=0,
        simplify=True,
        workspace=4,
        verbose=True,
    )

    engine = Path(out)
    target = ROOT / "yolov8n.engine"
    if engine.exists() and engine.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        engine.replace(target)
    print(f"[EXPORT] YOLO engine: {target}")
    return target


def export_flood() -> None:
    from tools.export_flood_tensorrt import export_classifier, export_segmenter

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for flood TensorRT export")
    device = torch.device("cuda:0")
    export_classifier(device)
    export_segmenter(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TensorRT engines for Jetson")
    parser.add_argument("--yolo-only", action="store_true")
    parser.add_argument("--flood-only", action="store_true")
    args = parser.parse_args()

    ensure_export_deps()

    try:
        if args.flood_only:
            export_flood()
        elif args.yolo_only:
            export_yolo()
        else:
            export_yolo()
            export_flood()
    except subprocess.CalledProcessError as exc:
        print(f"[EXPORT] pip failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"[EXPORT] Failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("[EXPORT] Done. Restart server with USE_TENSORRT=1 (default).")


if __name__ == "__main__":
    main()
