#!/usr/bin/env python3
"""
Export VisDrone-trained robust human detector to TensorRT on Jetson.

  cd ~/python-worker/model_server
  python3 tools/export_robust_human.py

Output:
  models/human_detector/yolo11s_visdrone_human_1280.engine
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROBUST_PT = ROOT / "models/human_detector/yolo11s_visdrone_human_1280.pt"
ROBUST_ENGINE = ROOT / "models/human_detector/yolo11s_visdrone_human_1280.engine"
ROBUST_IMGSZ = 1280


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


def export_robust_human() -> Path:
    _patch_ultralytics_jetson_checks()
    from ultralytics import YOLO

    if not ROBUST_PT.exists():
        raise FileNotFoundError(f"Missing weights: {ROBUST_PT}")

    print(f"[EXPORT] robust human source: {ROBUST_PT} imgsz={ROBUST_IMGSZ}")
    model = YOLO(str(ROBUST_PT))
    out = model.export(
        format="engine",
        imgsz=ROBUST_IMGSZ,
        half=True,
        device=0,
        simplify=True,
        workspace=4,
        verbose=True,
    )

    engine = Path(out)
    if engine.exists() and engine.resolve() != ROBUST_ENGINE.resolve():
        if ROBUST_ENGINE.exists():
            ROBUST_ENGINE.unlink()
        engine.replace(ROBUST_ENGINE)

    print(f"[EXPORT] robust human engine: {ROBUST_ENGINE}")
    return ROBUST_ENGINE


def main() -> None:
    try:
        export_robust_human()
    except Exception as exc:
        print(f"[EXPORT] Failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    print("[EXPORT] Done. Use HUMAN_DETECTOR_TIER=robust to load this model.")


if __name__ == "__main__":
    main()
