"""Runtime performance toggles (env overrides)."""

import os

PARALLEL_COMBINED = os.environ.get("PARALLEL_COMBINED", "1").strip() not in (
    "0",
    "false",
    "no",
)

USE_TENSORRT = os.environ.get("USE_TENSORRT", "1").strip() not in ("0", "false", "no")

USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0").strip() in (
    "1",
    "true",
    "yes",
)

# Smaller = faster human detection (320 recommended on Jetson).
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "320"))

ASYNC_POWER = os.environ.get("ASYNC_POWER", "1").strip() not in ("0", "false", "no")

CUDNN_BENCHMARK = os.environ.get("CUDNN_BENCHMARK", "1").strip() not in (
    "0",
    "false",
    "no",
)

# Skip expensive DeepLab unless classifier suspects flood or periodic refresh.
SMART_SEGMENT = os.environ.get("SMART_SEGMENT", "1").strip() not in ("0", "false", "no")

# Run segmentation every N frames even when classifier says dry (0 = only when needed).
SEG_INTERVAL = int(os.environ.get("SEG_INTERVAL", "8"))

# Ratio hysteresis — keep segmenting briefly after flood drops.
SEG_HYSTERESIS = float(os.environ.get("SEG_HYSTERESIS", "0.12"))

# Always run DeepLab every frame (slow — for debugging only).
ALWAYS_SEGMENT = os.environ.get("ALWAYS_SEGMENT", "0").strip() in ("1", "true", "yes")

# WebSocket minimum gap between completed inferences (seconds).
WS_MIN_INTERVAL = float(os.environ.get("WS_MIN_INTERVAL", "0.05"))
