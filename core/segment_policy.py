"""When to run expensive DeepLab segmentation."""

from __future__ import annotations

from core.perf_config import (
    ALWAYS_SEGMENT,
    SEG_HYSTERESIS,
    SEG_INTERVAL,
    SMART_SEGMENT,
)

# ImageFolder: 0 = Flood, 1 = Non_Flood
CLF_FLOOD_CLASS = 0


def should_run_segmentation(
    frame_index: int,
    clf_class_index: int,
    last_flood_ratio: float,
    context_allows: bool,
) -> bool:
    if not context_allows:
        return False
    if ALWAYS_SEGMENT:
        return True
    if not SMART_SEGMENT:
        return True

    if clf_class_index == CLF_FLOOD_CLASS:
        return True
    if last_flood_ratio >= SEG_HYSTERESIS:
        return True
    if SEG_INTERVAL > 0 and frame_index % SEG_INTERVAL == 0:
        return True
    return False
