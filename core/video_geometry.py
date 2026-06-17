"""Simulate drone altitude / ground coverage via resize (pre-flight trade-off study)."""

from __future__ import annotations

import cv2
import numpy as np

# Reference: 50 m AGL ≈ native 640 px width for configured camera FOV.
REF_ALTITUDE_M = 50.0
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def simulate_altitude_frame(
    frame_bgr: np.ndarray,
    altitude_m: float,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
) -> tuple[np.ndarray, dict]:
    """
    Higher altitude → smaller ground features → downscale before letterbox.

    Returns processed frame and metadata for reports.
    """
    h, w = frame_bgr.shape[:2]
    scale = REF_ALTITUDE_M / max(float(altitude_m), 5.0)
    sim_w = max(32, int(w * scale))
    sim_h = max(24, int(h * scale))
    scaled = cv2.resize(frame_bgr, (sim_w, sim_h), interpolation=cv2.INTER_AREA)

    if sim_w <= target_w and sim_h <= target_h:
        placement = "letterbox"
        out = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y0 = (target_h - sim_h) // 2
        x0 = (target_w - sim_w) // 2
        out[y0 : y0 + sim_h, x0 : x0 + sim_w] = scaled
    else:
        # Low altitude → larger GSD → upscaled view; crop to sensor output size.
        placement = "center_crop"
        crop_y0 = (sim_h - target_h) // 2
        crop_x0 = (sim_w - target_w) // 2
        out = scaled[
            crop_y0 : crop_y0 + target_h,
            crop_x0 : crop_x0 + target_w,
        ].copy()

    ground_coverage_factor = altitude_m / REF_ALTITUDE_M
    gsd_factor = altitude_m / REF_ALTITUDE_M

    meta = {
        "sim_altitude_m": altitude_m,
        "scale_factor": round(scale, 4),
        "sim_width": sim_w,
        "sim_height": sim_h,
        "placement": placement,
        "output_width": target_w,
        "output_height": target_h,
        "ground_coverage_factor": round(ground_coverage_factor, 3),
        "gsd_factor": round(gsd_factor, 3),
        "coverage_note": (
            f"At {altitude_m:.0f}m, one frame covers ~{ground_coverage_factor:.1f}× "
            f"the ground area vs {REF_ALTITUDE_M:.0f}m reference; "
            f"pixel size ~{gsd_factor:.1f}× coarser (less detail per pixel)."
        ),
    }
    return out, meta
