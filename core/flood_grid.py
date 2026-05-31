"""4x4 flood grid localization (same logic as deeplab_inference.analyze_grid)."""

from __future__ import annotations

import numpy as np

try:
    from geographiclib.geodesic import Geodesic

    _GEOD = Geodesic.WGS84
except ImportError:
    _GEOD = None

# Camera intrinsics (640x480) from deeplab_inference.py
_FX, _FY = 277.19, 277.19
_CX, _CY = 160.5, 120.5
_K_INV = np.linalg.inv(
    np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]], dtype=np.float64)
)


def analyze_grid(mask: np.ndarray, grid_size: int = 4, min_ratio: float = 0.3):
    h, w = mask.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size

    max_ratio = -1.0
    best_cell = None
    cell_ratios = []

    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = mask[y1:y2, x1:x2]
            ratio = float(np.mean(cell))
            row.append(round(ratio, 4))
            if ratio > max_ratio:
                max_ratio = ratio
                best_cell = (x1, y1, x2, y2, i, j)
        cell_ratios.append(row)

    if best_cell is None or max_ratio < min_ratio:
        return {
            "centroid": None,
            "gps_text": None,
            "max_cell_ratio": round(max_ratio, 4),
            "best_cell": None,
            "cell_ratios": cell_ratios,
        }

    x1, y1, x2, y2, row, col = best_cell
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    centroid = (int(cx), int(cy))

    gps_text = _estimate_gps(centroid)

    return {
        "centroid": centroid,
        "gps_text": gps_text,
        "max_cell_ratio": round(max_ratio, 4),
        "best_cell": {"row": row, "col": col},
        "cell_ratios": cell_ratios,
    }


def _estimate_gps(centroid, drone_altitude: float = 3.0, ref_gps=(37.7749, -122.4194)):
    if _GEOD is None:
        return None

    cx, cy = centroid
    pixel = np.array([cx, cy, 1.0], dtype=np.float64)
    ray_cam = _K_INV @ pixel
    ray_cam /= np.linalg.norm(ray_cam)

    theta = np.deg2rad(45.0)
    r_pitch = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=np.float64,
    )
    ray_world = r_pitch @ ray_cam
    ray_world /= ray_world[2]
    scale = -drone_altitude / ray_world[2]
    ground_point = scale * ray_world
    dx, dy = ground_point[0], -ground_point[2]

    lat, lon = ref_gps
    distance = float(np.sqrt(dx**2 + dy**2))
    azimuth = float(np.rad2deg(np.arctan2(dx, dy)))
    new_point = _GEOD.Direct(lat, lon, azimuth, distance)
    return f"GPS: {new_point['lat2']:.6f}, {new_point['lon2']:.6f}"


def draw_grid_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 4,
    min_ratio: float = 0.3,
):
    import cv2

    h, w = frame_bgr.shape[:2]
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    analysis = analyze_grid(mask_u8, grid_size=grid_size, min_ratio=min_ratio)

    color_mask = np.zeros_like(frame_bgr)
    color_mask[:, :, 2] = mask_u8 * 255
    out = cv2.addWeighted(frame_bgr, 0.7, color_mask, 0.3, 0)

    cell_h, cell_w = h // grid_size, w // grid_size
    grid_color = (0, 255, 255)
    for i in range(1, grid_size):
        y = i * cell_h
        x = i * cell_w
        cv2.line(out, (0, y), (w, y), grid_color, 1)
        cv2.line(out, (x, 0), (x, h), grid_color, 1)

    if analysis["centroid"] is not None:
        cv2.circle(out, analysis["centroid"], 8, (0, 0, 255), -1)
        if analysis["gps_text"]:
            cv2.putText(
                out,
                analysis["gps_text"],
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

    return out, analysis
