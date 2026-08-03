"""4x4 flood grid localization + human GPS via UavTargetLocator geolocation."""

from __future__ import annotations

import numpy as np

from core.gps_locator import DEFAULT_HEADING_DEG, GimbalOrientation, estimate_pixel_gps
from core.video_geometry import REF_ALTITUDE_M

# Simulated drone position when Pixhawk GPS is unavailable (Chittur, Kerala demo area).
FAKE_DRONE_REF_GPS = (10.7014, 76.3660)


def analyze_grid(
    mask: np.ndarray,
    grid_size: int = 4,
    min_ratio: float = 0.3,
    *,
    drone_altitude_m: float = REF_ALTITUDE_M,
    ref_gps: tuple[float, float] = FAKE_DRONE_REF_GPS,
    heading_deg: float = DEFAULT_HEADING_DEG,
    gimbal: GimbalOrientation | None = None,
):
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
        return _empty_grid_result(max_ratio, cell_ratios, drone_altitude_m=drone_altitude_m)

    x1, y1, x2, y2, row, col = best_cell
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    centroid = (int(cx), int(cy))

    geo = _estimate_gps(
        centroid,
        frame_width=w,
        frame_height=h,
        drone_altitude_m=drone_altitude_m,
        ref_gps=ref_gps,
        heading_deg=heading_deg,
        gimbal=gimbal,
    )

    return {
        "centroid": centroid,
        "gps_text": geo["gps_text"] if geo else None,
        "latitude": geo["latitude"] if geo else None,
        "longitude": geo["longitude"] if geo else None,
        "ref_latitude": geo["ref_latitude"] if geo else ref_gps[0],
        "ref_longitude": geo["ref_longitude"] if geo else ref_gps[1],
        "altitude_m": drone_altitude_m,
        "simulated": True,
        "gps_source": geo.get("gps_source", "uav_target_locator_gopro") if geo else "uav_target_locator_gopro",
        "max_cell_ratio": round(max_ratio, 4),
        "best_cell": {"row": row, "col": col},
        "cell_ratios": cell_ratios,
    }


def _empty_grid_result(
    max_ratio: float,
    cell_ratios: list,
    *,
    drone_altitude_m: float = REF_ALTITUDE_M,
) -> dict:
    return {
        "centroid": None,
        "gps_text": None,
        "latitude": None,
        "longitude": None,
        "ref_latitude": FAKE_DRONE_REF_GPS[0],
        "ref_longitude": FAKE_DRONE_REF_GPS[1],
        "altitude_m": drone_altitude_m,
        "simulated": True,
        "gps_source": "uav_target_locator_gopro",
        "max_cell_ratio": round(max_ratio, 4),
        "best_cell": None,
        "cell_ratios": cell_ratios,
    }


def estimate_gps_from_pixel(
    cx: int,
    cy: int,
    *,
    frame_width: int,
    frame_height: int,
    drone_altitude_m: float = REF_ALTITUDE_M,
    ref_gps: tuple[float, float] = FAKE_DRONE_REF_GPS,
    heading_deg: float = DEFAULT_HEADING_DEG,
    gimbal: GimbalOrientation | None = None,
) -> dict | None:
    """Project image pixel to ground GPS (UavTargetLocator / GoPro model)."""
    return _estimate_gps(
        (int(cx), int(cy)),
        frame_width=frame_width,
        frame_height=frame_height,
        drone_altitude_m=drone_altitude_m,
        ref_gps=ref_gps,
        heading_deg=heading_deg,
        gimbal=gimbal,
    )


def attach_gps_to_humans(
    humans: list[dict],
    *,
    frame_width: int,
    frame_height: int,
    drone_altitude_m: float = REF_ALTITUDE_M,
    ref_gps: tuple[float, float] = FAKE_DRONE_REF_GPS,
    heading_deg: float = DEFAULT_HEADING_DEG,
    gimbal: GimbalOrientation | None = None,
) -> list[dict]:
    """Add simulated GPS for each human bbox (bottom-center foot point)."""
    localized: list[dict] = []
    for idx, human in enumerate(humans):
        entry = dict(human)
        x1, y1, x2, y2 = entry["bbox"]
        u_px = (x1 + x2) / 2.0
        v_px = float(y2)
        entry["centroid"] = [int(round(u_px)), int(round(v_px))]
        entry["human_index"] = idx + 1
        entry["simulated"] = True
        entry["gps_source"] = "uav_target_locator_gopro"
        entry["ref_latitude"] = ref_gps[0]
        entry["ref_longitude"] = ref_gps[1]
        entry["altitude_m"] = drone_altitude_m
        geo = estimate_pixel_gps(
            u_px,
            v_px,
            frame_width=frame_width,
            frame_height=frame_height,
            drone_lat=ref_gps[0],
            drone_lon=ref_gps[1],
            drone_altitude_m=drone_altitude_m,
            heading_deg=heading_deg,
            gimbal=gimbal,
        )
        if geo:
            entry["latitude"] = geo["latitude"]
            entry["longitude"] = geo["longitude"]
            entry["gps_text"] = geo["gps_text"]
        else:
            entry["latitude"] = None
            entry["longitude"] = None
            entry["gps_text"] = None
        localized.append(entry)
    return localized


def _estimate_gps(
    centroid,
    *,
    frame_width: int,
    frame_height: int,
    drone_altitude_m: float,
    ref_gps: tuple[float, float],
    heading_deg: float = DEFAULT_HEADING_DEG,
    gimbal: GimbalOrientation | None = None,
) -> dict | None:
    cx, cy = centroid
    return estimate_pixel_gps(
        float(cx),
        float(cy),
        frame_width=frame_width,
        frame_height=frame_height,
        drone_lat=ref_gps[0],
        drone_lon=ref_gps[1],
        drone_altitude_m=drone_altitude_m,
        heading_deg=heading_deg,
        gimbal=gimbal,
    )


def serialize_grid_analysis(analysis: dict | None) -> dict | None:
    """JSON-safe grid payload for dashboard API responses."""
    if not analysis or analysis.get("centroid") is None:
        return None
    out = dict(analysis)
    centroid = out.get("centroid")
    if centroid is not None:
        out["centroid"] = [int(centroid[0]), int(centroid[1])]
    return out


def draw_grid_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    grid_size: int = 4,
    min_ratio: float = 0.3,
    annotate_gps: bool = True,
    *,
    drone_altitude_m: float = REF_ALTITUDE_M,
    ref_gps: tuple[float, float] = FAKE_DRONE_REF_GPS,
    heading_deg: float = DEFAULT_HEADING_DEG,
    gimbal: GimbalOrientation | None = None,
):
    import cv2

    h, w = frame_bgr.shape[:2]
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.shape[:2] != (h, w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    analysis = analyze_grid(
        mask_u8,
        grid_size=grid_size,
        min_ratio=min_ratio,
        drone_altitude_m=drone_altitude_m,
        ref_gps=ref_gps,
        heading_deg=heading_deg,
        gimbal=gimbal,
    )

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
        if annotate_gps and analysis["gps_text"]:
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
