"""GoPro-aware pixel → GPS geolocation (UavTargetLocator algorithm).

Ported from UavTargetLocator-main/src/uav_target_locator/geolocate/locator.py
for nadir gimbal + pinhole ray cast to flat ground + local ENU → WGS84.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, isfinite, radians, sin, sqrt, tan
from typing import Any

EARTH_RADIUS_M = 6_378_137.0

# GoPro Linear 16:9 nominal (planning/sensor.py in UavTargetLocator-main).
DEFAULT_GOPRO_HFOV_DEG = float(os.environ.get("GOPRO_HFOV_DEG", "87.0"))
DEFAULT_CAPTURE_WIDTH = int(os.environ.get("GOPRO_CAPTURE_WIDTH", "1920"))
DEFAULT_CAPTURE_HEIGHT = int(os.environ.get("GOPRO_CAPTURE_HEIGHT", "1080"))
DEFAULT_HEADING_DEG = float(os.environ.get("DRONE_HEADING_DEG", "0.0"))
DEFAULT_GIMBAL_PITCH_DEG = float(os.environ.get("GIMBAL_PITCH_DEG", "-90.0"))
DEFAULT_GIMBAL_YAW_DEG = float(os.environ.get("GIMBAL_YAW_DEG", "0.0"))
INTRINSICS_JSON = os.environ.get("GOPRO_INTRINSICS_JSON", "").strip()


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_coeffs: tuple[float, ...] = ()
    distortion_model: str = "none"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> CameraIntrinsics:
        if "camera_matrix" in data:
            matrix = data["camera_matrix"]
            fx = float(matrix[0][0])
            fy = float(matrix[1][1])
            cx = float(matrix[0][2])
            cy = float(matrix[1][2])
        else:
            fx = float(data["fx"])
            fy = float(data["fy"])
            cx = float(data["cx"])
            cy = float(data["cy"])
        coeffs = data.get("distortion_coeffs", data.get("dist_coeffs", ()))
        return cls(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_coeffs=tuple(float(v) for v in coeffs),
            distortion_model=str(data.get("distortion_model", "opencv")),
        )


@dataclass(frozen=True)
class GimbalOrientation:
    pitch_deg: float = DEFAULT_GIMBAL_PITCH_DEG
    yaw_deg: float = DEFAULT_GIMBAL_YAW_DEG
    roll_deg: float = 0.0


@dataclass(frozen=True)
class LocatorConfig:
    camera_to_body_yaw_offset_deg: float = 0.0
    min_ground_ray_down_component: float = 1e-6


@dataclass(frozen=True)
class LocationEstimate:
    valid: bool
    target_lat_deg: float | None
    target_lon_deg: float | None
    quality_flags: tuple[str, ...] = field(default_factory=tuple)


def intrinsics_from_fov(width: int, height: int, hfov_deg: float) -> CameraIntrinsics:
    fx = (width / 2.0) / tan(radians(hfov_deg) / 2.0)
    return CameraIntrinsics(
        fx=fx,
        fy=fx,
        cx=width / 2.0,
        cy=height / 2.0,
        distortion_coeffs=(),
        distortion_model="none",
    )


def load_intrinsics_json(path: str) -> CameraIntrinsics:
    with open(path, encoding="utf-8") as fh:
        return CameraIntrinsics.from_mapping(json.load(fh))


def scale_intrinsics(
    intrinsics: CameraIntrinsics,
    *,
    from_width: int,
    from_height: int,
    to_width: int,
    to_height: int,
) -> CameraIntrinsics:
    if from_width <= 0 or from_height <= 0:
        return intrinsics
    sx = to_width / from_width
    sy = to_height / from_height
    return CameraIntrinsics(
        fx=intrinsics.fx * sx,
        fy=intrinsics.fy * sy,
        cx=intrinsics.cx * sx,
        cy=intrinsics.cy * sy,
        distortion_coeffs=intrinsics.distortion_coeffs,
        distortion_model=intrinsics.distortion_model,
    )


_cached_intrinsics: CameraIntrinsics | None = None


def get_camera_intrinsics(frame_width: int, frame_height: int) -> CameraIntrinsics:
    global _cached_intrinsics
    if _cached_intrinsics is not None:
        return scale_intrinsics(
            _cached_intrinsics,
            from_width=DEFAULT_CAPTURE_WIDTH,
            from_height=DEFAULT_CAPTURE_HEIGHT,
            to_width=frame_width,
            to_height=frame_height,
        )

    if INTRINSICS_JSON and os.path.isfile(INTRINSICS_JSON):
        loaded = load_intrinsics_json(INTRINSICS_JSON)
        cal_w = int(os.environ.get("GOPRO_CALIB_WIDTH", str(DEFAULT_CAPTURE_WIDTH)))
        cal_h = int(os.environ.get("GOPRO_CALIB_HEIGHT", str(DEFAULT_CAPTURE_HEIGHT)))
        _cached_intrinsics = loaded
        return scale_intrinsics(
            loaded,
            from_width=cal_w,
            from_height=cal_h,
            to_width=frame_width,
            to_height=frame_height,
        )

    return intrinsics_from_fov(frame_width, frame_height, DEFAULT_GOPRO_HFOV_DEG)


def estimate_pixel_gps(
    u_px: float,
    v_px: float,
    *,
    frame_width: int,
    frame_height: int,
    drone_lat: float,
    drone_lon: float,
    drone_altitude_m: float,
    heading_deg: float = DEFAULT_HEADING_DEG,
    gimbal: GimbalOrientation | None = None,
    intrinsics: CameraIntrinsics | None = None,
    config: LocatorConfig | None = None,
) -> dict | None:
    """Project one image pixel to ground GPS using the UavTargetLocator model."""
    if frame_width <= 0 or frame_height <= 0:
        return None
    if not all(isfinite(v) for v in (u_px, v_px, drone_lat, drone_lon, drone_altitude_m)):
        return None
    if drone_altitude_m <= 0:
        return None

    cfg = config or LocatorConfig()
    gimbal_orientation = gimbal or GimbalOrientation()
    intr = intrinsics or get_camera_intrinsics(frame_width, frame_height)

    x_norm, y_norm = undistort_pixel_to_normalized_point(u_px, v_px, intr)
    ray_body = camera_ray_to_body(
        x_norm=x_norm,
        y_norm=y_norm,
        gimbal=gimbal_orientation,
        camera_to_body_yaw_offset_deg=cfg.camera_to_body_yaw_offset_deg,
    )
    ray_enu = body_vector_to_enu(
        x_right=ray_body[0],
        y_forward=ray_body[1],
        z_up=ray_body[2],
        heading_deg=heading_deg,
    )

    ray_up = ray_enu[2]
    if abs(ray_up) < cfg.min_ground_ray_down_component or ray_up >= 0.0:
        return None

    scale = -drone_altitude_m / ray_up
    east_m = scale * ray_enu[0]
    north_m = scale * ray_enu[1]

    target_lat, target_lon = add_enu_to_gps(
        lat_deg=drone_lat,
        lon_deg=drone_lon,
        east_m=east_m,
        north_m=north_m,
    )
    lat2 = round(float(target_lat), 6)
    lon2 = round(float(target_lon), 6)
    return {
        "latitude": lat2,
        "longitude": lon2,
        "ref_latitude": drone_lat,
        "ref_longitude": drone_lon,
        "altitude_m": drone_altitude_m,
        "heading_deg": heading_deg,
        "simulated": True,
        "gps_source": "uav_target_locator_gopro",
        "gps_text": f"GPS: {lat2:.6f}, {lon2:.6f}",
    }


def undistort_pixel_to_normalized_point(
    u_px: float,
    v_px: float,
    intrinsics: CameraIntrinsics,
    iterations: int = 5,
) -> tuple[float, float]:
    x_distorted = (u_px - intrinsics.cx) / intrinsics.fx
    y_distorted = (v_px - intrinsics.cy) / intrinsics.fy

    if intrinsics.distortion_model == "none" or not intrinsics.distortion_coeffs:
        return (x_distorted, y_distorted)
    if intrinsics.distortion_model != "opencv":
        raise ValueError(f"unsupported distortion model: {intrinsics.distortion_model}")

    coeffs = list(intrinsics.distortion_coeffs) + [0.0] * 12
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = coeffs[:12]
    x = x_distorted
    y = y_distorted

    for _ in range(iterations):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        if radial_num == 0.0:
            break
        inverse_radial = radial_den / radial_num
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4
        x = (x_distorted - delta_x) * inverse_radial
        y = (y_distorted - delta_y) * inverse_radial

    return (x, y)


def camera_ray_to_body(
    x_norm: float,
    y_norm: float,
    gimbal: GimbalOrientation,
    camera_to_body_yaw_offset_deg: float = 0.0,
) -> tuple[float, float, float]:
    yaw_rad = radians(gimbal.yaw_deg + camera_to_body_yaw_offset_deg)
    pitch_rad = radians(gimbal.pitch_deg)
    roll_rad = radians(gimbal.roll_deg)

    optical_axis = (
        sin(yaw_rad) * cos(pitch_rad),
        cos(yaw_rad) * cos(pitch_rad),
        sin(pitch_rad),
    )
    image_right_axis = (cos(yaw_rad), -sin(yaw_rad), 0.0)
    image_down_axis = _cross_product(optical_axis, image_right_axis)

    if roll_rad:
        c = cos(roll_rad)
        s = sin(roll_rad)
        image_right_axis, image_down_axis = (
            _add_vectors(_scale_vector(image_right_axis, c), _scale_vector(image_down_axis, s)),
            _add_vectors(_scale_vector(image_right_axis, -s), _scale_vector(image_down_axis, c)),
        )

    ray = _add_vectors(
        _add_vectors(_scale_vector(image_right_axis, x_norm), _scale_vector(image_down_axis, y_norm)),
        optical_axis,
    )
    return _normalize_vector(ray)


def body_vector_to_enu(
    x_right: float,
    y_forward: float,
    z_up: float,
    heading_deg: float,
) -> tuple[float, float, float]:
    east, north = _body_to_enu(x_right, y_forward, heading_deg)
    return (east, north, z_up)


def _body_to_enu(x_right_m: float, y_forward_m: float, heading_deg: float) -> tuple[float, float]:
    heading_rad = radians(heading_deg)
    east_m = x_right_m * cos(heading_rad) + y_forward_m * sin(heading_rad)
    north_m = y_forward_m * cos(heading_rad) - x_right_m * sin(heading_rad)
    return (east_m, north_m)


def add_enu_to_gps(lat_deg: float, lon_deg: float, east_m: float, north_m: float) -> tuple[float, float]:
    lat_rad = radians(lat_deg)
    target_lat = lat_deg + degrees(north_m / EARTH_RADIUS_M)
    cos_lat = cos(lat_rad)
    if abs(cos_lat) < 1e-12:
        raise ValueError("longitude is undefined near the poles")
    target_lon = lon_deg + degrees(east_m / (EARTH_RADIUS_M * cos_lat))
    return (target_lat, target_lon)


def _cross_product(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _add_vectors(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale_vector(vector: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _normalize_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])
    if norm == 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return (vector[0] / norm, vector[1] / norm, vector[2] / norm)
