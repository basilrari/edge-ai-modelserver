"""Run flood + human detection on the same camera frame (rescue scenario)."""

import base64
import time

import cv2

from core.shared_camera import get_camera, get_frame
from tools.detect_flood import detect_flood
from tools.detect_human import detect_human, draw_humans_on_frame


def _merge_status(flood_status: str, human_status: str) -> str:
    priority = {"CRITICAL": 4, "ALERT": 3, "WARNING": 2, "NORMAL": 1, "IDLE": 0}
    return max(
        (flood_status, human_status),
        key=lambda s: priority.get(s, 0),
    )


def detect_flood_and_human():
    frame = get_frame()
    if frame is None:
        return {"error": "Failed to read frame from camera", "task": "detect_combined"}

    total_start = time.perf_counter()

    flood = detect_flood(frame=frame, encode_frame=False)
    if flood.get("error"):
        flood["active_tools"] = ["detect_flood", "detect_human"]
        return flood

    out_frame = flood.pop("_out_frame", frame.copy())
    human = detect_human(frame=frame, encode_frame=False)
    if human.get("error"):
        human["active_tools"] = ["detect_flood", "detect_human"]
        return human

    humans = human.get("humans", [])
    composed = draw_humans_on_frame(out_frame, humans)

    _, buffer = cv2.imencode(".jpg", composed, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    total_ms = (time.perf_counter() - total_start) * 1000.0
    flood_ms = flood.get("metrics", {}).get("total_inference_ms", 0)
    human_ms = human.get("metrics", {}).get("detection_ms", 0)

    flood_status = flood.get("system", {}).get("status", "NORMAL")
    human_status = human.get("system", {}).get("status", "NORMAL")
    combined_status = _merge_status(flood_status, human_status)

    seg_active = flood.get("segmentation_active", False)
    human_count = human.get("human_count", 0)
    flood_label = flood.get("classification", {}).get("label", "—")

    log = (
        f"[{time.strftime('%H:%M:%S')}] COMBINED flood={flood_label} "
        f"ratio={flood.get('segmentation', {}).get('flood_ratio', 0):.2f} "
        f"humans={human_count} total={total_ms:.0f}ms cam={get_camera().device_path}"
    )
    print(log)

    overlay_parts = []
    if seg_active:
        overlay_parts.append("flood grid")
    if human_count:
        overlay_parts.append("human boxes")
    overlay = " + ".join(overlay_parts) if overlay_parts else "flood + human scan"

    return {
        "task": "detect_combined",
        "active_tools": ["detect_flood", "detect_human"],
        "active_tool": "detect_combined",
        "mode": "combined",
        **flood,
        "human_count": human_count,
        "humans": humans,
        "human_detection": {
            "human_count": human_count,
            "humans": humans,
            "metrics": human.get("metrics", {}),
            "system": human.get("system", {}),
            "active_models": human.get("active_models", {}),
        },
        "camera": {"device": get_camera().device_path},
        "system": {
            "status": combined_status,
            "flood_status": flood_status,
            "human_status": human_status,
            "latency_ms": round(total_ms, 2),
            "fps": round(1000.0 / total_ms, 2) if total_ms > 0 else 0,
        },
        "metrics": {
            **flood.get("metrics", {}),
            "human_detection_ms": human.get("metrics", {}).get("detection_ms", 0),
            "combined_latency_ms": round(total_ms, 2),
            "human_count": human_count,
        },
        "overlay_mode": overlay,
        "log": log,
        "frame_base64": frame_b64,
        "frame": frame_b64,
    }
