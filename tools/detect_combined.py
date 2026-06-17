"""Run flood + human detection on the same camera frame (rescue scenario)."""

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import psutil

from core.gpu_runtime import sync_all
from core.perf_config import PARALLEL_COMBINED
from core.power_monitor import get_power_monitor
from core.shared_camera import get_camera, get_frame
from tools.detect_flood import detect_flood
from tools.detect_human import draw_humans_on_frame, run_human_inference

_process = psutil.Process(os.getpid())


def _merge_status(flood_status: str, human_status: str) -> str:
    priority = {"CRITICAL": 4, "ALERT": 3, "WARNING": 2, "NORMAL": 1, "IDLE": 0}
    return max(
        (flood_status, human_status),
        key=lambda s: priority.get(s, 0),
    )


def _human_payload(humans, infer_ms, backend):
    human_count = len(humans)
    status = "ALERT" if human_count > 0 else "NORMAL"
    fps = round(1000.0 / infer_ms, 2) if infer_ms > 0 else 0.0
    return {
        "human_count": human_count,
        "humans": humans,
        "system": {"status": status, "fps": fps, "latency_ms": round(infer_ms, 2)},
        "metrics": {
            "detection_ms": round(infer_ms, 2),
            "total_latency_ms": round(infer_ms, 2),
            "instant_fps": fps,
            "human_count": human_count,
        },
        "active_models": {
            "detector": "YOLOv8n",
            "detector_key": "yolov8n",
            "backend": backend,
        },
    }


def detect_flood_and_human(frame=None, encode_frame=True):
    frame = frame if frame is not None else get_frame()
    if frame is None:
        return {"error": "Failed to read frame from camera", "task": "detect_combined"}

    total_start = time.perf_counter()
    parallel = PARALLEL_COMBINED

    if parallel:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="infer") as pool:
            flood_future = pool.submit(
                detect_flood, frame=frame, encode_frame=False
            )
            human_future = pool.submit(run_human_inference, frame)
            flood = flood_future.result()
            humans, human_ms, backend = human_future.result()
        sync_all()
        human = _human_payload(humans, human_ms, backend)
    else:
        flood = detect_flood(frame=frame, encode_frame=False)
        if flood.get("error"):
            flood["active_tools"] = ["detect_flood", "detect_human"]
            return flood
        humans, human_ms, backend = run_human_inference(frame)
        human = _human_payload(humans, human_ms, backend)

    if flood.get("error"):
        flood["active_tools"] = ["detect_flood", "detect_human"]
        return flood

    out_frame = flood.pop("_out_frame", frame.copy())
    humans = human.get("humans", [])
    composed = draw_humans_on_frame(out_frame, humans)

    frame_b64 = None
    if encode_frame:
        _, buffer = cv2.imencode(".jpg", composed, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

    total_ms = (time.perf_counter() - total_start) * 1000.0
    flood_ms = flood.get("metrics", {}).get("total_inference_ms", 0)
    human_ms = human.get("metrics", {}).get("detection_ms", 0)
    overlap_note = "parallel" if parallel else "sequential"

    flood_status = flood.get("system", {}).get("status", "NORMAL")
    human_status = human.get("system", {}).get("status", "NORMAL")
    combined_status = _merge_status(flood_status, human_status)

    seg_active = flood.get("segmentation_active", False)
    human_count = human.get("human_count", 0)
    flood_label = flood.get("classification", {}).get("label", "—")

    log = (
        f"[{time.strftime('%H:%M:%S')}] COMBINED({overlap_note}) flood={flood_label} "
        f"ratio={flood.get('segmentation', {}).get('flood_ratio', 0):.2f} "
        f"humans={human_count} flood_ms={flood_ms:.0f} human_ms={human_ms:.0f} "
        f"total={total_ms:.0f}ms src={getattr(frame, 'shape', 'frame')}"
    )
    print(log)

    overlay_parts = []
    if seg_active:
        overlay_parts.append("flood grid")
    if human_count:
        overlay_parts.append("human boxes")
    overlay = " + ".join(overlay_parts) if overlay_parts else "flood + human scan"

    power_metrics = get_power_monitor().record_inference_power()
    memory_mb = _process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=None)

    payload = {
        "task": "detect_combined",
        "active_tools": ["detect_flood", "detect_human"],
        "active_tool": "detect_combined",
        "mode": "combined",
        "inference_mode": overlap_note,
        **flood,
        "human_count": human_count,
        "humans": humans,
        "human_detection": human,
        "camera": {"device": getattr(get_camera(), "device_path", "offline")},
        "system": {
            "status": combined_status,
            "flood_status": flood_status,
            "human_status": human_status,
            "latency_ms": round(total_ms, 2),
            "fps": round(1000.0 / total_ms, 2) if total_ms > 0 else 0,
        },
        "metrics": {
            **flood.get("metrics", {}),
            "human_detection_ms": human_ms,
            "combined_latency_ms": round(total_ms, 2),
            "human_count": human_count,
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
            "idle_power_w": power_metrics["idle_power_w"],
            "inference_power_w": power_metrics["inference_power_w"],
            "extra_power_w": power_metrics["extra_power_w"],
            "peak_inference_power_w": power_metrics["peak_inference_power_w"],
            "peak_extra_power_w": power_metrics["peak_extra_power_w"],
            "power_w": power_metrics["inference_power_w"],
            "peak_power_w": power_metrics["peak_inference_power_w"],
        },
        "power": power_metrics,
        "overlay_mode": overlay,
        "log": log,
    }
    if encode_frame and frame_b64:
        payload["frame_base64"] = frame_b64
        payload["frame"] = frame_b64
    else:
        payload["_out_frame"] = composed
    return payload
