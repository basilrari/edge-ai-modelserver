import base64
import os
import time

import cv2
import psutil

from core.flood_grid import attach_gps_to_humans
from core.gpu_runtime import cuda_stream, get_human_stream, sync_all
from core.model_manager import ModelManager
from core.perf_config import YOLO_IMGSZ
from core.power_monitor import get_power_monitor
from core.shared_camera import get_camera as _get_shared_camera

PERSON_CLASS_ID = 0

_model_manager = None
_human_fps = 0.0
_process = psutil.Process(os.getpid())


def _get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def _get_camera():
    return _get_shared_camera()


def _parse_humans(result, conf_min=0.4):
    humans = []
    if result.boxes is None or len(result.boxes) == 0:
        return humans
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id != PERSON_CLASS_ID or conf < conf_min:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        humans.append(
            {
                "label": "person",
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return humans


def _draw_humans(frame, humans):
    out = frame.copy()
    for h in humans:
        x1, y1, x2, y2 = h["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"person {h['confidence']:.2f}"
        if h.get("latitude") is not None and h.get("longitude") is not None:
            label = f"{h['latitude']:.5f},{h['longitude']:.5f}"
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        out,
        f"humans: {len(humans)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return out


def draw_humans_on_frame(frame, humans):
    return _draw_humans(frame, humans)


def run_human_inference(frame, conf_min=0.4):
    """GPU human detection only (no JPEG encode / overlay)."""
    manager = _get_model_manager()
    model = manager.load_human_detector()
    total_start = time.perf_counter()

    with cuda_stream(get_human_stream()):
        results = model.predict(
            frame,
            conf=conf_min,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            device=0,
            imgsz=manager.human_imgsz,
            half=True,
            max_det=10,
        )
    sync_all()

    infer_ms = (time.perf_counter() - total_start) * 1000.0
    humans = attach_gps_to_humans(_parse_humans(results[0], conf_min=conf_min))
    return humans, infer_ms, manager.human_backend


def detect_human(frame=None, encode_frame=True):
    global _human_fps

    frame = frame if frame is not None else _get_camera().get_frame()
    if frame is None:
        return {"error": "Camera not accessible", "task": "detect_human"}

    humans, infer_ms, backend = run_human_inference(frame)
    _human_fps = round(1000.0 / infer_ms, 2) if infer_ms > 0 else 0.0

    power_metrics = get_power_monitor().record_inference_power()
    memory_mb = _process.memory_info().rss / (1024 * 1024)
    cpu_percent = psutil.cpu_percent(interval=None)

    frame_b64 = None
    if encode_frame:
        out = _draw_humans(frame, humans)
        _, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

    human_count = len(humans)
    status = "ALERT" if human_count > 0 else "NORMAL"
    log = (
        f"[{time.strftime('%H:%M:%S')}] YOLOv8n({backend}) humans={human_count} "
        f"infer={infer_ms:.1f}ms imgsz={YOLO_IMGSZ} cam={_get_camera().device_path}"
    )
    print(log)

    payload = {
        "task": "detect_human",
        "human_count": human_count,
        "humans": humans,
        "detections": humans,
        "camera": {"device": _get_camera().device_path},
        "system": {
            "status": status,
            "fps": _human_fps,
            "latency_ms": round(infer_ms, 2),
        },
        "metrics": {
            "detection_ms": round(infer_ms, 2),
            "total_latency_ms": round(infer_ms, 2),
            "instant_fps": _human_fps,
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
        "active_models": {
            "detector": "YOLOv8n",
            "detector_key": "yolov8n",
            "backend": backend,
            "imgsz": YOLO_IMGSZ,
        },
        "overlay_mode": "yolo_boxes",
        "log": log,
    }
    if encode_frame:
        payload["frame_base64"] = frame_b64
        payload["frame"] = frame_b64
    return payload
