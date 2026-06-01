import base64
import os
import time

import cv2

from core.model_manager import ModelManager
from core.shared_camera import get_camera as _get_shared_camera

PERSON_CLASS_ID = 0

_model_manager = None
_human_fps = 0.0


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
        cv2.putText(
            out,
            f"person {h['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
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


def detect_human(frame=None, encode_frame=True):
    global _human_fps

    frame = frame if frame is not None else _get_camera().get_frame()
    if frame is None:
        return {"error": "Camera not accessible", "task": "detect_human"}

    model = _get_model_manager().load_human_detector()
    total_start = time.perf_counter()
    results = model.predict(
        frame,
        conf=0.4,
        classes=[PERSON_CLASS_ID],
        verbose=False,
        device=0,
    )
    infer_ms = (time.perf_counter() - total_start) * 1000.0
    _human_fps = round(1000.0 / infer_ms, 2) if infer_ms > 0 else 0.0

    humans = _parse_humans(results[0])
    out = _draw_humans(frame, humans)

    frame_b64 = None
    if encode_frame:
        _, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")

    human_count = len(humans)
    status = "ALERT" if human_count > 0 else "NORMAL"
    log = (
        f"[{time.strftime('%H:%M:%S')}] YOLOv8n humans={human_count} "
        f"infer={infer_ms:.1f}ms cam={_get_camera().device_path}"
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
        },
        "active_models": {
            "detector": "YOLOv8n",
            "detector_key": "yolov8n",
        },
        "overlay_mode": "yolo_boxes",
        "log": log,
    }
    if encode_frame:
        payload["frame_base64"] = frame_b64
        payload["frame"] = frame_b64
    return payload
