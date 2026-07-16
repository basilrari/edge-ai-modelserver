import base64
import os
import time

import cv2
import psutil

from core.flood_grid import attach_gps_to_humans
from core.gpu_runtime import cuda_stream, get_human_stream, sync_all
from core.model_manager import ModelManager
from core.context_evaluator import ContextEvaluator
from core.human_detector_tier import resolve_tier_for_context
from core.perf_config import YOLO_IMGSZ, YOLO_ROBUST_IMGSZ
from core.power_monitor import get_power_monitor
from core.shared_camera import get_camera as _get_shared_camera

HUMAN_LABEL = "human"

_model_manager = None
_context_evaluator = None
_human_fps = 0.0
_process = psutil.Process(os.getpid())


def _get_model_manager():
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


def reset_session() -> None:
    """Called when live task switches — fresh human tier session."""
    global _human_fps
    from core.human_detector_tier import get_selector

    _human_fps = 0.0
    get_selector().reset()


def _get_context_evaluator():
    global _context_evaluator
    if _context_evaluator is None:
        _context_evaluator = ContextEvaluator()
    return _context_evaluator


def get_session_context() -> dict:
    try:
        from tools.detect_flood import get_session_context as flood_ctx

        return dict(flood_ctx())
    except Exception:
        return {}


def _build_context(extra: dict | None = None) -> dict:
    context = _get_context_evaluator().get_context()
    context.update(get_session_context())
    if extra:
        context.update(extra)
    if "priority" not in context:
        context["priority"] = float(context.get("mission_priority", 0.5))
    return context


def _get_camera():
    return _get_shared_camera()


def _parse_humans(result, class_ids, conf_min=0.4):
    humans = []
    if result.boxes is None or len(result.boxes) == 0:
        return humans
    allowed = set(class_ids)
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in allowed or conf < conf_min:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        humans.append(
            {
                "label": HUMAN_LABEL,
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
        label = f"human {h['confidence']:.2f}"
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


def run_human_inference(frame, conf_min=0.4, context_extra: dict | None = None):
    """GPU human detection only (no JPEG encode / overlay)."""
    context = _build_context(context_extra)
    tier_info = resolve_tier_for_context(context)

    manager = _get_model_manager()
    model = manager.load_human_detector()
    total_start = time.perf_counter()

    with cuda_stream(get_human_stream()):
        results = model.predict(
            frame,
            conf=conf_min,
            classes=list(manager.human_class_ids),
            verbose=False,
            device=0,
            imgsz=manager.human_imgsz,
            half=True,
            max_det=10,
        )
    sync_all()

    infer_ms = (time.perf_counter() - total_start) * 1000.0
    humans = attach_gps_to_humans(
        _parse_humans(results[0], manager.human_class_ids, conf_min=conf_min)
    )
    return (
        humans,
        infer_ms,
        manager.human_backend,
        manager.human_tier,
        manager.human_detector_key,
        tier_info,
    )


def detect_human(frame=None, encode_frame=True):
    global _human_fps

    frame = frame if frame is not None else _get_camera().get_frame()
    if frame is None:
        return {"error": "Camera not accessible", "task": "detect_human"}

    humans, infer_ms, backend, tier, detector_key, tier_info = run_human_inference(frame)
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
    imgsz = YOLO_ROBUST_IMGSZ if tier == "robust" else YOLO_IMGSZ
    selection_mode = tier_info.get("mode", "auto")
    log = (
        f"[{time.strftime('%H:%M:%S')}] {detector_key}({backend}) tier={tier} "
        f"mode={selection_mode} humans={human_count} infer={infer_ms:.1f}ms "
        f"imgsz={imgsz} cam={_get_camera().device_path}"
    )
    print(log)

    tier_switches = tier_info.get("tier_switches") or {}
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
        "human_detector": tier_info,
        "model_switches": tier_switches,
        "active_models": {
            "detector": detector_key,
            "detector_key": detector_key,
            "tier": tier,
            "backend": backend,
            "imgsz": imgsz,
            "human_label": HUMAN_LABEL,
            "mode": selection_mode,
            "selection": tier_info.get("metadata"),
        },
        "overlay_mode": "yolo_boxes",
        "log": log,
    }
    if encode_frame:
        payload["frame_base64"] = frame_b64
        payload["frame"] = frame_b64
    return payload
