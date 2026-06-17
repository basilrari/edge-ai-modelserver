import base64
import os
import time
import traceback

import cv2
import psutil
import torch

from core.flood_grid import draw_grid_overlay
from core.model_manager import ModelManager
from core.inference_engine import InferenceEngine
from core.context_evaluator import ContextEvaluator
from core.model_selector import FloodModelSelector
from core.power_monitor import PowerMonitor, get_power_monitor
from core.flood_grid import serialize_grid_analysis
from core.video_geometry import REF_ALTITUDE_M
from core.runtime_profiler import RuntimeProfiler
from core.gpu_runtime import sync_all
from core.segment_policy import should_run_segmentation
from core.shared_camera import get_camera as _get_shared_camera

_model_manager = None
_engine = None
_context_evaluator = None
_selector = None
_power = None
_last_flood_ratio = 0.0
_last_mask = None

_process = psutil.Process(os.getpid())
_frame_count = 0
_inference_frame_count = 0
_inference_ms_accum = 0.0
_inference_fps_start = time.perf_counter()
_fps = 0.0
_peak_memory_mb = 0.0
_peak_cpu_percent = 0.0
_clf_load_ms = 0.0
_seg_load_ms = 0.0


def reset_session() -> None:
    """Called when live task switches — fresh frame counter + periodic segment on frame 1."""
    global _frame_count, _last_flood_ratio, _last_mask
    _frame_count = 0
    _last_flood_ratio = 0.0
    _last_mask = None


def _components():
    global _model_manager, _engine, _context_evaluator, _selector, _power
    if _model_manager is None:
        _power = get_power_monitor()
        _model_manager = ModelManager()
        _engine = InferenceEngine()
        _context_evaluator = ContextEvaluator()
        _selector = FloodModelSelector()
        RuntimeProfiler()
    return _model_manager, _engine, _context_evaluator, _selector, _power


def _get_camera():
    return _get_shared_camera()


def _display_name(model_key):
    return {"resnet18": "ResNet18", "deeplabv3plus": "DeepLabv3+"}.get(
        model_key, model_key
    )


def _grid_for_api(grid_analysis):
    return serialize_grid_analysis(grid_analysis)


def _status_from_result(display_label, flood_ratio):
    if display_label == "Flooded":
        if flood_ratio > 0.70:
            return "CRITICAL"
        if flood_ratio > 0.40:
            return "ALERT"
        return "WARNING"
    return "NORMAL"


def _update_fps(total_inference_ms):
    global _fps, _inference_frame_count, _inference_ms_accum, _inference_fps_start
    _inference_frame_count += 1
    _inference_ms_accum += total_inference_ms
    elapsed = time.perf_counter() - _inference_fps_start
    if elapsed >= 1.0 and _inference_ms_accum > 0:
        _fps = (_inference_frame_count * 1000.0) / _inference_ms_accum
        _inference_frame_count = 0
        _inference_ms_accum = 0.0
        _inference_fps_start = time.perf_counter()


def detect_flood(frame=None, encode_frame=True, sample_power=True):
    global _frame_count, _peak_memory_mb, _peak_cpu_percent, _clf_load_ms, _seg_load_ms
    global _last_flood_ratio, _last_mask

    try:
        model_manager, engine, context_evaluator, selector, power = _components()
        total_start = time.perf_counter()

        frame = frame if frame is not None else _get_camera().get_frame()
        if frame is None:
            return {"error": "Failed to read frame from camera"}

        _frame_count += 1

        context = context_evaluator.get_context()
        context.setdefault("flood_ratio", 0.0)

        pre = selector.select_models(context)
        switches = selector.apply_selection(pre)

        t0 = time.perf_counter()
        clf_model = model_manager.load_flood_classifier()
        if _clf_load_ms == 0.0:
            _clf_load_ms = (time.perf_counter() - t0) * 1000.0

        # Fast path: classify first, skip DeepLab (~3s on Jetson) when scene is dry.
        clf_only = engine.run_flood_pipeline(
            clf_model, None, frame, run_segmenter=False
        )
        clf_pred = clf_only["clf_class_index"]
        run_seg = should_run_segmentation(
            _frame_count,
            clf_pred,
            _last_flood_ratio,
            pre["run_segmenter"],
        ) or _frame_count <= 1

        seg_model = None
        if run_seg:
            t1 = time.perf_counter()
            seg_model = model_manager.load_flood_segmenter()
            if _seg_load_ms == 0.0:
                _seg_load_ms = (time.perf_counter() - t1) * 1000.0

        if run_seg and seg_model is not None:
            seg_out = engine.run_segmentation(seg_model, frame)
            segmentation_ms = float(seg_out.pop("inference_ms", 0.0))
            mask = seg_out.pop("mask", None)
            flood_ratio = float(seg_out.get("flood_ratio", 0.0))
            flood_run = {
                **clf_only,
                "segmentation_ms": segmentation_ms,
                "segmentation": {"flood_ratio": flood_ratio, **seg_out},
                "mask": mask,
                "flood_ratio": flood_ratio,
                "segmentation_skipped": False,
                "total_inference_ms": clf_only["classification_ms"] + segmentation_ms,
            }
        else:
            flood_run = {
                **clf_only,
                "segmentation_ms": 0.0,
                "segmentation": {},
                "mask": _last_mask,
                "flood_ratio": _last_flood_ratio,
                "total_inference_ms": clf_only["classification_ms"],
            }

        if flood_run.get("mask") is not None:
            _last_mask = flood_run["mask"]
        if flood_run.get("flood_ratio", 0) > 0 or run_seg:
            _last_flood_ratio = float(flood_run.get("flood_ratio", 0))
        clf_result = flood_run["classification"]
        classification_ms = flood_run["classification_ms"]
        raw_classification = clf_result.get("label", "unknown")
        context["classification_label"] = raw_classification

        flood_ratio = flood_run["flood_ratio"]
        segmentation_ms = flood_run["segmentation_ms"]
        seg_result = flood_run["segmentation"]
        mask = flood_run["mask"]
        grid_analysis = None

        context["flood_ratio"] = flood_ratio

        post = selector.select_models(context)
        post_switches = selector.apply_selection(post)
        switches = {**switches, **post_switches}

        primary = selector.primary_model
        seg_active = selector.is_segmentation_active(flood_ratio)
        display_label = "Flooded" if seg_active else "Non-Flooded"
        status = _status_from_result(display_label, flood_ratio)

        total_inference_ms = flood_run["total_inference_ms"]
        _update_fps(total_inference_ms)
        total_latency_ms = (time.perf_counter() - total_start) * 1000.0
        instant_fps = (
            1000.0 / total_inference_ms if total_inference_ms > 0 else 0.0
        )

        sync_all()
        power_metrics = power.record_inference_power()

        memory_mb = _process.memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=None)
        _peak_memory_mb = max(_peak_memory_mb, memory_mb)
        _peak_cpu_percent = max(_peak_cpu_percent, cpu_percent)

        out_frame = frame
        show_grid = seg_active and mask is not None
        if show_grid:
            out_frame, grid_analysis = draw_grid_overlay(
                frame,
                mask,
                drone_altitude_m=REF_ALTITUDE_M,
            )

        result_frame = out_frame
        frame_base64 = None
        if encode_frame:
            _, buffer = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frame_base64 = base64.b64encode(buffer).decode("utf-8")

        active = {
            "classifier": _display_name(FloodModelSelector.RESNET18),
            "segmenter": _display_name(FloodModelSelector.DEEPLAB),
            "primary": _display_name(primary),
            "classifier_key": FloodModelSelector.RESNET18,
            "segmenter_key": FloodModelSelector.DEEPLAB,
            "primary_key": primary,
            "classifier_backend": getattr(model_manager, "clf_backend", "pytorch"),
            "segmenter_backend": getattr(model_manager, "seg_backend", "pytorch"),
        }

        switch_log = []
        for change in switches.values():
            switch_log.append(
                f"{_display_name(change['from'])} → {_display_name(change['to'])}"
            )

        camera_device = getattr(_get_camera(), "device_path", "unknown")
        log = (
            f"[{time.strftime('%H:%M:%S')}] cam={camera_device} "
            f"class={display_label} (raw={raw_classification}) ratio={flood_ratio:.2f} "
            f"primary={active['primary']} seg_active={seg_active} status={status} grid={show_grid}"
        )
        if switch_log:
            log += " | SWITCH " + "; ".join(switch_log)
        print(log)

        metrics = {
            "fps": round(_fps, 2),
            "instant_fps": round(instant_fps, 2),
            "total_inference_ms": round(total_inference_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "classification_ms": round(classification_ms, 2),
            "segmentation_ms": round(segmentation_ms, 2),
            "segmentation_skipped": flood_run.get("segmentation_skipped", not run_seg),
            "model_switch_latency_ms": round(total_inference_ms, 2),
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
            "peak_memory_mb": round(_peak_memory_mb, 2),
            "peak_cpu_percent": round(_peak_cpu_percent, 2),
            "clf_load_ms": round(_clf_load_ms, 2),
            "seg_load_ms": round(_seg_load_ms, 2),
            "flood_ratio": round(flood_ratio, 4),
            "idle_power_w": power_metrics["idle_power_w"],
            "inference_power_w": power_metrics["inference_power_w"],
            "extra_power_w": power_metrics["extra_power_w"],
            "peak_inference_power_w": power_metrics["peak_inference_power_w"],
            "peak_extra_power_w": power_metrics["peak_extra_power_w"],
            "power_w": power_metrics["inference_power_w"],
            "peak_power_w": power_metrics["peak_inference_power_w"],
        }

        payload = {
            "classification": {
                **clf_result,
                "raw_label": raw_classification,
                "label": display_label,
            },
            "segmentation": {
                "flood_ratio": flood_ratio,
                "active": seg_active,
                **seg_result,
            },
            "segmentation_active": seg_active,
            "grid": _grid_for_api(grid_analysis),
            "overlay_mode": "grid" if show_grid else "none",
            "context": context,
            "selected_models": selector.current_models.copy(),
            "primary_model": primary,
            "active_models": active,
            "model_switches": switches,
            "selection_metadata": post.get("metadata", {}),
            "camera": {"device": camera_device},
            "system": {
                "status": status,
                "fps": round(instant_fps, 2),
                "latency_ms": round(total_latency_ms, 2),
            },
            "metrics": metrics,
            "power": power_metrics,
            "log": log,
        }
        if encode_frame:
            payload["frame_base64"] = frame_base64
            payload["frame"] = frame_base64
        else:
            payload["_out_frame"] = result_frame
        return payload

    except Exception as e:
        print("Exception:", str(e))
        traceback.print_exc()
        return {"error": str(e)}
