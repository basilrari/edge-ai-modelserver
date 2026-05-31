import os

import cv2
from core.camera_stream import CameraStream
from core.model_manager import ModelManager
from core.inference_engine import InferenceEngine

_model_manager = None
_engine = None
_camera = None


def _components():
    global _model_manager, _engine
    if _model_manager is None:
        _model_manager = ModelManager()
        _engine = InferenceEngine()
    return _model_manager, _engine


def _get_camera():
    global _camera
    if _camera is None:
        _camera = CameraStream(device=os.environ.get("CAMERA_DEVICE"))
    return _camera


def detect_human():
    model_manager, engine = _components()
    frame = _get_camera().get_frame()
    if frame is None:
        return {"error": "Camera not accessible"}

    model = model_manager.load_human_detector()
    results = engine.run_detection(model, frame)

    return {"detections": str(results)}