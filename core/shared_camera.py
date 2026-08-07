"""Single shared camera for all detection tools (one frame per combined inference)."""

import os

from core.camera_stream import CameraStream

_camera = None


def _backend() -> str:
    return os.environ.get("CAMERA_BACKEND", "v4l2").strip().lower()


def get_camera():
    global _camera
    if _camera is None:
        backend = _backend()
        if backend in ("gopro", "go_pro", "udp"):
            from core.gopro_capture_adapter import GoproCameraStream

            _camera = GoproCameraStream()
        else:
            _camera = CameraStream(device=os.environ.get("CAMERA_DEVICE"))
    return _camera


def get_frame():
    return get_camera().get_frame()
