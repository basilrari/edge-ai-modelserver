"""Single shared camera for all detection tools (one frame per combined inference)."""

import os

from core.camera_stream import CameraStream

_camera = None


def get_camera() -> CameraStream:
    global _camera
    if _camera is None:
        _camera = CameraStream(device=os.environ.get("CAMERA_DEVICE"))
    return _camera


def get_frame():
    return get_camera().get_frame()
