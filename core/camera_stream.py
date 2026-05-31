import glob
import os
import time

# Silence OpenCV probe noise (harmless failed opens on metadata / busy nodes)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

import cv2

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except AttributeError:
    pass


def _is_metadata_device(device_path: str) -> bool:
    name = os.path.basename(device_path)
    sysfs = f"/sys/class/video4linux/{name}/name"
    try:
        with open(sysfs, encoding="utf-8") as f:
            label = f.read().strip().lower()
        return "metadata" in label
    except OSError:
        return False


def _list_v4l_capture_devices():
    paths = []
    for path in sorted(glob.glob("/dev/video*")):
        if _is_metadata_device(path):
            continue
        paths.append(path)
    return paths


class CameraStream:
    """Open external USB / V4L2 capture device on Jetson (skips metadata nodes)."""

    def __init__(self, device=None, width=640, height=480):
        self.width = width
        self.height = height
        self.device = device or os.environ.get("CAMERA_DEVICE")
        self.cap = None
        self.device_path = None
        self._init_camera()

    def _open_device(self, path: str):
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return None
        return cap

    def _candidate_devices(self):
        if self.device:
            yield self.device
            return
        yield from _list_v4l_capture_devices()

    def _init_camera(self):
        candidates = list(self._candidate_devices())
        if not candidates:
            raise RuntimeError(
                "No V4L2 capture devices under /dev/video*. "
                "Plug in the USB camera or set CAMERA_DEVICE=/dev/videoX"
            )

        print(f"[CAMERA] trying {len(candidates)} device(s)...")
        for path in candidates:
            cap = self._open_device(path)
            if cap is None:
                continue
            self.cap = cap
            self.device_path = path
            print(f"[CAMERA] ready on {self.device_path}")
            return

        raise RuntimeError(
            f"No working camera among: {candidates}. "
            "Set CAMERA_DEVICE to the correct node (often /dev/video0 or /dev/video1)."
        )

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            print("[CAMERA] reinitializing...")
            self._init_camera()

        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame

        print("[CAMERA] frame read failed, retrying...")
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._init_camera()
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("[CAMERA] released")
