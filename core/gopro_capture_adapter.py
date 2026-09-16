"""GoPro USB preview via drone-competition/perception VideoSource.

Set CAMERA_BACKEND=gopro and DRONE_PERCEPTION_PATH to the perception repo.
The preview publishes UDP frames that GoproCameraStream reads for WebRTC and
detection tools (same get_frame() surface as V4L2 CameraStream).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_PERCEPTION = Path(
    os.environ.get(
        "DRONE_PERCEPTION_PATH",
        _REPO / "drone-competition" / "perception",
    )
).resolve()


class GoproCameraStream:
    """Same surface as CameraStream for WebRTC + detection tools."""

    def __init__(self) -> None:
        if not _PERCEPTION.is_dir():
            raise RuntimeError(f"DRONE_PERCEPTION_PATH not found: {_PERCEPTION}")
        if str(_PERCEPTION) not in sys.path:
            sys.path.insert(0, str(_PERCEPTION))

        from video_source import VideoSource  # noqa: WPS433

        self.width = int(os.environ.get("CAMERA_WIDTH", "1280"))
        self.height = int(os.environ.get("CAMERA_HEIGHT", "720"))
        self.device_path = os.environ.get("GOPRO_UDP_URL", "gopro-udp")
        self._vs = VideoSource()
        self._vs.start()
        print(f"[CAMERA] GoPro capture via {_PERCEPTION}")

    def get_frame(self):
        frame, seq, _ = self._vs.latest_frame()
        if frame is None:
            time.sleep(0.02)
            frame, seq, _ = self._vs.latest_frame()
        return frame

    def frame_age_sec(self) -> float | None:
        """Seconds since the last real frame arrived (None before first frame)."""
        _, _, arrival = self._vs.latest_frame()
        if not arrival:
            return None
        return max(0.0, time.time() - arrival)

    def release(self) -> None:
        self._vs.stop()
        print("[CAMERA] GoPro released")
