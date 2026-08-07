"""Start/stop GoPro USB preview from Drone_LLM (single owner for WebRTC stack)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_started = False


def _perception_dir() -> Path:
    repo = Path(__file__).resolve().parents[1]
    return Path(
        os.environ.get(
            "DRONE_PERCEPTION_PATH",
            repo.parent / "drone-competition" / "perception",
        )
    ).resolve()


def _backend_gopro() -> bool:
    return os.environ.get("CAMERA_BACKEND", "v4l2").strip().lower() in (
        "gopro",
        "go_pro",
        "udp",
    )


def start_gopro_preview_if_needed() -> None:
    global _started
    if _started or os.environ.get("GOPRO_PREVIEW_EXTERNAL", "0") == "1":
        return
    if not _backend_gopro():
        return
    perception = _perception_dir()
    if not perception.is_dir():
        raise RuntimeError(f"DRONE_PERCEPTION_PATH not found: {perception}")
    if str(perception) not in sys.path:
        sys.path.insert(0, str(perception))
    from gopro_enable import start_gopro_webcam_session  # noqa: WPS433

    start_gopro_webcam_session()
    _started = True
    print(f"[CAMERA] GoPro USB preview started ({perception})")


def stop_gopro_preview_if_started() -> None:
    global _started
    if not _started:
        return
    perception = _perception_dir()
    if str(perception) not in sys.path:
        sys.path.insert(0, str(perception))
    try:
        from gopro_enable import stop_gopro_webcam_session  # noqa: WPS433

        stop_gopro_webcam_session()
    except Exception as exc:  # noqa: BLE001
        print(f"[CAMERA] GoPro preview stop: {exc}")
    _started = False
