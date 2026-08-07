"""WebRTC live view for the SAR frontend (uses shared V4L2 camera)."""

from __future__ import annotations

import asyncio
import logging
import os

import cv2
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from av import VideoFrame

logger = logging.getLogger("drone_llm.webrtc")

relay = MediaRelay()
_pcs: set[RTCPeerConnection] = set()
_shared_track: "SharedCameraVideoTrack | None" = None


from core.webrtc_ice import ice_servers_for_peer


def warmup_camera() -> None:
    from core.shared_camera import get_camera

    cam = get_camera()
    frame = cam.get_frame()
    if frame is None:
        raise RuntimeError(f"camera opened on {cam.device_path} but no frame yet")


class SharedCameraVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._target_fps = max(1.0, float(os.environ.get("CAMERA_FPS", "12")))
        self._max_width = max(320, int(os.environ.get("WEBRTC_MAX_WIDTH", "960")))

    async def recv(self) -> VideoFrame:
        from core.shared_camera import get_camera

        pts, time_base = await self.next_timestamp()
        frame_bgr = await asyncio.to_thread(get_camera().get_frame)
        wait = 0
        while frame_bgr is None and wait < 50:
            await asyncio.sleep(0.02)
            wait += 1
            frame_bgr = await asyncio.to_thread(get_camera().get_frame)
        if frame_bgr is None:
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = frame_bgr.shape[:2]
        if w > self._max_width:
            nh = max(1, int(h * self._max_width / w))
            frame_bgr = cv2.resize(
                frame_bgr, (self._max_width, nh), interpolation=cv2.INTER_AREA
            )

        video = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        video.pts = pts
        video.time_base = time_base
        await asyncio.sleep(max(0.0, (1.0 / self._target_fps) * 0.85))
        return video


def ensure_track() -> SharedCameraVideoTrack:
    global _shared_track
    if _shared_track is None:
        _shared_track = SharedCameraVideoTrack()
    return _shared_track


async def close_all_peers() -> None:
    coros = [pc.close() for pc in list(_pcs)]
    _pcs.clear()
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


async def handle_offer(sdp: str, offer_type: str) -> dict[str, str]:
    # Stale peers after a crash/reconnect can destabilize aiortc on Jetson.
    if len(_pcs) >= 2:
        await close_all_peers()

    offer = RTCSessionDescription(sdp=sdp, type=offer_type)
    configuration = RTCConfiguration(iceServers=ice_servers_for_peer())
    pc = RTCPeerConnection(configuration=configuration)
    _pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if pc.connectionState in ("failed", "closed"):
            await pc.close()
            _pcs.discard(pc)

    pc.addTrack(relay.subscribe(ensure_track()))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    for _ in range(100):
        if pc.iceGatheringState == "complete":
            break
        await asyncio.sleep(0.05)

    local = pc.localDescription
    if local is None:
        raise RuntimeError("missing local WebRTC description")

    sdp_out = local.sdp
    announced = os.environ.get("WEBRTC_ANNOUNCED_IP", "").strip()
    if announced:
        sdp_out = sdp_out.replace("127.0.0.1", announced)
        sdp_out = sdp_out.replace("0.0.0.0", announced)

    return {"sdp": sdp_out, "type": local.type}


def camera_status() -> dict:
    backend = os.environ.get("CAMERA_BACKEND", "v4l2").strip().lower()
    try:
        from core.shared_camera import get_camera

        cam = get_camera()
        frame = cam.get_frame()
        return {
            "connected": frame is not None,
            "device": getattr(cam, "device_path", None),
            "width": getattr(cam, "width", None),
            "height": getattr(cam, "height", None),
            "backend": backend,
        }
    except Exception as e:
        return {"connected": False, "error": str(e), "backend": backend}
