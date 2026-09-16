"""WebRTC live camera stream for remote viewers (V4L2 or GoPro via shared_camera)."""

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
from av import VideoFrame

logger = logging.getLogger("drone_llm.webrtc")


def _tune_encoder_bitrates() -> None:
    """Raise aiortc's built-in encoder bitrate caps (defaults are 0.5-1.5 Mbps VP8,
    which looks grainy at 720p). Constants are read at clamp time, so patching the
    module values before any peer connection is created takes effect everywhere."""
    min_bps = int(os.environ.get("WEBRTC_MIN_BITRATE", "1000000"))
    default_bps = int(os.environ.get("WEBRTC_DEFAULT_BITRATE", "3000000"))
    max_bps = int(os.environ.get("WEBRTC_MAX_BITRATE", "6000000"))
    try:
        import aiortc.codecs.vpx as vpx

        vpx.MIN_BITRATE = min_bps
        vpx.DEFAULT_BITRATE = default_bps
        vpx.MAX_BITRATE = max_bps
    except ImportError:
        pass
    try:
        import aiortc.codecs.h264 as h264

        h264.MIN_BITRATE = min_bps
        h264.DEFAULT_BITRATE = default_bps
        h264.MAX_BITRATE = max_bps
    except ImportError:
        pass
    logger.info(
        "WebRTC encoder bitrates: min=%d default=%d max=%d", min_bps, default_bps, max_bps
    )


_tune_encoder_bitrates()

_pcs: set[RTCPeerConnection] = set()


from core.webrtc_ice import ice_servers_for_peer


def warmup_camera() -> None:
    from core.shared_camera import get_camera

    cam = get_camera()
    frame = cam.get_frame()
    if frame is None:
        raise RuntimeError(f"camera opened on {cam.device_path} but no frame yet")


class SharedCameraVideoTrack(VideoStreamTrack):
    """Per-peer track. Each peer gets its own instance and its own frame copies:
    sharing one native av.VideoFrame across encoder threads (MediaRelay fan-out)
    segfaults in libvpx/PyAV. The camera itself is shared and thread-safe."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._target_fps = max(1.0, float(os.environ.get("CAMERA_FPS", "15")))
        self._max_width = max(320, int(os.environ.get("WEBRTC_MAX_WIDTH", "1280")))
        # Latched on first real frame; keeps output resolution constant for the
        # whole track lifetime (mid-stream size changes force encoder re-init).
        self._out_size: tuple[int, int] | None = None
        self._last_frame: np.ndarray | None = None

    def _fit(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._out_size is None:
            h, w = frame_bgr.shape[:2]
            if w > self._max_width:
                out_w = self._max_width
                out_h = max(2, int(h * self._max_width / w) // 2 * 2)
            else:
                out_w = w // 2 * 2
                out_h = h // 2 * 2
            self._out_size = (out_w, out_h)
        out_w, out_h = self._out_size
        if frame_bgr.shape[1] != out_w or frame_bgr.shape[0] != out_h:
            frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(frame_bgr)

    async def recv(self) -> VideoFrame:
        from core.shared_camera import get_camera

        pts, time_base = await self.next_timestamp()
        frame_bgr = await asyncio.to_thread(get_camera().get_frame)
        wait = 0
        while frame_bgr is None and self._last_frame is None and wait < 50:
            await asyncio.sleep(0.02)
            wait += 1
            frame_bgr = await asyncio.to_thread(get_camera().get_frame)

        if frame_bgr is not None:
            frame_bgr = self._fit(frame_bgr)
            self._last_frame = frame_bgr
        elif self._last_frame is not None:
            # Brief camera hiccup: repeat last frame instead of switching to a
            # differently-sized placeholder.
            frame_bgr = self._last_frame
        else:
            out_w, out_h = self._out_size or (self._max_width, self._max_width * 9 // 16)
            frame_bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        video = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
        video.pts = pts
        video.time_base = time_base
        await asyncio.sleep(max(0.0, (1.0 / self._target_fps) * 0.85))
        return video


async def close_all_peers() -> None:
    coros = [pc.close() for pc in list(_pcs)]
    _pcs.clear()
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)


async def handle_offer(sdp: str, offer_type: str) -> dict[str, str]:
    # Cap concurrent viewers; a browser reconnect leaves a zombie peer behind,
    # and each peer runs its own encoder (CPU-bound on Jetson).
    max_peers = int(os.environ.get("WEBRTC_MAX_PEERS", "2"))
    if len(_pcs) >= max_peers:
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

    # Per-peer track: no shared native frame objects between encoder threads.
    pc.addTrack(SharedCameraVideoTrack())

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
        age = None
        age_fn = getattr(cam, "frame_age_sec", None)
        if callable(age_fn):
            age = age_fn()
        stale = age is not None and age > 5.0
        return {
            "connected": frame is not None and not stale,
            "stale": stale,
            "frame_age_sec": round(age, 2) if age is not None else None,
            "device": getattr(cam, "device_path", None),
            "width": getattr(cam, "width", None),
            "height": getattr(cam, "height", None),
            "backend": backend,
        }
    except Exception as e:
        return {"connected": False, "error": str(e), "backend": backend}
