"""ICE / TURN settings (must match gateway WEBRTC_* env)."""

from __future__ import annotations

import os

from aiortc import RTCIceServer


def _split_urls(raw: str) -> list[str]:
    return [u.strip() for u in raw.split(",") if u.strip()]


def ice_servers_for_peer() -> list[RTCIceServer]:
    servers: list[RTCIceServer] = []
    stun = os.environ.get("WEBRTC_STUN_URL", "stun:stun.l.google.com:19302").strip()
    if stun:
        servers.append(RTCIceServer(urls=[stun]))

    urls: list[str] = []
    server_turn = os.environ.get("WEBRTC_TURN_URL_SERVER", "").strip()
    if server_turn:
        urls.extend(_split_urls(server_turn))
    elif os.environ.get("WEBRTC_TURN_URLS", "").strip():
        urls.extend(_split_urls(os.environ.get("WEBRTC_TURN_URLS", "")))
    elif os.environ.get("WEBRTC_TURN_URL", "").strip():
        urls.append(os.environ.get("WEBRTC_TURN_URL", "").strip())
    elif os.environ.get("WEBRTC_PUBLIC_HOST", "").strip():
        # Local coturn on the Jetson — avoid NAT hairpin to public IP.
        urls.extend(
            [
                "turn:127.0.0.1:3478?transport=udp",
                "turn:127.0.0.1:3478?transport=tcp",
            ]
        )

    user = os.environ.get("WEBRTC_TURN_USERNAME", "")
    cred = os.environ.get("WEBRTC_TURN_PASSWORD", "")
    for url in urls:
        if user:
            servers.append(RTCIceServer(urls=[url], username=user, credential=cred))
        else:
            servers.append(RTCIceServer(urls=[url]))
    return servers
