"""Stress test /camera/webrtc/offer: sequential rounds plus concurrent peers.

The concurrent phase reproduces the browser-reconnect scenario (two encoders
running at once) that previously segfaulted the server.
"""

import asyncio

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription

URL = "http://127.0.0.1:8000/camera/webrtc/offer"


async def connect_peer(hold_sec: float) -> str:
    pc = RTCPeerConnection()
    pc.addTransceiver("video", direction="recvonly")
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            URL,
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        ) as resp:
            body = await resp.json()
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=body["sdp"], type=body["type"])
    )
    for _ in range(40):
        if pc.connectionState == "connected":
            break
        await asyncio.sleep(0.2)
    state = pc.connectionState
    if state == "connected":
        await asyncio.sleep(hold_sec)
    await pc.close()
    return state


async def main() -> None:
    print("-- sequential rounds --")
    for i in range(3):
        state = await connect_peer(hold_sec=2.0)
        print(f"round {i + 1} state={state}")
        await asyncio.sleep(0.5)

    print("-- concurrent peers (reconnect scenario) --")
    for i in range(3):
        results = await asyncio.gather(
            connect_peer(hold_sec=5.0),
            connect_peer(hold_sec=5.0),
            return_exceptions=True,
        )
        print(f"concurrent round {i + 1}: {results}")
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(main())
