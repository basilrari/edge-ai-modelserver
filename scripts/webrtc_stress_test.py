import asyncio

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription


async def test() -> None:
    for i in range(5):
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://127.0.0.1:8000/camera/webrtc/offer",
                json={
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                },
            ) as resp:
                body = await resp.json()
        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=body["sdp"], type=body["type"])
        )
        for _ in range(40):
            if pc.connectionState == "connected":
                break
            await asyncio.sleep(0.2)
        print(f"round {i + 1} state={pc.connectionState}")
        await pc.close()
        await asyncio.sleep(1.0)


if __name__ == "__main__":
    asyncio.run(test())
