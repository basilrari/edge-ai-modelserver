# Remote WebRTC (internet viewers)

Signaling uses HTTP (`POST /camera/webrtc/offer` on the model server, or a gateway proxy in front of it).
**Video** usually needs a TURN server on the Jetson public IP when viewers are off the local LAN.

## One-time setup

```bash
cd Drone_LLM/deploy/coturn
chmod +x configure-coturn.sh run-coturn.sh
export WEBRTC_PUBLIC_HOST=<your-jetson-public-ip>
./configure-coturn.sh
```

Copy the printed `WEBRTC_TURN_USERNAME`, `WEBRTC_TURN_PASSWORD`, and `WEBRTC_TURN_URLS` into your environment (`.env` or shell profile).

Install coturn if needed:

```bash
sudo apt install -y coturn
```

**Firewall / router** must allow inbound to the Jetson:

- UDP/TCP **3478** (TURN)
- UDP **49160–49252** (relay ports; match `WEBRTC_TURN_MIN_PORT` / `MAX`)

## Run coturn

```bash
./run-coturn.sh
```

When TURN is configured, browsers can use `iceTransportPolicy: relay` for reliable NAT traversal.
The model server prefers `WEBRTC_TURN_URL_SERVER` (local `127.0.0.1:3478`) for the encoder side to avoid hairpin NAT.

## Example env

```bash
WEBRTC_PUBLIC_HOST=<jetson-public-ip>
WEBRTC_TURN_URLS=turn:<jetson-public-ip>:3478?transport=udp,turn:<jetson-public-ip>:3478?transport=tcp
WEBRTC_TURN_USERNAME=webrtc
WEBRTC_TURN_PASSWORD=<from configure-coturn.sh>
WEBRTC_ICE_TRANSPORT_POLICY=relay
WEBRTC_TURN_URL_SERVER=turn:127.0.0.1:3478?transport=udp,turn:127.0.0.1:3478?transport=tcp
```
