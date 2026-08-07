# Remote WebRTC (internet viewers)

Cloudflare tunnel carries **signaling** (`/camera/webrtc/offer`) only. **Video** uses UDP/TCP TURN on the Jetson **public IP** (campus: see `drone-competition/IP.txt`).

## One-time setup

```bash
cd Drone_LLM/deploy/coturn
chmod +x configure-coturn.sh run-coturn.sh
export WEBRTC_PUBLIC_HOST=140.123.105.214   # your Jetson public IP
./configure-coturn.sh
```

Copy the printed `WEBRTC_TURN_USERNAME`, `WEBRTC_TURN_PASSWORD`, and `WEBRTC_TURN_URLS` into `Code/sar-stack.env`.

Install coturn if needed:

```bash
sudo apt install -y coturn
```

**Firewall / router** must allow inbound to the Jetson:

- UDP/TCP **3478** (TURN)
- UDP **49160–49252** (relay ports, match `WEBRTC_TURN_MIN_PORT` / `MAX`)

## Run with SAR stack

`sar-stack.sh` starts coturn when `COTURN_ENABLE=1` (default). Or manually:

```bash
./run-coturn.sh
```

Gateway exposes `GET /camera/webrtc/ice` with STUN + TURN credentials; the browser uses `iceTransportPolicy: relay` when TURN is configured.

## Env (sar-stack.env)

```bash
WEBRTC_PUBLIC_HOST=140.123.105.214
WEBRTC_TURN_URLS=turn:140.123.105.214:3478?transport=udp,turn:140.123.105.214:3478?transport=tcp
WEBRTC_TURN_USERNAME=webrtc
WEBRTC_TURN_PASSWORD=<from configure-coturn.sh>
WEBRTC_ICE_TRANSPORT_POLICY=relay
WEBRTC_TURN_URL_SERVER=turn:127.0.0.1:3478?transport=udp,turn:127.0.0.1:3478?transport=tcp
```

Drone_LLM uses `WEBRTC_TURN_URL_SERVER` so the encoder talks to local coturn; browsers use the public `WEBRTC_TURN_URLS`.
