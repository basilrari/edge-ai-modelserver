# Remote internet viewing (TURN)

Signaling goes through **Cloudflare** → gateway (`https://edge-ai.basilrari.com/camera/webrtc/*`).  
**Video** cannot use the HTTP tunnel; it uses **TURN** on the Jetson public IP (`WEBRTC_PUBLIC_HOST`, campus default `140.123.105.214`).

1. Run `Drone_LLM/deploy/coturn/configure-coturn.sh` and add printed credentials to `sar-stack.env`.
2. Open firewall: UDP/TCP **3478**, UDP **49160–49252**.
3. `sar-stack.sh` starts coturn when `COTURN_ENABLE=1`, GoPro USB preview when `GOPRO_ENABLE=1` and `CAMERA_BACKEND=gopro`, then Drone_LLM with the same camera backend.
4. Browser fetches `GET /camera/webrtc/ice` and uses `iceTransportPolicy: relay`.

**One-command stack:** `./sar-stack.sh start` — tmux windows include `gopro` (UDP :8554) and `model` (uvicorn). Production UI: https://edge-ai-frontend-mauve.vercel.app/camera

See [deploy/coturn/README.md](deploy/coturn/README.md).
