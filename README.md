# Drone_LLM — Model Server

LLM-driven vision API for Jetson Orin: adaptive flood detection, human detection, and combined rescue-mode inference from a live USB camera.

The server starts **idle** and only runs inference when an external LLM (or you via `curl`) activates a tool.

## Features

- **LLM tool activation** — `detect_flood`, `detect_human`, or both at once
- **Adaptive flood pipeline** — ResNet18 classifier + DeepLabv3+ segmenter with context-aware model switching
- **Flood threshold** — `Flooded` status and 4×4 grid overlay when flood ratio ≥ 0.2
- **Human detection** — YOLOv8n person bounding boxes
- **Combined mode** — one camera frame → flood grid + human boxes (e.g. person stuck in flood)
- **Live dashboard** — task status, metrics, power, and annotated video at `/`
- **WebSocket stream** — continuous inference while a tool is active (`WS /ws/live`)
- **Power metrics** — idle / inference / extra power via tegrastats

## Quick start

```bash
cd model_server
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/

Optional camera device:

```bash
export CAMERA_DEVICE=/dev/video0
```

## LLM / API usage

### Activate tools

| Goal | Command |
|------|---------|
| Flood only | `curl -X POST http://localhost:8000/tool -H "Content-Type: application/json" -d '{"tool":"detect_flood"}'` |
| Human only | `curl -X POST http://localhost:8000/tool -H "Content-Type: application/json" -d '{"tool":"detect_human"}'` |
| Both (rescue) | `curl -X POST http://localhost:8000/tool -H "Content-Type: application/json" -d '{"tools":["detect_flood","detect_human"]}'` |
| Stop | `curl -X POST http://localhost:8000/stop` |
| Status | `curl http://localhost:8000/status` |

Shorthand endpoints:

```bash
curl -X POST http://localhost:8000/detect_flood
curl -X POST http://localhost:8000/detect_human
curl -X POST http://localhost:8000/detect_combined
```

Combined mode also accepts:

```json
{"tool": "detect_flood_and_human"}
{"tool": ["detect_flood", "detect_human"]}
```

Aliases: `detect_combined`, `rescue`, `flood_and_human`

### Status response

```json
{
  "active_tool": "detect_combined",
  "active_tools": ["detect_flood", "detect_human"],
  "inference_enabled": true
}
```

### Important for LLM orchestration

Send **both tools in one request** for rescue scenarios. Calling `detect_flood` and then `detect_human` separately will switch modes instead of running both together.

## API reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live dashboard |
| `/health` | GET | Health + active task |
| `/status` | GET | Current active tool(s) |
| `/tool` | POST | Activate tool(s) or stop (`{"tool":"idle"}`) |
| `/detect_flood` | POST | Run flood detection (activates flood mode) |
| `/detect_human` | POST | Run human detection (activates human mode) |
| `/detect_combined` | POST | Run flood + human on same frame |
| `/stop` | POST | Stop all detection |
| `/ws/live` | WS | Stream inference while active |

Each POST returns JSON with metrics and a base64 JPEG frame (`frame_base64`).

## Architecture

```
External LLM  →  POST /tool  →  TaskSession  →  detect_flood | detect_human | detect_combined
                                                      ↓
                                              shared camera (one frame)
                                                      ↓
                              ResNet18 + DeepLabv3+  |  YOLOv8n (combined: composite overlay)
```

Key modules:

| Path | Role |
|------|------|
| `main.py` | FastAPI app, routes, WebSocket |
| `core/task_session.py` | Active tool state (single or combined) |
| `tools/detect_flood.py` | Flood classifier + segmenter + grid |
| `tools/detect_human.py` | YOLOv8n person detection |
| `tools/detect_combined.py` | Same-frame flood + human pipeline |
| `core/shared_camera.py` | Shared V4L2 camera for all tools |
| `core/model_selector.py` | ResNet ↔ DeepLab switching (ratio ≥ 0.2) |
| `core/flood_grid.py` | 4×4 grid overlay + GPS localization |
| `core/power_monitor.py` | tegrastats power sampling |

## Standalone human detection test

```bash
python3 tools/human_detection_live.py --frames 50 --conf 0.35
```

## Model weights

Included in repo:

- `yolov8n.pt` — human detection
- `models/flood_classifier/flood_resnet18.pth` — flood classifier
- `models/flood_segmentation/DeepLabv3_plus/flood_segmentation/best_model.pth` — flood segmenter

Training datasets are excluded (see `.gitignore`).

## Hardware

- Tested on Jetson Orin with CUDA
- External USB camera (default `/dev/video0`)

## Repository

https://github.com/aykumar21/Drone_LLM
