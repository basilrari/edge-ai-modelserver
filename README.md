# Drone_LLM — Flood Detection Model Server

Adaptive flood detection API for Jetson with live camera inference, ResNet18 classification, and DeepLabv3+ segmentation.

## Features

- FastAPI server with live dashboard (`/`) and flood detection API (`POST /detect_flood`)
- Context-aware model switching between ResNet18 and DeepLabv3+
- 4×4 grid localization overlay when segmentation is active (flood ratio ≥ 0.2)
- GPU inference on CUDA (Jetson Orin)
- Power metrics via tegrastats

## Quick start

```bash
cd model_server
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/

## API

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /` | Live dashboard |
| `POST /detect_flood` | Run inference on camera frame |
| `WS /ws/live` | WebSocket stream |

## Camera

Set external USB camera device if needed:

```bash
export CAMERA_DEVICE=/dev/video0
```

## Model weights

Included in repo:
- `models/flood_classifier/flood_resnet18.pth`
- `models/flood_segmentation/DeepLabv3_plus/flood_segmentation/best_model.pth`

Training datasets are excluded (see `.gitignore`).
