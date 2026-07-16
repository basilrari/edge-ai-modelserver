# Drone_LLM — Model Server

LLM-driven vision API for Jetson Orin: adaptive flood detection, human detection, and combined rescue-mode inference from a live USB camera.

The server starts **idle** and only runs inference when an external LLM (or you via `curl`) activates a tool.

## Features

- **LLM tool activation** — `detect_flood`, `detect_human`, or both at once
- **Adaptive flood pipeline** — ResNet18 classifier + DeepLabv3+ segmenter with context-aware model switching
- **Flood threshold** — `Flooded` status and 4×4 grid overlay when flood ratio ≥ 0.2
- **Human detection — dual tier** — **YOLOv8n** (lightweight patrol) or **YOLO11s VisDrone** (robust aerial small-human) with automatic context-aware switching
- **Combined mode** — one camera frame → flood grid + human boxes (e.g. person stuck in flood); flood ratio feeds human tier selection
- **Live dashboard** — task status, metrics, power, and annotated video at `/`
- **WebSocket stream** — continuous inference while a tool is active (`WS /ws/live`)
- **Power metrics** — idle / inference / extra power via tegrastats (background sampling)
- **Performance** — parallel combined inference, TensorRT engines included in repo, CUDA streams
- **Benchmark artifacts** — per-clip metrics CSVs, latency/FPS plots, and HTML reports committed under `benchmarks/results/`

## Performance tuning (Jetson)

Combined flood + human runs **in parallel** on separate CUDA streams by default.

**TensorRT engines ship with this repo** — clone and run with `USE_TENSORRT=1` (default). Re-export only if you retrain weights or move to a different Jetson/CUDA build:

```bash
# Re-export ALL TensorRT engines (optional — only if missing or stale)
bash tools/install_export_deps.sh
python3 tools/export_tensorrt.py
# Or flood models only (~10–20 min):
python3 tools/export_flood_tensorrt.py

# Re-export robust human only (~5–15 min):
python3 tools/export_robust_human.py
```

Pre-built artifacts in repo (PyTorch + ONNX + TensorRT):

| Model | `.pt` / `.pth` | `.onnx` | `.engine` |
|-------|----------------|---------|-----------|
| YOLOv8n (lightweight human) | `yolov8n.pt` | `yolov8n.onnx` | `yolov8n.engine` |
| YOLO11s VisDrone (robust human) | `models/human_detector/yolo11s_visdrone_human_1280.pt` | same dir `.onnx` | same dir `.engine` |
| ResNet18 flood classifier | `models/flood_classifier/flood_resnet18.pth` | `.onnx` | `.engine` |
| DeepLabv3+ flood segmenter | `models/flood_segmentation/.../best_model.pth` | `flood_deeplab.onnx` | `flood_deeplab.engine` |

# Optional env toggles (defaults shown)
export PARALLEL_COMBINED=1    # overlap flood + human GPU work
export ASYNC_POWER=1          # non-blocking tegrastats
export USE_TENSORRT=1         # load *.engine files when present
export SMART_SEGMENT=1          # skip DeepLab when ResNet says dry (~3s saved)
export SEG_INTERVAL=8           # occasional full segment refresh
export YOLO_IMGSZ=320           # lightweight human input size
export YOLO_ROBUST_IMGSZ=1280   # robust YOLO11s input size
export HUMAN_DETECTOR_TIER=lightweight  # env default before context kicks in
export USE_TORCH_COMPILE=0    # set 1 to try torch.compile on flood models
export CUDNN_BENCHMARK=1
```

Restart the server after exporting TensorRT engines.


```bash
cd model_server
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/

### Gateway (edge-ai-gateway)

The dashboard sends natural-language commands through the model server to **edge-ai-gateway** (default `http://127.0.0.1:3000`). Gateway LLM tool names are mapped to model-server tools without renaming them:

| Gateway (`category: model`) | Model server |
|----------------------------|--------------|
| `human_detect` | `detect_human` |
| `flood_seg`, `flood_class` | `detect_flood` |
| both flood + human in one plan | `detect_combined` |

```bash
# Start gateway (separate terminal)
cd ~/edge-ai-gateway-master && cargo run

# Optional override
export GATEWAY_URL=http://127.0.0.1:3000
```

Dashboard: type a prompt in **LLM command** and press Send. **Quick presets** still call `/tool` directly (no gateway).

<img width="624" height="687" alt="dashboard" src="https://github.com/user-attachments/assets/7e250fbe-1a91-44c0-a77e-7d059812b759" />


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

---

## Human detection — YOLO11s robust tier

The server runs **two human detectors** and picks between them every frame based on mission context (or a manual dashboard override).

| Tier | Model | Input size | Classes | Best for |
|------|-------|------------|---------|----------|
| **lightweight** | YOLOv8n (COCO person) | 320×320 | `person` (id 0) | Fast patrol, low power, clear scenes |
| **robust** | YOLO11s VisDrone-trained | 1280×1280 | `pedestrian` + `people` (ids 0, 1) | High altitude, low visibility, flood rescue, small/distant humans |

### Why YOLO11s VisDrone?

Standard COCO YOLOv8n is tuned for ground-level video. SAR drones see humans from above at long range — often only a few pixels tall. The **robust** tier uses a **YOLO11s** checkpoint fine-tuned on **VisDrone** aerial imagery at **1280px** input, which dramatically improves recall on small humans in flood/rescue footage.

**Weights and TensorRT engines are included in the repo** (see [Model weights](#model-weights)). To re-export after a Jetson OS upgrade:

```bash
python3 tools/export_robust_human.py
```

If robust weights are missing locally, the server **falls back to YOLOv8n** automatically.

### Human tier API & dashboard

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/human/detector` | GET | Current tier, mode (`auto` / `forced`), last switch |
| `/human/detector-tier` | POST | Force tier or return to auto |

```bash
# Force robust (YOLO11s) — stays until you reset
curl -X POST http://localhost:8000/human/detector-tier \
  -H "Content-Type: application/json" \
  -d '{"tier":"robust","force":true}'

# Force lightweight (YOLOv8n)
curl -X POST http://localhost:8000/human/detector-tier \
  -H "Content-Type: application/json" \
  -d '{"tier":"lightweight","force":true}'

# Back to context-aware auto switching
curl -X POST http://localhost:8000/human/detector-tier \
  -H "Content-Type: application/json" \
  -d '{"mode":"auto"}'
```

On the dashboard **Human detection** panel: **Force lightweight**, **Force robust**, or **Auto (context)**. Tier switches unload the previous weights and reload the new detector (TensorRT engine preferred when `USE_TENSORRT=1`).

<img width="1880" height="937" alt="new dashboard" src="https://github.com/user-attachments/assets/5d05bccd-f5ae-41e0-addb-2a589b10905a" />

---

## Context-aware adaptive switching (all models)

Every inference frame builds a **context dictionary** (`core/context_evaluator.py`) and passes it to the flood and human selectors. Switching is **per-frame** while a tool is active; when you stop (`idle`), selectors reset.

### Context signals

| Signal | Source | Used by |
|--------|--------|---------|
| `cpu_usage`, `memory_usage` | `psutil` (real) | Flood segmenter gate, human tier |
| `gpu_usage` | `/sys/devices/gpu.0/load` (real) | Telemetry |
| `battery`, `altitude`, `speed`, `gps_signal_strength` | Simulated UAV (placeholder until flight controller) | Human tier |
| `visibility`, `wind`, `light`, `rain`, `temperature` | Simulated environment | Human tier |
| `priority`, `mission` | Simulated mission | Human tier |
| `flood_ratio` | **Live** from last flood/segment inference | Flood primary switch, human tier, combined mode |
| `classification_label` | ResNet18 output | Flood metadata |

In **combined mode**, flood runs first (or in parallel); its `flood_ratio` is injected into the human context so a flooded scene can escalate to YOLO11s even before the human path finishes.

### Flood pipeline — ResNet18 + DeepLabv3+

**Modules:** `core/model_selector.py`, `core/segment_policy.py`, `tools/detect_flood.py`

Both flood models stay in the design; switching controls **which model leads the decision** and **whether DeepLab runs this frame**.

```
                    ┌─────────────────┐
  camera frame ───► │  ResNet18 clf   │  (~fast, every frame)
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │  SMART_SEGMENT policy       │
              │  run DeepLab this frame?    │
              └──────────────┬──────────────┘
                             ▼
                    ┌─────────────────┐
                    │ DeepLabv3+ seg  │  (~slow, skipped when dry)
                    └────────┬────────┘
                             ▼
                    flood_ratio → context → primary switch
```

**Primary model switch** (`FloodModelSelector`):

| Condition | Primary model | UI effect |
|-----------|---------------|-----------|
| `flood_ratio < 0.20` | **ResNet18** | Label `Non-Flooded`, no grid |
| `flood_ratio ≥ 0.20` | **DeepLabv3+** | Label `Flooded`, 4×4 grid + GPS cells |

**Segmenter gating** (skip expensive DeepLab when safe):

- **Off** if `battery < 15%` or `cpu_usage > 90%`
- **On** if ResNet predicts flood class (`Flood`)
- **On** if `last_flood_ratio ≥ SEG_HYSTERESIS` (default 0.12) — brief hold after flood drops
- **On** every `SEG_INTERVAL` frames (default 8) for periodic refresh
- **On** frame 1 always (cold start)
- Set `SMART_SEGMENT=0` to run DeepLab every frame (debug only)

Status severity from ratio when flooded: `WARNING` (≥0.2), `ALERT` (≥0.4), `CRITICAL` (≥0.7).

### Human pipeline — lightweight ↔ robust

**Modules:** `core/human_model_selector.py`, `core/human_detector_tier.py`, `tools/detect_human.py`

Decision order in `HumanModelSelector._choose_tier()`:

1. **Force lightweight** if `battery < 20%` OR `cpu_usage > 90%`
2. **Force robust** if any of:
   - `priority ≥ 0.9` (high mission priority)
   - `altitude ≥ 60 m`
   - `visibility < 0.4` (low visibility)
   - `flood_ratio ≥ 0.20` (person-in-flood rescue)
   - `human_escalate` flag set
   - `priority ≥ 0.7` AND `altitude ≥ 45 m` (borderline SAR)
3. **Hysteresis:** stay on robust if already robust and `flood_ratio ≥ 0.12`
4. Otherwise → **lightweight** (YOLOv8n @ 320)

When tier changes, `human_detector_tier` unloads the cached YOLO weights and reloads the correct checkpoint/engine before inference.

### Combined mode — cross-model context

**Module:** `tools/detect_combined.py`

With `PARALLEL_COMBINED=1` (default), flood and human inference overlap on separate CUDA streams. Human context includes `flood_ratio` from the flood session so tier selection can escalate to YOLO11s during active flooding.

Typical rescue flow:

1. LLM or dashboard activates `detect_combined`
2. ResNet classifies → DeepLab segments if flood suspected → `flood_ratio` updated
3. Human selector sees `flood_ratio ≥ 0.2` → switches to **YOLO11s @ 1280**
4. Overlay: flood grid + human boxes + per-person GPS on one JPEG

### Response fields for switching

Flood payloads include `model_switches`, `primary_model`, `selection_metadata`, and full `context`.

Human payloads include `human_detector`, `active_models.tier`, `active_models.mode` (`auto` / `forced`), and `model_switches.tier` when a swap occurred.

Example human switch log:

```
[HUMAN MODEL SWITCH] lightweight → robust
[HUMAN TIER] lightweight → robust (YOLOv8n → YOLO11s VisDrone) [auto]
```

---

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
| `/gateway/status` | GET | Proxy: edge-ai-gateway health + `active_command` |
| `/gateway/infer` | POST | Proxy: LLM infer → map tools → activate model server (`{"prompt":"..."}`) |
| `/human/detector` | GET | Human tier status (auto/forced, active detector, last switch) |
| `/human/detector-tier` | POST | Force `lightweight` / `robust` or `{"mode":"auto"}` |
| `/ws/live` | WS | Stream inference while active |

Each POST returns JSON with metrics and a base64 JPEG frame (`frame_base64`).

## Architecture

```
External LLM / dashboard
        │
        ▼
  POST /tool  or  POST /gateway/infer
        │
        ▼
   TaskSession (idle | detect_flood | detect_human | detect_combined)
        │
        ▼
   shared camera (one frame per tick)
        │
        ├──────────────────────────────┬─────────────────────────────┐
        ▼                              ▼                             │
 ContextEvaluator.get_context()   flood_ratio (session)               │
        │                              │                             │
        ▼                              ▼                             │
 FloodModelSelector              HumanModelSelector                   │
 ResNet18 + DeepLabv3+           YOLOv8n  ↔  YOLO11s VisDrone         │
 (primary + smart segment)       (lightweight ↔ robust)               │
        │                              │                             │
        └──────── detect_combined ─────┘                             │
                     composite overlay (grid + human boxes + GPS) ◄──┘
```

Key modules:

| Path | Role |
|------|------|
| `main.py` | FastAPI app, routes, WebSocket, gateway proxy |
| `core/task_session.py` | Active tool state (single or combined) |
| `core/context_evaluator.py` | Per-frame context (system + mission signals) |
| `core/model_selector.py` | Flood primary switch ResNet18 ↔ DeepLab (ratio ≥ 0.2) |
| `core/segment_policy.py` | Skip/run DeepLab based on classifier + hysteresis |
| `core/human_model_selector.py` | Human tier rules (altitude, flood, visibility, …) |
| `core/human_detector_tier.py` | Runtime tier state, force override, weight swap |
| `tools/detect_flood.py` | Flood classifier + segmenter + grid |
| `tools/detect_human.py` | Context-aware YOLOv8n or YOLO11s |
| `tools/detect_combined.py` | Same-frame flood + human; shares `flood_ratio` |
| `core/shared_camera.py` | Shared V4L2 camera for all tools |
| `core/flood_grid.py` | 4×4 grid overlay + GPS localization |
| `core/power_monitor.py` | tegrastats power sampling |
| `core/gateway_client.py` | HTTP proxy to edge-ai-gateway + LLM health |
| `core/gateway_tools.py` | Gateway tool name mapping + keyword fallback |

## Standalone human detection test

```bash
python3 tools/human_detection_live.py --frames 50 --conf 0.35
```

## Model weights

All deployed inference weights are **committed to this repository** (PyTorch checkpoints, ONNX exports, and Jetson TensorRT `.engine` files).

| Role | PyTorch | ONNX | TensorRT |
|------|---------|------|----------|
| Lightweight human (YOLOv8n) | `yolov8n.pt` | `yolov8n.onnx` | `yolov8n.engine` |
| Robust human (YOLO11s VisDrone @ 1280) | `models/human_detector/yolo11s_visdrone_human_1280.pt` | `.../yolo11s_visdrone_human_1280.onnx` | `.../yolo11s_visdrone_human_1280.engine` |
| Flood classifier (ResNet18) | `models/flood_classifier/flood_resnet18.pth` | `flood_resnet18.onnx` | `flood_resnet18.engine` |
| Flood segmenter (DeepLabv3+) | `models/flood_segmentation/DeepLabv3_plus/flood_segmentation/best_model.pth` | `flood_deeplab.onnx` | `flood_deeplab.engine` |
| Legacy U-Net | `models/flood_segmentation/U-Net/unet_model.pth` | — | — |

`ModelManager` prefers `.engine` when `USE_TENSORRT=1` and the file exists; otherwise it loads `.pt` / `.pth`.

**Not in repo** (too large — download or generate locally):

- Training datasets (`models/flood_classifier/dataset/`, segmentation `JPEGImages/`, etc.)
- Benchmark source/overlay videos (`benchmarks/videos/`, `benchmarks/results/videos/`, `*_overlay.mp4`)

---

## Benchmark results & performance plots

Pre-computed evaluation artifacts are committed so you can review Jetson performance without re-running benchmarks.

### Aggregate report (all clips)

| Path | Contents |
|------|----------|
| `benchmarks/results/report/report.html` | Cross-clip HTML report (open in browser) |
| `benchmarks/results/report/REPORT.md` | Markdown summary |
| `benchmarks/results/report/summary.json` | Mean latency, FPS, flood ratio |
| `benchmarks/results/report/*.png` | Latency vs altitude, FPS, flood ratio, model-switch events, load/unload |
| `benchmarks/results/drone_video_benchmark.csv` | Per-run benchmark table |
| `benchmarks/results/system_benchmark.json` | System-level benchmark snapshot |

### Per-video dashboard exports (`benchmarks/results/dashboard/<clip_id>/`)

Each clip folder includes:

- `metrics.csv` — per-frame latency, FPS, flood ratio, human count, CPU, memory, power
- `summary.json` / `REPORT.md` / `report.html`
- Plots: `series_*.png`, `overview_per_frame.png`, `flood_ratio_by_altitude.png`, `human_count_over_time.png`, `tradeoff_altitude_latency.png`, `model_load_times.png`

Example clips with full plot sets: `chittur_river_rescue`, `youtube_cumbria_flood_rescue_828618`, `archive_flood_airfield_274c83`, `kherson_drone_rescue`.

### Live-server runtime plots (`logs/`)

CUDA profiling outputs from on-device runs:

- `logs/metrics.csv`, `logs/metrics_cuda.csv`
- `logs/*_plot.png`, `logs/*_plot_cuda.png` — classification, segmentation, latency, FPS, CPU, memory, power, model-switch timing
- `logs/cuda_summary.json`

Regenerate benchmarks: see `benchmarks/README.md` and `tools/run_drone_video_benchmark.py`.

---

## Hardware

- Tested on Jetson Orin with CUDA
- External USB camera (default `/dev/video0`)

## Repository

https://github.com/aykumar21/Drone_LLM
