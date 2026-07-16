# Model Server — Plain English Handout (1 page)

**Project:** Context-Aware Adaptive AI for Flood-Rescue UAVs  
**Platform:** NVIDIA Jetson Orin · **My role:** Model Server (vision + AI only)

---

## What is it?

The **Model Server** is the **“eyes and analyst”** on the drone’s onboard computer. It watches video from a camera, runs AI to detect **flood water** and **people**, and sends back **results + annotated pictures + speed/power numbers**.

It does **not** fly the drone. Flying is handled separately by the Drone Server and flight controller.

---

## Who does what in the full system?

| Role | Simple job |
|------|------------|
| **User** | Says what they want (“scan for flood and people”) |
| **Gateway** | Routes messages between user, AI, and services |
| **LLM** | Understands plain English → structured command |
| **Model Server (mine)** | Camera + AI → detections, GPS estimates, metrics |
| **Rule Gate** | Safety check before flight commands |
| **Drone Server** | Actually controls the drone |

---

## What can it do today?

- **Flood detection** — fast check (ResNet18) + detailed water map (DeepLabv3+); switches primary model when flood ratio ≥ 20%
- **Human detection** — **two detectors**: fast YOLOv8n (patrol) or **YOLO11s VisDrone** (robust, 1280px) for tiny aerial humans; auto-picks based on altitude, visibility, flood level, and mission priority
- **Combined rescue mode** — flood + people on the **same frame**; flood level can trigger the stronger human model
- **GPS estimates** — most flooded grid cell + each person’s location *(simulated ref — no Pixhawk yet)*
- **Live dashboard** — video, metrics, power, WebSocket stream
- **Offline video testing** — upload MP4 → CSV, plots, overlay video, report

---

## How does one frame get processed? (10 steps)

1. **Command arrives** — Gateway or dashboard says: start flood / human / both / stop  
2. **Task Session** — turns on the right mode; loads AI models into GPU (warmup)  
3. **Camera** — grabs one snapshot (live USB camera or offline video file)  
4. **Context check** — CPU, memory, power, simulated drone altitude/priority/visibility, and **live flood ratio** from the last frame  
5. **Smart choice** — skip slow flood map when scene is dry; pick **lightweight vs robust** human model (YOLOv8n ↔ YOLO11s) from context  
6. **AI runs** — classify → segment (if needed) → detect humans with the selected YOLO tier (if human/combined)  
7. **Overlays** — flood mask, 4×4 grid, red dot on worst cell, person boxes  
8. **GPS math** — estimate lat/lon from camera + fake drone reference point  
9. **Metrics** — FPS, latency (~30 ms steady), memory, Jetson power  
10. **Send back** — JSON + annotated JPEG to Gateway or browser  

---

## Key results (offline benchmark — *chittur_river_rescue*)

Best demo clip: **people + flood together** (real rescue scenario).

| Metric | Value |
|--------|-------|
| Mode | Combined (flood + human) |
| Steady speed | **~34 FPS** |
| Steady latency (p95) | **~30 ms** |
| Mean flood ratio | **57%** (consistently flooded) |
| Mean humans per frame | **~1.7** (1–3 people) |
| Memory | **~1.2 GB** stable (no leak) |

*First frame after startup is slower (~400 ms) due to model load — warm up before flight.*

---

## What is real vs simulated?

| Real today | Simulated / planned |
|------------|---------------------|
| AI on camera/video | Drone battery, altitude from Pixhawk |
| Speed, FPS, latency | Wind, environment sensors |
| CPU, memory, Jetson power (live) | GPS uses fixed demo reference |
| Dashboard + API for Gateway | Heavier trained models (future) |

---

## Why this design?

- **Separation:** LLM decides *what* to look for; Model Server decides *how* to run AI; Drone Server flies.  
- **Edge-first:** Everything runs on Jetson — works without cloud during disasters.  
- **Adaptive:** Skip expensive steps when scene is dry or computer is stressed.  
- **Honest testing:** Lightweight models validate the **pipeline**; accuracy improves with better training later.

---

**Repo:** github.com/aykumar21/Drone_LLM  
**Contact / author:** [your name · institution]
