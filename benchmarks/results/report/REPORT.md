# Drone Video Benchmark Report

## Run summary
- Frames analyzed: **68**
- Videos: archive_flood_airfield.mp4
- Modes: flood, combined
- Simulated altitudes (m): [25.0, 50.0, 75.0, 100.0]
- Mean latency: **26.4 ms**
- Mean instant FPS: **51.31**

## Altitude / coverage trade-offs

| Altitude (m) | Coverage factor (×) | Mean latency (ms) | Mean flood ratio | Mean FPS |
|-------------|---------------------|-------------------|------------------|----------|
| 25 | 0.50 | 26.2 | 0.581 | 51.67 |
| 50 | 1.00 | 26.6 | 0.431 | 58.45 |
| 75 | 1.50 | 21.5 | 0.424 | 56.65 |
| 100 | 2.00 | 31.1 | 0.509 | 36.67 |

Higher altitude → **wider area per frame** but **coarser pixels** (flood/human details shrink).

## live_pipeline.py metrics (included in CSV)

| Metric | Description |
|--------|-------------|
| clf_load_ms / seg_load_ms | One-time model load at session start |
| classification_ms / segmentation_ms | Per-frame GPU forward pass |
| model_switch_latency_ms | clf + seg latency (combined path) |
| total_latency_ms / total_inference_ms | End-to-end inference time |
| fps / instant_fps | Throughput |
| memory_mb / cpu_percent / power_w | Resource use |
| flood_ratio | DeepLab flood pixel fraction |

## Model switch & load/unload

- **flood_classifier**: load 117 ms, unload 202 ms, reload 53 ms
- **flood_segmenter**: load 46 ms, unload 198 ms, reload 65 ms
- **human_detector**: load 4 ms, unload 195 ms, reload 1 ms
- **Logical primary switch** (ResNet↔DeepLab): 36 events, mean 0.017 ms orchestration
- **Physical swap** (unload clf + load seg): 248.65 ms

## Plots

See PNG files in this folder.
