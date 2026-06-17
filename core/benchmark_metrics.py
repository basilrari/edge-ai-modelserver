"""CSV schema for offline drone-video benchmarks (extends live_pipeline metrics)."""

import csv
import os
from pathlib import Path

# Same core metrics as live_pipeline / metrics_cuda.py
CORE_METRIC_KEYS = [
    "fps",
    "instant_fps",
    "total_inference_ms",
    "total_latency_ms",
    "classification_ms",
    "segmentation_ms",
    "model_switch_latency_ms",
    "memory_mb",
    "cpu_percent",
    "power_w",
    "peak_memory_mb",
    "peak_cpu_percent",
    "peak_power_w",
    "clf_load_ms",
    "seg_load_ms",
    "human_load_ms",
    "model_load_total_ms",
    "flood_ratio",
    "warmup",
]

VIDEO_BENCHMARK_COLUMNS = [
    "timestamp",
    "video_id",
    "video_name",
    "frame_idx",
    "sim_altitude_m",
    "input_width",
    "input_height",
    "mode",
    "active_tool",
    "classification_label",
    "primary_model",
    "model_switch_occurred",
    "segmentation_skipped",
    "human_count",
    "clf_backend",
    "seg_backend",
    "human_backend",
    *CORE_METRIC_KEYS,
    "session_first_frame",
]

NUMERIC_BENCHMARK_COLUMNS = [
    "frame_idx",
    "sim_altitude_m",
    "input_width",
    "input_height",
    "human_count",
    *[
        k
        for k in CORE_METRIC_KEYS
        if k not in ("warmup",)
    ],
]


def ensure_benchmark_csv(path: Path, *, reset: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset or not path.exists() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(VIDEO_BENCHMARK_COLUMNS)


def append_benchmark_row(path: Path, row: dict) -> None:
    ensure_benchmark_csv(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([row.get(col) for col in VIDEO_BENCHMARK_COLUMNS])
