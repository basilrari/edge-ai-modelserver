import csv
import os

import pandas as pd

METRICS_CUDA_COLUMNS = [
    "timestamp",
    "fps",
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
    "flood_ratio",
    "warmup",
]

NUMERIC_COLUMNS = [
    "fps",
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
    "flood_ratio",
]

PLOT_SPECS = [
    ("fps", "Inference FPS (rolling)", "fps_plot_cuda.png", "FPS"),
    ("total_latency_ms", "Total inference latency", "latency_plot_cuda.png", "Latency (ms)"),
    ("classification_ms", "Classification forward pass", "classification_plot_cuda.png", "Latency (ms)"),
    ("segmentation_ms", "Segmentation forward pass", "segmentation_plot_cuda.png", "Latency (ms)"),
    ("model_switch_latency_ms", "Combined model latency", "model_switch_plot_cuda.png", "Latency (ms)"),
    ("clf_load_ms", "Classifier load time", "clf_load_plot_cuda.png", "Load time (ms)"),
    ("seg_load_ms", "Segmenter load time", "seg_load_plot_cuda.png", "Load time (ms)"),
    ("memory_mb", "Process memory", "memory_plot_cuda.png", "Memory (MB)"),
    ("cpu_percent", "CPU usage", "cpu_plot_cuda.png", "CPU (%)"),
    ("power_w", "Board power (VDD_IN)", "power_plot_cuda.png", "Power (W)"),
    ("peak_power_w", "Peak board power", "peak_power_plot_cuda.png", "Power (W)"),
    ("flood_ratio", "Flood ratio", "flood_ratio_plot_cuda.png", "Ratio"),
]


def ensure_csv_header(file_path):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w", newline="") as f:
            csv.writer(f).writerow(METRICS_CUDA_COLUMNS)
        return

    with open(file_path, newline="") as f:
        first_row = next(csv.reader(f), None)

    if first_row and first_row[0] == "timestamp":
        return

    with open(file_path, "r", encoding="utf-8") as f:
        body = f.read()

    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(METRICS_CUDA_COLUMNS)
        if body:
            f.write(body if body.endswith("\n") else body + "\n")


def load_metrics_cuda(file_path):
    ensure_csv_header(file_path)

    df = pd.read_csv(file_path)
    if list(df.columns) != METRICS_CUDA_COLUMNS:
        df = pd.read_csv(file_path, names=METRICS_CUDA_COLUMNS, header=0)

    df = df[df["timestamp"].astype(str) != "timestamp"]
    df = df.dropna(how="all")

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    if "warmup" in df.columns:
        df["warmup"] = (
            df["warmup"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    df = df.dropna(subset=["timestamp", "fps", "total_latency_ms"])
    return df.reset_index(drop=True)
