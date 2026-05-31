import csv
import os
import time

from core.metrics_csv import METRICS_CUDA_COLUMNS, ensure_csv_header


class MetricsLogger:

    def __init__(self, file_path="logs/metrics_cuda.csv"):
        os.makedirs("logs", exist_ok=True)
        self.file_path = file_path
        ensure_csv_header(self.file_path)
        self.columns = METRICS_CUDA_COLUMNS

    def log(self, metrics):
        row_map = {
            "timestamp": time.time(),
            "fps": metrics.get("fps"),
            "total_latency_ms": metrics.get("total_latency_ms"),
            "classification_ms": metrics.get("classification_ms"),
            "segmentation_ms": metrics.get("segmentation_ms"),
            "model_switch_latency_ms": metrics.get("model_switch_latency_ms"),
            "memory_mb": metrics.get("memory_mb"),
            "cpu_percent": metrics.get("cpu_percent"),
            "power_w": metrics.get("power_w"),
            "peak_memory_mb": metrics.get("peak_memory_mb"),
            "peak_cpu_percent": metrics.get("peak_cpu_percent"),
            "peak_power_w": metrics.get("peak_power_w"),
            "clf_load_ms": metrics.get("clf_load_ms"),
            "seg_load_ms": metrics.get("seg_load_ms"),
            "flood_ratio": metrics.get("flood_ratio"),
            "warmup": metrics.get("warmup"),
        }

        with open(self.file_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row_map.get(col) for col in self.columns])
