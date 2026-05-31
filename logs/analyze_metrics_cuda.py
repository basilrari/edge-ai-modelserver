import os
import sys
import json

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.metrics_csv import load_metrics_cuda

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metrics_cuda.csv")


def p95(x):
    return np.percentile(x, 95)


def stats(x):
    return {
        "mean": float(np.mean(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p95": float(p95(x)),
    }


def main():
    df = load_metrics_cuda(CSV_PATH)

    if "warmup" in df.columns:
        df = df[df["warmup"] == False]

    print("\n[INFO] Loaded CUDA metrics:", len(df), "rows")

    results = {
        "total_latency_ms": stats(df["total_latency_ms"]),
        "classification_ms": stats(df["classification_ms"]),
        "segmentation_ms": stats(df["segmentation_ms"]),
        "model_switch_latency_ms": stats(df["model_switch_latency_ms"]),
        "fps": stats(df["fps"]),
        "cpu_percent": stats(df["cpu_percent"]),
        "memory_mb": stats(df["memory_mb"]),
        "clf_load_ms": stats(df["clf_load_ms"]),
        "seg_load_ms": stats(df["seg_load_ms"]),
        "flood_ratio": stats(df["flood_ratio"]),
    }

    if df["power_w"].fillna(0).max() > 0:
        results["power_w"] = stats(df["power_w"])
        results["peak_power_w"] = stats(df["peak_power_w"])

    avg_fps = df["fps"].mean()
    avg_latency = df["total_latency_ms"].mean()

    results["derived"] = {
        "inferences_per_second": float(avg_fps),
        "avg_latency_ms": float(avg_latency),
        "fps_per_watt": float(df["fps"].mean() / df["power_w"].mean())
        if df["power_w"].fillna(0).max() > 0
        else None,
    }

    print("\n========== CUDA BENCHMARK SUMMARY ==========\n")
    for key, value in results.items():
        print(f"[{key}]")
        if isinstance(value, dict):
            for k, v in value.items():
                if v is not None:
                    print(f"  {k}: {v:.4f}")
        print()

    print("============================================\n")
    print(f"Total frames: {len(df)}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average latency: {avg_latency:.2f} ms")
    print(f"Classifier load (last row): {df['clf_load_ms'].iloc[-1]:.2f} ms")
    print(f"Segmenter load (last row): {df['seg_load_ms'].iloc[-1]:.2f} ms")

    out_file = os.path.join(BASE_DIR, "cuda_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[INFO] Saved summary -> {out_file}")


if __name__ == "__main__":
    main()
