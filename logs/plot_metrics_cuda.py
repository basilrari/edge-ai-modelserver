import os
import sys

import matplotlib.pyplot as plt

# Allow imports from model_server root when run from logs/
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.metrics_csv import PLOT_SPECS, load_metrics_cuda

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "metrics_cuda.csv")


def plot_column(df, column, title, filename, ylabel):
    if column not in df.columns:
        print(f"[WARN] Missing column: {column}")
        return

    series = df[column].dropna()
    if series.empty:
        print(f"[WARN] No data for column: {column}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(series.values)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, filename))
    plt.close()
    print(f"[OK] {filename}")


def main():
    df = load_metrics_cuda(CSV_PATH)

    if "warmup" in df.columns:
        df = df[df["warmup"] == False]

    print("[INFO] Loaded metrics_cuda.csv")
    print(f"[INFO] Rows after cleanup: {len(df)}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(df.head())

    for column, title, filename, ylabel in PLOT_SPECS:
        plot_column(df, column, title, filename, ylabel)

    print("\n========== CUDA SUMMARY ==========")
    print(f"Average FPS: {df['fps'].mean():.2f}")
    print(f"Average total inference latency: {df['total_latency_ms'].mean():.2f} ms")
    print(f"Average classification latency: {df['classification_ms'].mean():.2f} ms")
    print(f"Average segmentation latency: {df['segmentation_ms'].mean():.2f} ms")
    print(f"Classifier load time: {df['clf_load_ms'].iloc[-1]:.2f} ms")
    print(f"Segmenter load time: {df['seg_load_ms'].iloc[-1]:.2f} ms")
    print(f"Peak memory: {df['memory_mb'].max():.2f} MB")
    print(f"Average CPU: {df['cpu_percent'].mean():.2f} %")

    if df["power_w"].fillna(0).max() > 0:
        print(f"Average power: {df['power_w'].mean():.2f} W")
        print(f"Peak power: {df['peak_power_w'].max():.2f} W")

    print("===================================\n")
    print("[INFO] CUDA plots generated successfully")


if __name__ == "__main__":
    main()
