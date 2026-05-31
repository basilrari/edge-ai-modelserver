import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PATH SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.path.join(BASE_DIR, "metrics.csv")

# ==========================================
# LOAD CSV
# ==========================================
df = pd.read_csv(CSV_PATH)

print("[INFO] Loaded metrics.csv")
print(df.head())

# ==========================================
# FPS PLOT
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["fps"])

plt.title("FPS vs Frame")
plt.xlabel("Frame")
plt.ylabel("FPS")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "fps_plot.png")
)

# ==========================================
# TOTAL LATENCY
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["total_latency_ms"])

plt.title("Total Latency")
plt.xlabel("Frame")
plt.ylabel("Latency (ms)")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "latency_plot.png")
)

# ==========================================
# CLASSIFICATION LATENCY
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["classification_ms"])

plt.title("Classification Latency")
plt.xlabel("Frame")
plt.ylabel("Latency (ms)")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "classification_plot.png")
)

# ==========================================
# SEGMENTATION LATENCY
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["segmentation_ms"])

plt.title("Segmentation Latency")
plt.xlabel("Frame")
plt.ylabel("Latency (ms)")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "segmentation_plot.png")
)

# ==========================================
# MEMORY USAGE
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["memory_mb"])

plt.title("Memory Usage")
plt.xlabel("Frame")
plt.ylabel("Memory (MB)")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "memory_plot.png")
)

# ==========================================
# CPU USAGE
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["cpu_percent"])

plt.title("CPU Usage")
plt.xlabel("Frame")
plt.ylabel("CPU (%)")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "cpu_plot.png")
)

# ==========================================
# POWER CONSUMPTION
# ==========================================
if "power_w" in df.columns:

    plt.figure(figsize=(10, 5))

    plt.plot(df["power_w"])

    plt.title("Power Consumption")
    plt.xlabel("Frame")
    plt.ylabel("Power (W)")

    plt.grid(True)

    plt.savefig(
        os.path.join(BASE_DIR, "power_plot.png")
    )

# ==========================================
# FLOOD RATIO
# ==========================================
plt.figure(figsize=(10, 5))

plt.plot(df["flood_ratio"])

plt.title("Flood Ratio")
plt.xlabel("Frame")
plt.ylabel("Flood Ratio")

plt.grid(True)

plt.savefig(
    os.path.join(BASE_DIR, "flood_ratio_plot.png")
)

# ==========================================
# SUMMARY STATISTICS
# ==========================================
print("\n========== SUMMARY ==========")

print(f"Average FPS: {df['fps'].mean():.2f}")

print(
    f"Average Total Latency: "
    f"{df['total_latency_ms'].mean():.2f} ms"
)

print(
    f"Average Classification Latency: "
    f"{df['classification_ms'].mean():.2f} ms"
)

print(
    f"Average Segmentation Latency: "
    f"{df['segmentation_ms'].mean():.2f} ms"
)

print(
    f"Peak Memory Usage: "
    f"{df['memory_mb'].max():.2f} MB"
)

print(
    f"Average CPU Usage: "
    f"{df['cpu_percent'].mean():.2f} %"
)

if "power_w" in df.columns:

    print(
        f"Average Power Consumption: "
        f"{df['power_w'].mean():.2f} W"
    )

print("=============================\n")

print("[INFO] All plots generated successfully")
