#!/usr/bin/env python3
"""Generate plots + HTML/Markdown report from drone video benchmark CSV."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks/results"
CSV_PATH = RESULTS / "drone_video_benchmark.csv"
SYSTEM_PATH = RESULTS / "system_benchmark.json"
REPORT_DIR = RESULTS / "report"


def load_benchmark_df() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    for col in df.columns:
        if col in ("warmup", "model_switch_occurred", "segmentation_skipped"):
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        elif col not in ("video_id", "video_name", "mode", "classification_label", "primary_model"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "warmup" in df.columns:
        df = df[df["warmup"] != True]
    return df.dropna(subset=["total_latency_ms"])


def plot_latency_vs_altitude(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metrics = [
        ("total_latency_ms", "Total latency (ms)"),
        ("classification_ms", "Classification (ms)"),
        ("segmentation_ms", "Segmentation (ms)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        for mode in sorted(df["mode"].dropna().unique()):
            sub = df[df["mode"] == mode]
            grp = sub.groupby("sim_altitude_m")[col].mean()
            ax.plot(grp.index, grp.values, marker="o", label=mode)
        ax.set_xlabel("Simulated altitude (m AGL)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(out / "latency_vs_altitude.png", dpi=120)
    plt.close()


def plot_fps_vs_altitude(df: pd.DataFrame, out: Path) -> None:
    plt.figure(figsize=(8, 5))
    for mode in sorted(df["mode"].dropna().unique()):
        sub = df[df["mode"] == mode]
        grp = sub.groupby("sim_altitude_m")["instant_fps"].mean()
        plt.plot(grp.index, grp.values, marker="s", label=mode)
    plt.xlabel("Simulated altitude (m AGL)")
    plt.ylabel("Instant FPS")
    plt.title("FPS vs altitude (higher altitude → smaller input → often faster)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fps_vs_altitude.png", dpi=120)
    plt.close()


def plot_flood_ratio(df: pd.DataFrame, out: Path) -> None:
    altitudes = sorted(df["sim_altitude_m"].dropna().unique())
    n = len(altitudes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, alt in zip(axes, altitudes):
        sub = df[(df["sim_altitude_m"] == alt) & (df["mode"].isin(["flood", "combined"]))]
        ax.plot(sub["frame_idx"].values, sub["flood_ratio"].values, alpha=0.7)
        ax.axhline(0.2, color="r", linestyle="--", label="threshold 0.2")
        ax.set_title(f"{alt:.0f} m AGL")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Flood ratio")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Flood ratio over time by simulated altitude")
    plt.tight_layout()
    plt.savefig(out / "flood_ratio_by_altitude.png", dpi=120)
    plt.close()


def plot_tradeoff_scatter(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["mode"].isin(["flood", "combined"])]
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        sub["sim_altitude_m"],
        sub["total_latency_ms"],
        c=sub["flood_ratio"],
        s=30,
        cmap="viridis",
        alpha=0.7,
    )
    plt.colorbar(sc, label="Flood ratio")
    plt.xlabel("Simulated altitude (m)")
    plt.ylabel("Total latency (ms)")
    plt.title("Trade-off: altitude vs latency (color = flood ratio)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "tradeoff_altitude_latency.png", dpi=120)
    plt.close()


def plot_coverage_effect(df: pd.DataFrame, out: Path) -> None:
    """Ground coverage factor vs detection stability."""
    ref_alt = 50.0
    sub = df.copy()
    sub["coverage_factor"] = sub["sim_altitude_m"] / ref_alt
    grp = sub.groupby("sim_altitude_m").agg(
        mean_latency=("total_latency_ms", "mean"),
        mean_flood=("flood_ratio", "mean"),
        mean_fps=("instant_fps", "mean"),
        coverage=("coverage_factor", "first"),
    )
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(
        grp.index.astype(str),
        grp["coverage"],
        alpha=0.3,
        label="Ground coverage factor (×)",
    )
    ax1.set_xlabel("Altitude (m)")
    ax1.set_ylabel("Coverage factor vs 50m ref")
    ax2 = ax1.twinx()
    ax2.plot(grp.index, grp["mean_latency"], "r-o", label="Mean latency (ms)")
    ax2.set_ylabel("Mean latency (ms)")
    plt.title("Area coverage vs inference cost")
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out / "coverage_vs_latency.png", dpi=120)
    plt.close()


def plot_live_pipeline_metrics(df: pd.DataFrame, out: Path) -> None:
    """Same charts as logs/plot_metrics_cuda.py for benchmark run."""
    specs = [
        ("total_latency_ms", "Total inference latency"),
        ("classification_ms", "Classification"),
        ("segmentation_ms", "Segmentation"),
        ("memory_mb", "Memory (MB)"),
        ("cpu_percent", "CPU %"),
        ("flood_ratio", "Flood ratio"),
    ]
    for col, title in specs:
        if col not in df.columns or df[col].dropna().empty:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(df[col].values)
        plt.title(f"{title} (live_pipeline fields)")
        plt.xlabel("Sample index")
        plt.ylabel(col)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / f"series_{col}.png", dpi=120)
        plt.close()


def plot_system_benchmark(system: dict, out: Path) -> None:
    if not system:
        return
    lu = system.get("load_unload", {})
    names = list(lu.keys())
    load_ms = [lu[n]["load_ms"] for n in names]
    unload_ms = [lu[n]["unload_ms"] for n in names]
    reload_ms = [lu[n].get("reload_ms") or 0 for n in names]

    x = np.arange(len(names))
    w = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - w, load_ms, w, label="Cold load")
    plt.bar(x, reload_ms, w, label="Reload")
    plt.bar(x + w, unload_ms, w, label="Unload")
    plt.xticks(x, names, rotation=15)
    plt.ylabel("Time (ms)")
    plt.title("Model load / unload / reload (ModelManager)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "load_unload_reload.png", dpi=120)
    plt.close()

    sw = system.get("logical_primary_switch", {})
    if sw.get("events"):
        ev = sw["events"]
        plt.figure(figsize=(8, 3))
        plt.stem(
            [e["step"] for e in ev],
            [e["latency_ms"] for e in ev],
        )
        plt.xlabel("Switch event")
        plt.ylabel("Orchestration latency (ms)")
        plt.title(f"Primary model switches (count={sw.get('switch_count', 0)})")
        plt.tight_layout()
        plt.savefig(out / "model_switch_events.png", dpi=120)
        plt.close()


def write_report(df: pd.DataFrame, system: dict, out: Path) -> None:
    summary = {
        "frames": len(df),
        "videos": df["video_name"].unique().tolist(),
        "modes": df["mode"].unique().tolist(),
        "altitudes_m": sorted(df["sim_altitude_m"].dropna().unique().tolist()),
        "mean_latency_ms": float(df["total_latency_ms"].mean()),
        "mean_fps": float(df["instant_fps"].mean()),
        "mean_flood_ratio": float(df["flood_ratio"].mean()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Drone Video Benchmark Report",
        "",
        "## Run summary",
        f"- Frames analyzed: **{summary['frames']}**",
        f"- Videos: {', '.join(summary['videos'])}",
        f"- Modes: {', '.join(summary['modes'])}",
        f"- Simulated altitudes (m): {summary['altitudes_m']}",
        f"- Mean latency: **{summary['mean_latency_ms']:.1f} ms**",
        f"- Mean instant FPS: **{summary['mean_fps']:.2f}**",
        "",
        "## Altitude / coverage trade-offs",
        "",
        "| Altitude (m) | Coverage factor (×) | Mean latency (ms) | Mean flood ratio | Mean FPS |",
        "|-------------|---------------------|-------------------|------------------|----------|",
    ]
    ref = 50.0
    for alt in summary["altitudes_m"]:
        sub = df[df["sim_altitude_m"] == alt]
        md.append(
            f"| {alt:.0f} | {alt/ref:.2f} | {sub['total_latency_ms'].mean():.1f} | "
            f"{sub['flood_ratio'].mean():.3f} | {sub['instant_fps'].mean():.2f} |"
        )

    md.extend(
        [
            "",
            "Higher altitude → **wider area per frame** but **coarser pixels** (flood/human details shrink).",
            "",
            "## live_pipeline.py metrics (included in CSV)",
            "",
            "| Metric | Description |",
            "|--------|-------------|",
            "| clf_load_ms / seg_load_ms | One-time model load at session start |",
            "| classification_ms / segmentation_ms | Per-frame GPU forward pass |",
            "| model_switch_latency_ms | clf + seg latency (combined path) |",
            "| total_latency_ms / total_inference_ms | End-to-end inference time |",
            "| fps / instant_fps | Throughput |",
            "| memory_mb / cpu_percent / power_w | Resource use |",
            "| flood_ratio | DeepLab flood pixel fraction |",
            "",
            "## Model switch & load/unload",
            "",
        ]
    )

    if system:
        lu = system.get("load_unload", {})
        for name, v in lu.items():
            md.append(
                f"- **{name}**: load {v['load_ms']:.0f} ms, unload {v['unload_ms']:.0f} ms, "
                f"reload {v.get('reload_ms', 0):.0f} ms"
            )
        sw = system.get("logical_primary_switch", {})
        md.append(
            f"- **Logical primary switch** (ResNet↔DeepLab): "
            f"{sw.get('switch_count', 0)} events, "
            f"mean {sw.get('switch_latency_ms', {}).get('mean', 0):.3f} ms orchestration"
        )
        phys = system.get("physical_model_swap", {})
        md.append(
            f"- **Physical swap** (unload clf + load seg): "
            f"{phys.get('flood_classifier_to_segmenter_ms', 'n/a')} ms"
        )

    md.extend(
        [
            "",
            "## Plots",
            "",
            "See PNG files in this folder.",
            "",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    imgs = sorted(out.glob("*.png"))
    html_imgs = "\n".join(
        f'<h3>{p.stem}</h3><img src="{p.name}" width="900"/>' for p in imgs
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Drone Benchmark Report</title>
    <style>body{{font-family:sans-serif;max-width:960px;margin:2em auto}}</style>
    </head><body>
    <h1>Drone Video Benchmark Report</h1>
    <pre>{json.dumps(summary, indent=2)}</pre>
    {html_imgs}
    </body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists():
        print(f"Missing {CSV_PATH} — run tools/run_drone_video_benchmark.py first")
        sys.exit(1)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    system = {}
    if SYSTEM_PATH.exists():
        system = json.loads(SYSTEM_PATH.read_text(encoding="utf-8"))

    from core.benchmark_report import generate_benchmark_report

    print(f"[PLOT] → {REPORT_DIR}")
    summary = generate_benchmark_report(CSV_PATH, REPORT_DIR)
    if system:
        plot_system_benchmark(system, REPORT_DIR)
    print(f"[PLOT] summary frames={summary.get('frames')} stride={summary.get('sampling_stride')}")

    print(f"[DONE] Report: {REPORT_DIR / 'REPORT.md'}")
    print(f"[DONE] HTML:  {REPORT_DIR / 'report.html'}")


if __name__ == "__main__":
    main()
