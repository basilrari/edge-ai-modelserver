"""Generate benchmark plots and HTML/Markdown report from a metrics CSV."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Per-frame metrics to plot vs frame_idx (genuine time series)
FRAME_SERIES_METRICS = [
    ("total_latency_ms", "Total wall latency (ms)", False),
    ("total_inference_ms", "GPU inference time (ms)", False),
    ("instant_fps", "Instant FPS", False),
    ("classification_ms", "Classification (ms)", False),
    ("segmentation_ms", "Segmentation (ms)", False),
    ("flood_ratio", "Flood ratio", False),
    ("human_count", "Humans detected", True),
    ("memory_mb", "Memory (MB)", False),
    ("cpu_percent", "CPU %", False),
]

# Session-level constants — never plot as time series (flat lines)
SESSION_CONSTANT_COLS = frozenset(
    {
        "clf_load_ms",
        "seg_load_ms",
        "human_load_ms",
        "model_load_total_ms",
        "sim_altitude_m",
        "input_width",
        "input_height",
    }
)


def load_benchmark_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if col in (
            "warmup",
            "model_switch_occurred",
            "segmentation_skipped",
            "session_first_frame",
        ):
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False, "1": True, "0": False}
            )
        elif col not in (
            "video_id",
            "video_name",
            "mode",
            "active_tool",
            "classification_label",
            "primary_model",
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "warmup" in df.columns:
        df = df[df["warmup"] != True]
    if "frame_idx" in df.columns:
        df = df.sort_values("frame_idx").reset_index(drop=True)
    return df.dropna(subset=["total_latency_ms"])


def _filter_primary_series(df: pd.DataFrame) -> pd.DataFrame:
    """One row per frame_idx — avoids zigzag lines from multi-altitude/mode CSVs."""
    if df.empty:
        return df

    group_cols = [
        c
        for c in ("video_id", "video_name", "mode", "active_tool", "sim_altitude_m")
        if c in df.columns and df[c].notna().any()
    ]
    if group_cols:
        counts = df.groupby(group_cols, dropna=False).size()
        primary_key = counts.idxmax()
        mask = pd.Series(True, index=df.index)
        for col, val in zip(group_cols, primary_key):
            mask &= df[col].fillna("__na__") == (val if pd.notna(val) else "__na__")
        df = df[mask].copy()

    if "frame_idx" in df.columns:
        df = df.drop_duplicates(subset=["frame_idx"], keep="last")
        df = df.sort_values("frame_idx").reset_index(drop=True)
    return df


def _detect_sampling_stride(df: pd.DataFrame) -> int:
    if "frame_idx" not in df.columns or len(df) < 2:
        return 1
    gaps = df["frame_idx"].diff().dropna()
    gaps = gaps[gaps > 0]
    if gaps.empty:
        return 1
    return int(max(1, round(float(gaps.median()))))


def _x_values(df: pd.DataFrame) -> np.ndarray:
    if "frame_idx" in df.columns and df["frame_idx"].notna().any():
        return df["frame_idx"].values.astype(float)
    return np.arange(len(df), dtype=float)


def _is_sparse_sampling(x: np.ndarray, n_points: int) -> bool:
    """Sparse stride or few samples — connecting points misleads (looks linear)."""
    if n_points <= 15:
        return True
    if len(x) < 2:
        return True
    gaps = np.diff(x)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return True
    return float(gaps.max()) > max(2.0, 1.5 * float(np.median(gaps)))


def _plot_series(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    ylabel: str,
    discrete: bool = False,
    n_points: int,
    sparse: bool = False,
) -> None:
    valid = ~np.isnan(y)
    y_unique = len(np.unique(y[valid])) if valid.any() else 0

    if sparse:
        ax.scatter(
            x,
            y,
            s=48,
            color="#f472b6",
            alpha=0.9,
            zorder=3,
            edgecolors="#38bdf8",
            linewidths=0.6,
        )
        ax.set_title(f"{title} (sparse sampling — points only)")
    elif discrete or y_unique <= 8:
        ax.step(x, y, where="post", color="#38bdf8", linewidth=1.2, alpha=0.9)
        ax.scatter(x, y, s=18, color="#f472b6", alpha=0.85, zorder=3)
        ax.set_title(title)
    elif n_points <= 20:
        ax.plot(x, y, color="#38bdf8", linewidth=1.0, alpha=0.7)
        ax.scatter(x, y, s=22, color="#f472b6", alpha=0.9, zorder=3)
        ax.set_title(title)
    else:
        ax.plot(x, y, color="#38bdf8", linewidth=0.8, alpha=0.75)
        ax.scatter(x, y, s=8, color="#f472b6", alpha=0.35, zorder=3)
        ax.set_title(title)

    ax.set_xlabel("Video frame index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if len(x) > 1:
        ax.set_xlim(x.min(), x.max())


def plot_metrics_vs_frame(df: pd.DataFrame, out: Path) -> None:
    """Per-metric plots against real frame_idx (not sample index)."""
    df = _filter_primary_series(df)
    x = _x_values(df)
    n = len(df)
    sparse = _is_sparse_sampling(x, n)

    for col, title, discrete in FRAME_SERIES_METRICS:
        if col not in df.columns or df[col].dropna().empty:
            continue
        if col in SESSION_CONSTANT_COLS:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue

        y = df[col].values.astype(float)
        fig, ax = plt.subplots(figsize=(11, 3.5))
        _plot_series(
            ax,
            x,
            y,
            title=title,
            ylabel=col,
            discrete=discrete,
            n_points=n,
            sparse=sparse,
        )
        plt.tight_layout()
        plt.savefig(out / f"series_{col}.png", dpi=120)
        plt.close()

    # Combined overview (key metrics normalized on twin axes)
    key = [
        c
        for c, _, _ in FRAME_SERIES_METRICS
        if c in df.columns and df[c].nunique(dropna=True) > 1
    ][:6]
    if len(key) >= 2:
        fig, axes = plt.subplots(len(key), 1, figsize=(11, 2.2 * len(key)), sharex=True)
        if len(key) == 1:
            axes = [axes]
        for ax, col in zip(axes, key):
            discrete = col == "human_count"
            _plot_series(
                ax,
                x,
                df[col].values.astype(float),
                title=col,
                ylabel=col,
                discrete=discrete,
                n_points=n,
                sparse=sparse,
            )
        axes[-1].set_xlabel("Video frame index")
        title = "Per-frame metrics (aligned to source video frames)"
        if sparse:
            title += " — sparse sampling; use stride=1 for full detail"
        plt.suptitle(title, y=1.002)
        plt.tight_layout()
        plt.savefig(out / "overview_per_frame.png", dpi=120)
        plt.close()


def plot_latency_vs_altitude(df: pd.DataFrame, out: Path) -> None:
    if df["sim_altitude_m"].nunique() < 2:
        return
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
    if df["sim_altitude_m"].nunique() < 2:
        return
    plt.figure(figsize=(8, 5))
    for mode in sorted(df["mode"].dropna().unique()):
        sub = df[df["mode"] == mode]
        grp = sub.groupby("sim_altitude_m")["instant_fps"].mean()
        plt.plot(grp.index, grp.values, marker="s", label=mode)
    plt.xlabel("Simulated altitude (m AGL)")
    plt.ylabel("Instant FPS")
    plt.title("FPS vs altitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "fps_vs_altitude.png", dpi=120)
    plt.close()


def plot_flood_ratio(df: pd.DataFrame, out: Path) -> None:
    sub = _filter_primary_series(df[df["mode"].isin(["flood", "combined"])])
    if sub.empty or "flood_ratio" not in sub.columns:
        return
    if sub["flood_ratio"].nunique(dropna=True) <= 1:
        return

    x = _x_values(sub)
    sparse = _is_sparse_sampling(x, len(sub))
    fig, ax = plt.subplots(figsize=(11, 4))
    _plot_series(
        ax,
        x,
        sub["flood_ratio"].values.astype(float),
        title="Flood ratio over video frames",
        ylabel="Flood ratio",
        discrete=False,
        n_points=len(sub),
        sparse=sparse,
    )
    ax.axhline(0.2, color="r", linestyle="--", linewidth=1, label="threshold 0.2")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "flood_ratio_by_altitude.png", dpi=120)
    plt.close()


def plot_human_count(df: pd.DataFrame, out: Path) -> None:
    sub = _filter_primary_series(df[df["mode"].isin(["human", "combined"])])
    if sub.empty or "human_count" not in sub.columns:
        return
    if sub["human_count"].nunique(dropna=True) <= 1:
        return

    x = _x_values(sub)
    sparse = _is_sparse_sampling(x, len(sub))
    fig, ax = plt.subplots(figsize=(11, 4))
    _plot_series(
        ax,
        x,
        sub["human_count"].values.astype(float),
        title="Human count over video frames",
        ylabel="Humans",
        discrete=True,
        n_points=len(sub),
        sparse=sparse,
    )
    plt.tight_layout()
    plt.savefig(out / "human_count_over_time.png", dpi=120)
    plt.close()


def plot_tradeoff_scatter(df: pd.DataFrame, out: Path) -> None:
    sub = _filter_primary_series(df[df["mode"].isin(["flood", "combined"])])
    if sub.empty:
        return

    x = _x_values(sub)
    y = sub["total_latency_ms"].values.astype(float)
    c = sub["flood_ratio"].values.astype(float) if "flood_ratio" in sub.columns else None

    plt.figure(figsize=(10, 5))
    if df["sim_altitude_m"].nunique() >= 2:
        sc = plt.scatter(
            sub["sim_altitude_m"],
            y,
            c=c,
            s=30,
            cmap="viridis",
            alpha=0.7,
        )
        plt.xlabel("Simulated altitude (m)")
        plt.title("Altitude vs latency (color = flood ratio)")
    else:
        sc = plt.scatter(
            x,
            y,
            c=c,
            s=30,
            cmap="viridis",
            alpha=0.7,
        )
        plt.xlabel("Video frame index")
        plt.title("Latency per frame (color = flood ratio)")
    plt.colorbar(sc, label="Flood ratio")
    plt.ylabel("Total latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "tradeoff_altitude_latency.png", dpi=120)
    plt.close()


def plot_model_load(lifecycle: dict, out: Path) -> None:
    if not lifecycle:
        return
    labels = ["clf_load_ms", "seg_load_ms", "human_load_ms"]
    vals = [lifecycle.get(k, 0) or 0 for k in labels]
    if not any(vals):
        return
    plt.figure(figsize=(7, 4))
    plt.bar(
        ["Classifier", "Segmenter", "Human"],
        vals,
        color=["#3b82f6", "#22c55e", "#a855f7"],
    )
    plt.ylabel("Load time (ms)")
    plt.title("Model load times (session start — not per-frame)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "model_load_times.png", dpi=120)
    plt.close()


def write_report(df: pd.DataFrame, out: Path, *, lifecycle: dict | None = None) -> dict:
    plot_df = _filter_primary_series(df)
    sampling_stride = _detect_sampling_stride(plot_df)
    summary = {
        "frames": len(plot_df),
        "csv_rows_raw": len(df),
        "sampling_stride": sampling_stride,
        "videos": plot_df["video_name"].unique().tolist()
        if "video_name" in plot_df.columns
        else [],
        "modes": plot_df["mode"].unique().tolist() if "mode" in plot_df.columns else [],
        "active_tools": (
            plot_df["active_tool"].unique().tolist()
            if "active_tool" in plot_df.columns
            else []
        ),
        "altitudes_m": sorted(plot_df["sim_altitude_m"].dropna().unique().tolist()),
        "frame_idx_range": [
            int(plot_df["frame_idx"].min()),
            int(plot_df["frame_idx"].max()),
        ]
        if "frame_idx" in plot_df.columns and len(plot_df)
        else [],
        "mean_latency_ms": float(plot_df["total_latency_ms"].mean()),
        "std_latency_ms": float(plot_df["total_latency_ms"].std()),
        "mean_fps": float(plot_df["instant_fps"].mean()),
        "std_fps": float(plot_df["instant_fps"].std()),
        "mean_flood_ratio": float(plot_df["flood_ratio"].mean())
        if "flood_ratio" in plot_df
        else 0,
        "mean_human_count": float(plot_df["human_count"].mean())
        if "human_count" in plot_df
        else 0,
    }
    if sampling_stride > 1:
        summary["plot_warning"] = (
            f"Metrics sampled every {sampling_stride} frames — plots show only "
            f"{summary['frames']} points. Re-run with stride=1 for full per-frame curves."
        )
    if lifecycle:
        summary["lifecycle"] = lifecycle

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Dashboard Offline Benchmark Report",
        "",
        f"- Frames recorded: **{summary['frames']}**",
        f"- Sampling stride: **{sampling_stride}** (1 = every frame)",
        f"- Frame index range: **{summary.get('frame_idx_range', [])}**",
        f"- Mean latency: **{summary['mean_latency_ms']:.1f} ± {summary['std_latency_ms']:.1f} ms**",
        f"- Mean FPS: **{summary['mean_fps']:.2f} ± {summary['std_fps']:.2f}**",
        f"- Mean flood ratio: **{summary['mean_flood_ratio']:.3f}**",
        "",
        "Time-series plots use **video frame index** on the x-axis (not sample index).",
    ]
    if sampling_stride > 1:
        md.append(
            f"\n> **Warning:** stride={sampling_stride} — only {summary['frames']} samples. "
            "Use stride **1** in the dashboard for genuine per-frame plots."
        )
    (out / "REPORT.md").write_text("\n".join(md), encoding="utf-8")

    imgs = sorted(out.glob("*.png"))
    html_imgs = "\n".join(
        f'<h3>{p.stem}</h3><img src="{p.name}" width="900"/>' for p in imgs
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
    <title>Benchmark Report</title>
    <style>body{{font-family:sans-serif;max-width:960px;margin:2em auto}}</style>
    </head><body>
    <h1>Offline Benchmark Report</h1>
    <pre>{json.dumps(summary, indent=2)}</pre>
    {html_imgs}
    </body></html>"""
    (out / "report.html").write_text(html, encoding="utf-8")
    return summary


def generate_benchmark_report(
    csv_path: Path,
    out_dir: Path,
    *,
    lifecycle: dict | None = None,
    system: dict | None = None,
) -> dict:
    """Write all plots + report into out_dir. Returns summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {"error": "no metrics csv", "output_dir": str(out_dir)}

    df_raw = load_benchmark_df(csv_path)
    if df_raw.empty:
        return {"error": "empty metrics", "output_dir": str(out_dir)}

    df = _filter_primary_series(df_raw)
    plot_metrics_vs_frame(df_raw, out_dir)
    plot_latency_vs_altitude(df_raw, out_dir)
    plot_fps_vs_altitude(df_raw, out_dir)
    plot_flood_ratio(df_raw, out_dir)
    plot_human_count(df_raw, out_dir)
    plot_tradeoff_scatter(df_raw, out_dir)
    if lifecycle:
        plot_model_load(lifecycle, out_dir)

    summary = write_report(df_raw, out_dir, lifecycle=lifecycle)
    summary["output_dir"] = str(out_dir)
    summary["csv"] = str(csv_path)
    return summary
