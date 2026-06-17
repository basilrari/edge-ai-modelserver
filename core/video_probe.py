"""Read source video metadata and derive benchmark/render settings."""

from __future__ import annotations

from pathlib import Path

import cv2


def probe_video(video_path: Path) -> dict:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 0:
        fps = 25.0
    duration_s = frame_count / fps if frame_count > 0 else 0.0

    return {
        "path": str(video_path),
        "name": video_path.name,
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "duration_s": round(duration_s, 2),
        "duration_mmss": _format_mmss(duration_s),
    }


def _format_mmss(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


def plan_render_match_input(info: dict, render_stride: int = 1, render_fps: float = 0.0) -> dict:
    """Settings so annotated output duration matches the source video."""
    stride = max(1, int(render_stride))
    out_fps = render_fps if render_fps > 0 else info["fps"]
    frames_in = info["frame_count"]
    # Every stride-th frame is inferred; output length = written_frames / out_fps
    written = max(1, (frames_in + stride - 1) // stride) if frames_in else 0
    duration_s = written / out_fps if out_fps > 0 else 0.0

    return {
        "render_stride": stride,
        "render_fps": out_fps,
        "source_frames": frames_in,
        "frames_to_write": written,
        "expected_duration_s": round(duration_s, 2),
        "expected_duration_mmss": _format_mmss(duration_s),
        "matches_input": stride == 1 and abs(duration_s - info["duration_s"]) < 1.0,
    }


def plan_csv_sampling(
    info: dict,
    num_altitudes: int,
    target_rows: int = 60,
) -> dict:
    """
    Spread CSV metric rows across the full video length.

    Note: processed counter increments once per altitude per source frame.
    """
    num_altitudes = max(1, num_altitudes)
    target_rows = max(num_altitudes, target_rows)
    source_hits = max(1, target_rows // num_altitudes)
    frame_count = max(1, info["frame_count"])
    stride = max(1, frame_count // source_hits)
    max_frames = source_hits * num_altitudes

    return {
        "max_frames": max_frames,
        "stride": stride,
        "source_samples": source_hits,
        "approx_span_s": round(source_hits * stride / info["fps"], 1),
    }


def print_video_plan(info: dict, render_plan: dict | None = None, csv_plan: dict | None = None) -> None:
    print(
        f"[VIDEO] {info['name']}: {info['frame_count']} frames @ {info['fps']} fps "
        f"→ {info['duration_mmss']} ({info['duration_s']}s) {info['width']}x{info['height']}"
    )
    if render_plan:
        match = "matches input" if render_plan["matches_input"] else "SHORTER than input"
        print(
            f"[RENDER PLAN] write {render_plan['frames_to_write']} frames @ "
            f"{render_plan['render_fps']} fps → {render_plan['expected_duration_mmss']} "
            f"({render_plan['expected_duration_s']}s) — {match}"
        )
        if not render_plan["matches_input"]:
            print(
                "[RENDER PLAN] Use --render-stride 1 (default) for full 1:1 duration."
            )
    if csv_plan:
        print(
            f"[CSV PLAN] max_frames={csv_plan['max_frames']} stride={csv_plan['stride']} "
            f"(~{csv_plan['source_samples']} positions × {csv_plan.get('altitudes', '?')} altitudes)"
        )
