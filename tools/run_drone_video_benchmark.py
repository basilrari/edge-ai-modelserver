#!/usr/bin/env python3
"""
Offline drone-video benchmark: metrics + altitude trade-offs + system timings.

  cd ~/python-worker/model_server
  python3 tools/download_drone_videos.py --archive-only
  python3 tools/run_drone_video_benchmark.py --video benchmarks/videos/archive_flood_airfield.mp4 --render-only
  python3 tools/plot_drone_benchmark_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.system_benchmark import run_system_benchmark
from core.tool_mapping import normalize_mode, normalize_tool, tool_to_mode
from core.video_benchmark import VideoBenchmarkRunner
from core.video_probe import plan_csv_sampling, plan_render_match_input, print_video_plan, probe_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Drone video offline benchmark")
    parser.add_argument(
        "--video",
        type=str,
        help="Path to .mp4 (default: first file in benchmarks/videos/)",
    )
    parser.add_argument(
        "--altitudes",
        type=str,
        default="25,50,75,100",
        help="Comma-separated simulated AGL heights in metres",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="CSV metric rows cap (default: auto ~60 spread over full video)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="CSV sample every Nth source frame (default: auto from video length)",
    )
    parser.add_argument(
        "--tool",
        type=str,
        default=None,
        help="Explicit live tool: detect_flood | detect_human | detect_combined (recommended)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help="Benchmark modes flood|human|combined (default: derived from --tool)",
    )
    parser.add_argument(
        "--test-unload",
        action="store_true",
        help="After each mode, unload models and record unload timing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
    )
    parser.add_argument("--skip-system", action="store_true")
    parser.add_argument(
        "--render-video",
        action="store_true",
        help="Write full-length annotated MP4 (matches input duration)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Mode for output video (default: combined, or first in --modes)",
    )
    parser.add_argument(
        "--render-altitude",
        type=float,
        default=50.0,
        help="Simulated altitude (m) used for the review video",
    )
    parser.add_argument(
        "--render-fps",
        type=float,
        default=0.0,
        help="Output FPS (0 = same as source — keeps 1:28 → 1:28)",
    )
    parser.add_argument(
        "--render-stride",
        type=int,
        default=1,
        help="1 = every frame, full duration (do not increase unless speeding up)",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Only write annotated review video (no CSV)",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    video_dir = ROOT / "benchmarks/videos"
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = ROOT / video_path
    else:
        candidates = sorted(video_dir.glob("*.mp4"))
        if not candidates:
            print("No video found. Run: python3 tools/download_drone_videos.py")
            print("Or pass: --video /path/to/drone_flood.mp4")
            sys.exit(1)
        video_path = candidates[0]

    info = probe_video(video_path)
    altitudes = [float(x.strip()) for x in args.altitudes.split(",") if x.strip()]

    if args.tool:
        primary_mode = tool_to_mode(normalize_tool(args.tool) or args.tool)
        if not primary_mode:
            print(f"Unknown --tool {args.tool!r}. Use detect_flood | detect_human | detect_combined")
            sys.exit(1)
        modes = (primary_mode,)
        print(f"[BENCHMARK] explicit --tool {normalize_tool(args.tool)} → mode={primary_mode}")
    elif args.modes:
        modes = tuple(
            normalize_mode(m.strip()) or m.strip()
            for m in args.modes.split(",")
            if m.strip()
        )
    else:
        modes = ("combined",)
        print("[BENCHMARK] default tool=detect_combined (pass --tool to override)")

    render_mode = normalize_mode(args.render_mode) if args.render_mode else None
    render_mode = render_mode or (
        "combined" if "combined" in modes else (modes[0] if modes else "flood")
    )
    if args.render_only:
        args.render_video = True
        args.skip_system = True

    render_plan = None
    if args.render_video:
        render_plan = plan_render_match_input(
            info, render_stride=args.render_stride, render_fps=args.render_fps
        )

    csv_plan = None
    max_frames = args.max_frames
    stride = args.stride
    if not args.render_only:
        if max_frames is None or stride is None:
            csv_plan = plan_csv_sampling(info, num_altitudes=len(altitudes))
            if max_frames is None:
                max_frames = csv_plan["max_frames"]
            if stride is None:
                stride = csv_plan["stride"]
            csv_plan["altitudes"] = len(altitudes)

    print_video_plan(info, render_plan=render_plan, csv_plan=csv_plan)

    csv_path = out_dir / "drone_video_benchmark.csv"
    system_json = out_dir / "system_benchmark.json"
    run_meta = out_dir / "run_metadata.json"

    if not args.skip_system:
        run_system_benchmark(system_json)

    render_path = None
    if args.render_video:
        render_path = (
            out_dir
            / "videos"
            / f"{video_path.stem}_{render_mode}_{int(args.render_altitude)}m.mp4"
        )

    runner = VideoBenchmarkRunner(
        csv_path,
        modes=modes,
        render_video_path=render_path,
        render_altitude_m=args.render_altitude,
        render_mode=render_mode,
        render_fps=args.render_fps,
        render_stride=args.render_stride,
    )
    runner._test_unload = args.test_unload

    if args.render_only:
        summary = {**info, **runner.render_full_video(video_path)}
    else:
        print(f"[BENCHMARK] altitudes={altitudes} modes={modes}")
        print(f"[BENCHMARK] CSV sampling: max_frames={max_frames} stride={stride}")
        summary = {
            **info,
            **runner.run_video(
                video_path,
                altitudes_m=altitudes,
                max_frames=max_frames,
                frame_stride=stride,
            ),
        }

    run_meta.write_text(
        json.dumps(
            {
                "video_info": info,
                "render_plan": render_plan,
                "csv_plan": csv_plan,
                "active_tool": normalize_tool(args.tool) if args.tool else None,
                "benchmark_modes": list(modes),
                "summary": summary,
                "lifecycle": summary.get("lifecycle"),
                "csv": str(csv_path),
                "system_benchmark": str(system_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if not args.render_only:
        print(f"\n[DONE] Metrics CSV: {csv_path}")
        print(f"[DONE] System JSON: {system_json}")
    if summary.get("render_video"):
        src = summary.get("duration_mmss") or info["duration_mmss"]
        out_d = summary.get("render_duration_s", 0)
        out_mm = f"{int(out_d)//60}:{int(round(out_d))%60:02d}"
        match = summary.get("duration_matches_input", False)
        tag = "matches input" if match else "check duration"
        print(
            f"[DONE] Review video: {summary['render_video']} "
            f"({out_mm}, {tag}; source was {src})"
        )
    if not args.render_only:
        print("[NEXT] python3 tools/plot_drone_benchmark_report.py")


if __name__ == "__main__":
    main()
