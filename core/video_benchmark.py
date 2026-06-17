"""Offline drone-video benchmark runner (non-realtime)."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from core.benchmark_lifecycle import BenchmarkLifecycle
from core.benchmark_metrics import append_benchmark_row, ensure_benchmark_csv
from core.benchmark_overlay import compose_benchmark_frame, compose_export_frame
from core.flood_grid import serialize_grid_analysis
from core.gpu_runtime import sync_all
from core.model_manager import ModelManager
from core.power_monitor import get_power_monitor
from core.model_selector import FloodModelSelector
from core.segment_policy import should_run_segmentation
from core.tool_mapping import mode_to_tool, normalize_mode, tool_to_mode
from core.video_geometry import REF_ALTITUDE_M, simulate_altitude_frame
from core.video_probe import probe_video

_process = psutil.Process(os.getpid())


class VideoBenchmarkRunner:
    MODES = ("flood", "human", "combined")

    def __init__(
        self,
        output_csv: Path,
        modes: tuple[str, ...] = ("flood", "combined"),
        warmup_frames: int = 5,
        render_video_path: Path | None = None,
        render_altitude_m: float | None = None,
        render_mode: str | None = None,
        render_fps: float = 0.0,
        render_stride: int = 1,
    ):
        self.output_csv = Path(output_csv)
        self.modes = modes
        self.warmup_frames = warmup_frames
        self.render_video_path = Path(render_video_path) if render_video_path else None
        self.render_altitude_m = render_altitude_m
        self.render_mode = render_mode
        self.render_fps = render_fps
        self.render_stride = max(1, int(render_stride))
        self._video_writer: cv2.VideoWriter | None = None

        self.model_manager = ModelManager()
        self.selector = FloodModelSelector()
        self._last_flood_ratio = 0.0
        self._last_mask = None
        self._frame_count = 0
        self._force_segment = False

        self._clf_load_ms = 0.0
        self._seg_load_ms = 0.0
        self._human_load_ms = 0.0
        self._lifecycle_by_mode: dict[str, BenchmarkLifecycle] = {}
        self._session_first_frame = True
        self._test_unload = False
        self._power = None

    def _sample_power_metrics(self) -> dict:
        try:
            if self._power is None:
                self._power = get_power_monitor()
            return self._power.record_inference_power()
        except Exception:
            return {
                "idle_power_w": 0.0,
                "inference_power_w": 0.0,
                "extra_power_w": 0.0,
                "peak_inference_power_w": 0.0,
                "peak_extra_power_w": 0.0,
                "power_source": "tegrastats_vdd_in",
            }

    def _lifecycle(self, mode: str) -> BenchmarkLifecycle:
        mode = normalize_mode(mode) or mode
        if mode not in self._lifecycle_by_mode:
            lc = BenchmarkLifecycle(
                active_tool=mode_to_tool(mode) or mode,
                benchmark_mode=mode,
            )
            self._lifecycle_by_mode[mode] = lc
        return self._lifecycle_by_mode[mode]

    def _load_models(self, mode: str) -> None:
        mode = normalize_mode(mode) or mode
        active_tool = mode_to_tool(mode) or mode
        print(f"[BENCHMARK] explicit tool={active_tool} mode={mode} (not auto-activated live task)")

        t0 = time.perf_counter()
        self.clf_model = self.model_manager.load_flood_classifier()
        clf_ms = (time.perf_counter() - t0) * 1000.0
        self._clf_load_ms = clf_ms

        seg_ms = 0.0
        self.seg_model = None
        if mode in ("flood", "combined"):
            t1 = time.perf_counter()
            self.seg_model = self.model_manager.load_flood_segmenter()
            seg_ms = (time.perf_counter() - t1) * 1000.0
            self._seg_load_ms = seg_ms

        human_ms = 0.0
        self.human_model = None
        if mode in ("human", "combined"):
            t2 = time.perf_counter()
            self.human_model = self.model_manager.load_human_detector()
            human_ms = (time.perf_counter() - t2) * 1000.0
            self._human_load_ms = human_ms

        self._lifecycle(mode).record_load(
            clf_ms=clf_ms, seg_ms=seg_ms, human_ms=human_ms
        )

        from core.inference_engine import InferenceEngine

        self.engine = InferenceEngine()

    def _maybe_test_unload(self, mode: str) -> None:
        if not self._test_unload:
            return
        lc = self._lifecycle(mode)
        for name in ("human_detector", "flood_segmenter", "flood_classifier"):
            ms = self.model_manager.unload_model(name) * 1000.0
            if ms:
                lc.record_unload(name, ms)

    def _run_flood(self, frame: np.ndarray) -> dict:
        self._frame_count += 1
        context = {"flood_ratio": self._last_flood_ratio, "battery": 100, "cpu_usage": 40}
        pre = self.selector.select_models(context)
        switches = self.selector.apply_selection(pre)

        clf_only = self.engine.run_flood_pipeline(
            self.clf_model, None, frame, run_segmenter=False
        )
        clf_pred = clf_only["clf_class_index"]
        run_seg = self._force_segment or should_run_segmentation(
            self._frame_count, clf_pred, self._last_flood_ratio, True
        )

        switch_occurred = bool(switches.get("primary"))
        seg_skipped = not run_seg

        raw_label = clf_only["classification"]["label"]
        mask = None

        if run_seg and self.seg_model is not None:
            seg_out = self.engine.run_segmentation(self.seg_model, frame)
            seg_ms = seg_out.pop("inference_ms", 0.0)
            mask = seg_out.get("mask")
            flood_ratio = float(seg_out.get("flood_ratio", 0.0))
            clf_ms = clf_only["classification_ms"]
            if mask is not None:
                self._last_mask = mask
            if flood_ratio > 0 or run_seg:
                self._last_flood_ratio = flood_ratio
            context["flood_ratio"] = flood_ratio
            post = self.selector.select_models(context)
            post_sw = self.selector.apply_selection(post)
            if post_sw.get("primary"):
                switch_occurred = True
            total_ms = clf_ms + seg_ms
        else:
            clf_ms = clf_only["classification_ms"]
            seg_ms = 0.0
            flood_ratio = self._last_flood_ratio
            mask = self._last_mask
            total_ms = clf_ms

        seg_active = self.selector.is_segmentation_active(flood_ratio)
        display_label = "Flooded" if seg_active else "Non-Flooded"
        primary = self.selector.primary_model
        return {
            "classification_ms": round(clf_ms, 2),
            "segmentation_ms": round(seg_ms, 2),
            "total_inference_ms": round(total_ms, 2),
            "total_latency_ms": round(total_ms, 2),
            "flood_ratio": round(flood_ratio, 4),
            "classification_label": display_label,
            "raw_classification_label": raw_label,
            "segmentation_active": seg_active,
            "primary_model": primary,
            "model_switch_occurred": switch_occurred,
            "segmentation_skipped": seg_skipped,
            "human_count": 0,
            "_mask": mask,
        }

    def _run_human(self, frame: np.ndarray) -> dict:
        from tools.detect_human import run_human_inference

        humans, infer_ms, _ = run_human_inference(frame)
        return {
            "classification_ms": 0.0,
            "segmentation_ms": 0.0,
            "total_inference_ms": round(infer_ms, 2),
            "total_latency_ms": round(infer_ms, 2),
            "flood_ratio": 0.0,
            "classification_label": "n/a",
            "primary_model": "yolov8n",
            "model_switch_occurred": False,
            "segmentation_skipped": True,
            "human_count": len(humans),
            "_humans": humans,
            "_mask": None,
        }

    def _run_combined(self, frame: np.ndarray) -> dict:
        flood = self._run_flood(frame)
        from tools.detect_human import run_human_inference

        humans, infer_ms, _ = run_human_inference(frame)
        total = flood["total_inference_ms"] + infer_ms
        return {
            **flood,
            "total_inference_ms": round(total, 2),
            "total_latency_ms": round(total, 2),
            "human_count": len(humans),
            "_humans": humans,
            "mode_note": "flood+human sequential on same frame",
        }

    def _ensure_video_writer(self, frame: np.ndarray, fps: float) -> None:
        if self._video_writer is not None or self.render_video_path is None:
            return
        self.render_video_path.parent.mkdir(parents=True, exist_ok=True)
        if self.render_video_path.exists():
            self.render_video_path.unlink()

        h, w = frame.shape[:2]
        for fourcc_name in ("mp4v", "avc1", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            ext = ".avi" if fourcc_name == "MJPG" else ".mp4"
            path = self.render_video_path.with_suffix(ext)
            writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
            if writer.isOpened():
                self._video_writer = writer
                self.render_video_path = path
                return
            writer.release()

        raise RuntimeError(f"Cannot open video writer: {self.render_video_path}")

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None
            print(f"[RENDER] Wrote {self.render_video_path}")

    def _record_metrics_row(
        self,
        out: dict,
        wall_ms: float,
        *,
        mode: str,
        video_id: str,
        video_name: str,
        frame_idx: int,
        altitude_m: float,
        geom_meta: dict,
        is_warmup: bool,
        power_metrics: dict | None = None,
    ) -> None:
        """Append one CSV row from a single inference result (no re-inference)."""
        instant_fps = 1000.0 / wall_ms if wall_ms > 0 else 0.0
        memory_mb = _process.memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=None)
        active_tool = mode_to_tool(mode) or mode
        self._lifecycle(mode).record_frame_latency(wall_ms, is_warmup=is_warmup)
        power = power_metrics or self._sample_power_metrics()
        infer_power = float(power.get("inference_power_w", 0) or 0)
        peak_power = float(power.get("peak_inference_power_w", infer_power) or infer_power)

        row = {
            "timestamp": time.time(),
            "video_id": video_id,
            "video_name": video_name,
            "frame_idx": frame_idx,
            "sim_altitude_m": altitude_m,
            "input_width": geom_meta["output_width"],
            "input_height": geom_meta["output_height"],
            "mode": mode,
            "active_tool": active_tool,
            "classification_label": out.get("classification_label"),
            "primary_model": out.get("primary_model"),
            "model_switch_occurred": out.get("model_switch_occurred", False),
            "segmentation_skipped": out.get("segmentation_skipped", False),
            "human_count": out.get("human_count", 0),
            "clf_backend": self.model_manager.clf_backend,
            "seg_backend": self.model_manager.seg_backend,
            "human_backend": self.model_manager.human_backend,
            "fps": round(instant_fps, 2),
            "instant_fps": round(instant_fps, 2),
            "total_inference_ms": out["total_inference_ms"],
            "total_latency_ms": round(wall_ms, 2),
            "classification_ms": out["classification_ms"],
            "segmentation_ms": out["segmentation_ms"],
            "model_switch_latency_ms": out["classification_ms"] + out["segmentation_ms"],
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
            "power_w": infer_power,
            "peak_memory_mb": round(memory_mb, 2),
            "peak_cpu_percent": round(cpu_percent, 2),
            "peak_power_w": peak_power,
            "clf_load_ms": round(self._clf_load_ms, 2),
            "seg_load_ms": round(self._seg_load_ms, 2),
            "human_load_ms": round(self._human_load_ms, 2),
            "model_load_total_ms": round(
                self._clf_load_ms + self._seg_load_ms + self._human_load_ms, 2
            ),
            "session_first_frame": self._session_first_frame and not is_warmup,
            "flood_ratio": out["flood_ratio"],
            "warmup": is_warmup,
        }
        if not is_warmup:
            self._session_first_frame = False
        append_benchmark_row(self.output_csv, row)

    def _process_frame(
        self,
        frame: np.ndarray,
        mode: str,
        video_id: str,
        video_name: str,
        frame_idx: int,
        altitude_m: float,
        geom_meta: dict,
        is_warmup: bool,
    ) -> dict:
        t0 = time.perf_counter()
        out = self._infer_frame(frame, mode)
        sync_all()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        power_metrics = self._sample_power_metrics()
        self._record_metrics_row(
            out,
            wall_ms,
            mode=mode,
            video_id=video_id,
            video_name=video_name,
            frame_idx=frame_idx,
            altitude_m=altitude_m,
            geom_meta=geom_meta,
            is_warmup=is_warmup,
            power_metrics=power_metrics,
        )
        return out

    def _metrics_from_inference(
        self,
        frame: np.ndarray,
        mode: str,
        out: dict,
        wall_ms: float,
        *,
        power_metrics: dict | None = None,
    ) -> dict:
        infer_ms = out["total_inference_ms"]
        instant_fps = 1000.0 / wall_ms if wall_ms > 0 else 0.0
        memory_mb = _process.memory_info().rss / (1024 * 1024)
        cpu_percent = psutil.cpu_percent(interval=None)
        power = power_metrics or self._sample_power_metrics()
        return {
            "classification_label": out.get("classification_label"),
            "raw_classification_label": out.get("raw_classification_label"),
            "segmentation_active": out.get("segmentation_active", False),
            "primary_model": out.get("primary_model"),
            "model_switch_occurred": out.get("model_switch_occurred", False),
            "segmentation_skipped": out.get("segmentation_skipped", False),
            "human_count": out.get("human_count", 0),
            "fps": round(instant_fps, 2),
            "total_inference_ms": out["total_inference_ms"],
            "total_latency_ms": round(wall_ms, 2),
            "classification_ms": out["classification_ms"],
            "segmentation_ms": out["segmentation_ms"],
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
            "flood_ratio": out["flood_ratio"],
            "warmup": False,
            "idle_power_w": power.get("idle_power_w", 0),
            "inference_power_w": power.get("inference_power_w", 0),
            "extra_power_w": power.get("extra_power_w", 0),
            "peak_inference_power_w": power.get("peak_inference_power_w", 0),
            "peak_extra_power_w": power.get("peak_extra_power_w", 0),
            "power_w": power.get("inference_power_w", 0),
            "peak_power_w": power.get("peak_inference_power_w", 0),
        }

    def _infer_frame(self, frame: np.ndarray, mode: str) -> dict:
        if mode == "flood":
            return self._run_flood(frame)
        if mode == "human":
            return self._run_human(frame)
        return self._run_combined(frame)

    def begin_streaming_export(
        self,
        output_dir: Path,
        *,
        video_id: str,
        tool: str,
        altitude_m: float,
        native_fps: float,
        stride: int = 1,
    ) -> dict:
        """Dashboard offline: one folder per video with overlay MP4 + metrics CSV."""
        mode = normalize_mode(tool_to_mode(tool) or tool) or "combined"
        active_tool = mode_to_tool(mode) or tool
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        out_fps = native_fps / max(1, stride)
        overlay_name = f"{video_id}_{mode}_{int(altitude_m)}m_overlay.mp4"

        self.output_csv = output_dir / "metrics.csv"
        ensure_benchmark_csv(self.output_csv, reset=True)
        self.render_video_path = output_dir / overlay_name
        self.render_mode = mode
        self.render_altitude_m = altitude_m
        self.render_fps = out_fps
        self._video_writer = None
        self._frame_count = 0
        self._last_flood_ratio = 0.0
        self._last_mask = None
        self._session_first_frame = True
        self._force_segment = True
        self._export_video_id = video_id
        self._export_stride = max(1, int(stride))
        # Dashboard offline: save/preview clean video; metrics live in portal + CSV/plots.
        self._export_clean_video = True
        self._power = get_power_monitor()

        self._load_models(mode)
        print(
            f"[DASHBOARD EXPORT] dir={output_dir} tool={active_tool} "
            f"mode={mode} alt={altitude_m:.0f}m fps={out_fps:.2f}"
        )
        return {
            "output_dir": str(output_dir),
            "overlay_video": str(self.render_video_path),
            "metrics_csv": str(self.output_csv),
            "active_tool": active_tool,
            "benchmark_mode": mode,
        }

    def stream_export_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        video_name: str,
    ) -> dict:
        """Process one frame: infer, write viz MP4 + CSV, return dashboard preview (metrics in JSON)."""
        mode = self.render_mode or "combined"
        altitude_m = self.render_altitude_m or REF_ALTITUDE_M
        video_id = getattr(self, "_export_video_id", "video")

        sim_frame, geom_meta = simulate_altitude_frame(frame_bgr, altitude_m)
        t0 = time.perf_counter()
        out = self._infer_frame(sim_frame, mode)
        sync_all()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        power_metrics = self._sample_power_metrics()

        self._record_metrics_row(
            out,
            wall_ms,
            mode=mode,
            video_id=video_id,
            video_name=video_name,
            frame_idx=frame_idx,
            altitude_m=altitude_m,
            geom_meta=geom_meta,
            is_warmup=False,
            power_metrics=power_metrics,
        )
        metrics = self._metrics_from_inference(
            sim_frame, mode, out, wall_ms, power_metrics=power_metrics
        )
        grid_analysis = None
        if getattr(self, "_export_clean_video", False):
            vis, grid_raw = compose_export_frame(
                sim_frame,
                mode=mode,
                mask=out.get("_mask"),
                humans=out.get("_humans"),
                drone_altitude_m=altitude_m,
            )
            grid_analysis = serialize_grid_analysis(grid_raw)
        else:
            vis = compose_benchmark_frame(
                sim_frame,
                mode=mode,
                metrics=metrics,
                geom_meta=geom_meta,
                frame_idx=frame_idx,
                mask=out.get("_mask"),
                humans=out.get("_humans"),
            )
        self._ensure_video_writer(vis, self.render_fps or 25.0)
        if self._video_writer is not None:
            self._video_writer.write(vis)

        _, buffer = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frame_b64 = base64.b64encode(buffer).decode("utf-8")
        active_tool = mode_to_tool(mode) or mode
        flood_ratio = float(out.get("flood_ratio", 0) or 0)
        human_count = int(out.get("human_count", 0) or 0)
        if flood_ratio >= 0.7:
            status = "CRITICAL"
        elif flood_ratio >= 0.4:
            status = "ALERT"
        elif flood_ratio >= 0.2:
            status = "WARNING"
        elif human_count > 0:
            status = "ALERT"
        else:
            status = "NORMAL"

        return {
            "active_tool": active_tool,
            "active_tools": (
                ["detect_flood", "detect_human"]
                if mode == "combined"
                else [active_tool]
            ),
            "classification": {
                "label": out.get("classification_label"),
                "raw_label": out.get("raw_classification_label"),
            },
            "segmentation": {"flood_ratio": out.get("flood_ratio", 0)},
            "segmentation_active": out.get("segmentation_active", False),
            "primary_model": out.get("primary_model"),
            "human_count": out.get("human_count", 0),
            "humans": out.get("_humans") or [],
            "metrics": metrics,
            "power": power_metrics,
            "grid": grid_analysis,
            "overlay_mode": "grid" if grid_analysis else "none",
            "system": {
                "status": status,
                "fps": metrics.get("fps", 0),
                "latency_ms": metrics.get("total_latency_ms", 0),
            },
            "frame_base64": frame_b64,
            "frame": frame_b64,
            "input_source": "offline_video",
        }

    def finish_streaming_export(self) -> dict:
        """Close overlay video and generate plots/report in the output folder."""
        self._close_video_writer()
        self._force_segment = False
        mode = self.render_mode or "combined"
        lc = self._lifecycle(mode).summary()
        out_dir = self.output_csv.parent if self.output_csv else None
        if not out_dir:
            return {"error": "no export session"}

        meta = {
            "video_id": getattr(self, "_export_video_id", ""),
            "active_tool": mode_to_tool(mode),
            "benchmark_mode": mode,
            "altitude_m": self.render_altitude_m,
            "stride": getattr(self, "_export_stride", 1),
            "overlay_video": str(self.render_video_path),
            "metrics_csv": str(self.output_csv),
            "lifecycle": lc,
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        from core.benchmark_report import generate_benchmark_report

        report = generate_benchmark_report(
            self.output_csv, out_dir, lifecycle=lc
        )
        print(f"[DASHBOARD EXPORT] done → {out_dir}")
        return {
            "output_dir": str(out_dir),
            "overlay_video": str(self.render_video_path),
            "metrics_csv": str(self.output_csv),
            "report_html": str(out_dir / "report.html"),
            "report_md": str(out_dir / "REPORT.md"),
            "lifecycle": lc,
            "summary": report,
        }

    def render_full_video(self, video_path: Path) -> dict:
        """Render every source frame (full duration) with grid + metrics overlay."""
        if not self.render_video_path or not self.render_mode:
            return {}

        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        info = probe_video(video_path)
        native_fps = info["fps"]
        total_frames = info["frame_count"]
        out_fps = self.render_fps if self.render_fps > 0 else native_fps
        altitude_m = self.render_altitude_m or REF_ALTITUDE_M
        mode = self.render_mode
        video_id = video_path.stem
        source_duration_s = info["duration_s"]

        print(
            f"\n[RENDER] full video mode={mode} alt={altitude_m:.0f}m "
            f"source={info['duration_mmss']} ({source_duration_s}s) "
            f"frames={total_frames} stride={self.render_stride} fps={out_fps:.2f}"
        )

        self._load_models(mode)
        self._frame_count = 0
        self._last_flood_ratio = 0.0
        self._last_mask = None
        self._session_first_frame = True
        self._force_segment = True
        self._video_writer = None

        frame_idx = 0
        written = 0
        t_start = time.perf_counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.render_stride != 0:
                frame_idx += 1
                continue

            sim_frame, geom_meta = simulate_altitude_frame(frame, altitude_m)
            t0 = time.perf_counter()
            out = self._infer_frame(sim_frame, mode)
            sync_all()
            wall_ms = (time.perf_counter() - t0) * 1000.0
            metrics = self._metrics_from_inference(sim_frame, mode, out, wall_ms)

            vis = compose_benchmark_frame(
                sim_frame,
                mode=mode,
                metrics=metrics,
                geom_meta=geom_meta,
                frame_idx=frame_idx,
                mask=out.get("_mask"),
                humans=out.get("_humans"),
            )
            self._ensure_video_writer(vis, out_fps)
            if self._video_writer is not None:
                self._video_writer.write(vis)
                written += 1

            if written == 1 or written % 50 == 0:
                elapsed = time.perf_counter() - t_start
                print(
                    f"[RENDER] frame {frame_idx}/{total_frames} "
                    f"written={written} elapsed={elapsed:.0f}s"
                )

            frame_idx += 1

        cap.release()
        self._close_video_writer()
        self._force_segment = False
        duration_s = written / out_fps if out_fps > 0 else 0.0
        delta = abs(duration_s - source_duration_s)
        match = self.render_stride == 1 and delta < 1.5
        print(
            f"[RENDER] done: {written} frames → {duration_s:.1f}s "
            f"(source {source_duration_s:.1f}s) at {out_fps:.1f} fps"
        )
        if not match:
            print(
                f"[RENDER] WARNING: output duration differs from input by {delta:.1f}s. "
                "Use --render-stride 1 and --render-fps 0 to match input length."
            )
        lc = self._lifecycle(mode).summary()
        self._maybe_test_unload(mode)
        return {
            "render_video": str(self.render_video_path),
            "render_frames_written": written,
            "render_fps": out_fps,
            "render_duration_s": round(duration_s, 2),
            "source_duration_s": source_duration_s,
            "source_frames": total_frames,
            "duration_matches_input": match,
            "video_id": video_id,
            "active_tool": mode_to_tool(mode),
            "lifecycle": lc,
        }

    def run_video(
        self,
        video_path: Path,
        altitudes_m: list[float],
        max_frames: int = 100,
        frame_stride: int = 5,
    ) -> dict:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(video_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps_native = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_id = video_path.stem
        summary = {
            "video": str(video_path),
            "video_id": video_id,
            "native_fps": fps_native,
            "total_frames": total_frames,
            "ref_altitude_m": REF_ALTITUDE_M,
            "altitudes_m": altitudes_m,
            "modes": list(self.modes),
        }

        for mode in self.modes:
            print(f"\n[VIDEO BENCHMARK] mode={mode} video={video_id}")
            self._load_models(mode)
            self._frame_count = 0
            self._last_flood_ratio = 0.0
            self._last_mask = None
            self._session_first_frame = True

            frame_idx = 0
            processed = 0
            warmup_done = 0

            while processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                for altitude_m in altitudes_m:
                    sim_frame, geom_meta = simulate_altitude_frame(frame, altitude_m)
                    is_warmup = warmup_done < self.warmup_frames
                    self._process_frame(
                        sim_frame,
                        mode,
                        video_id,
                        video_path.name,
                        frame_idx,
                        altitude_m,
                        geom_meta,
                        is_warmup,
                    )
                    if is_warmup:
                        warmup_done += 1
                    else:
                        processed += 1

                frame_idx += 1

            self._maybe_test_unload(mode)
            lc = self._lifecycle(mode).summary()
            print(
                f"[BENCHMARK] lifecycle tool={lc['active_tool']} "
                f"load={lc['model_load_total_ms']}ms "
                f"reloads={lc['model_reload_events']} "
                f"first_frame={lc['first_frame_latency_ms']}ms "
                f"steady_p50={lc['steady_latency_ms']['p50']}ms"
            )

        cap.release()
        summary["lifecycle"] = {
            m: lc.summary() for m, lc in self._lifecycle_by_mode.items()
        }

        if self.render_video_path:
            render_summary = self.render_full_video(video_path)
            summary.update(render_summary)

        return summary
