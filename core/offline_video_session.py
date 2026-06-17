"""Dashboard offline-video playback with per-video export folder (overlay + plots)."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

from core.tool_mapping import normalize_tool
from core.video_probe import probe_video

if TYPE_CHECKING:
    from core.video_benchmark import VideoBenchmarkRunner

_UPLOAD_ROOT = Path(__file__).resolve().parents[1] / "uploads" / "offline_videos"
_EXPORT_ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "results" / "dashboard"


class OfflineVideoSession:
    """Process uploaded MP4 with explicit tool; save overlay video + metrics + plots."""

    _running = False
    _video_path: Path | None = None
    _video_id: str | None = None
    _tool: str | None = None
    _altitude_m: float = 50.0
    _stride: int = 1
    _frame_idx: int = 0
    _cap: cv2.VideoCapture | None = None
    _info: dict | None = None
    _first_frame_ms: float | None = None
    _frames_processed: int = 0
    _runner: VideoBenchmarkRunner | None = None
    _output_dir: Path | None = None
    _export_info: dict | None = None
    _step_lock = threading.Lock()

    @classmethod
    def export_root(cls) -> Path:
        _EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
        return _EXPORT_ROOT

    @classmethod
    def upload_dir(cls) -> Path:
        _UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        return _UPLOAD_ROOT

    @classmethod
    def save_upload(cls, filename: str, data: bytes) -> dict:
        vid = Path(filename).stem.replace(" ", "_")[:48] or uuid.uuid4().hex[:12]
        dest = cls.upload_dir() / f"{vid}.mp4"
        if dest.exists():
            vid = f"{vid}_{uuid.uuid4().hex[:6]}"
            dest = cls.upload_dir() / f"{vid}.mp4"
        dest.write_bytes(data)
        info = probe_video(dest)
        return {"video_id": vid, "path": str(dest), **info}

    @classmethod
    def list_videos(cls) -> list[dict]:
        out = []
        for path in sorted(cls.upload_dir().glob("*.mp4")):
            try:
                info = probe_video(path)
                export_dir = cls.export_root() / path.stem
                out.append(
                    {
                        "video_id": path.stem,
                        "path": str(path),
                        "duration_mmss": info["duration_mmss"],
                        "frame_count": info["frame_count"],
                        "fps": info["fps"],
                        "has_export": export_dir.exists() and any(export_dir.glob("*.mp4")),
                        "export_dir": str(export_dir) if export_dir.exists() else None,
                    }
                )
            except OSError:
                continue
        return out

    @classmethod
    def get_export(cls, video_id: str) -> dict:
        out_dir = cls.export_root() / video_id
        if not out_dir.exists():
            return {"video_id": video_id, "exists": False}
        plots = sorted(p.name for p in out_dir.glob("*.png"))
        overlay = sorted(out_dir.glob("*_overlay.mp4"))
        return {
            "video_id": video_id,
            "exists": True,
            "output_dir": str(out_dir),
            "overlay_video": str(overlay[0]) if overlay else None,
            "metrics_csv": str(out_dir / "metrics.csv"),
            "report_html": str(out_dir / "report.html"),
            "plots": plots,
        }

    @classmethod
    def status(cls) -> dict:
        return {
            "running": cls._running,
            "video_id": cls._video_id,
            "active_tool": cls._tool or "idle",
            "frame_idx": cls._frame_idx,
            "frames_processed": cls._frames_processed,
            "total_frames": cls._info["frame_count"] if cls._info else 0,
            "altitude_m": cls._altitude_m,
            "stride": cls._stride,
            "first_frame_ms": cls._first_frame_ms,
            "input_source": "offline_video" if cls._running else "idle",
            "output_dir": str(cls._output_dir) if cls._output_dir else None,
        }

    @classmethod
    def _resolve_video(cls, video_id: str | None) -> Path:
        if not video_id:
            raise ValueError("video_id required")
        path = cls.upload_dir() / f"{video_id}.mp4"
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_id}")
        return path

    @classmethod
    def start(
        cls,
        *,
        video_id: str,
        tool: str,
        altitude_m: float = 50.0,
        stride: int = 1,
    ) -> dict:
        # Finalize any in-progress export (overlay + CSV + plots) before starting anew.
        cls.stop(finalize=True)
        normalized = normalize_tool(tool)
        if not normalized or normalized == "idle":
            raise ValueError(f"Invalid tool for offline run: {tool}")

        path = cls._resolve_video(video_id)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        info = probe_video(path)
        output_dir = cls.export_root() / video_id
        from core.video_benchmark import VideoBenchmarkRunner

        runner = VideoBenchmarkRunner(output_dir / "metrics.csv")
        export_info = runner.begin_streaming_export(
            output_dir,
            video_id=video_id,
            tool=normalized,
            altitude_m=float(altitude_m),
            native_fps=info["fps"],
            stride=max(1, int(stride)),
        )

        cls._cap = cap
        cls._video_path = path
        cls._video_id = video_id
        cls._tool = normalized
        cls._altitude_m = float(altitude_m)
        cls._stride = max(1, int(stride))
        cls._frame_idx = 0
        cls._frames_processed = 0
        cls._first_frame_ms = None
        cls._info = info
        cls._runner = runner
        cls._output_dir = output_dir
        cls._export_info = export_info
        cls._running = True

        print(
            f"[OFFLINE] started video={video_id} tool={normalized} "
            f"alt={cls._altitude_m:.0f}m stride={cls._stride} → {output_dir}"
        )
        return {
            **cls.status(),
            **export_info,
            "message": f"Offline export started ({normalized})",
        }

    @classmethod
    def stop(cls, *, finalize: bool = True, discard: bool = False) -> dict:
        export_result = None
        runner = cls._runner
        frames_done = cls._frames_processed
        should_finalize = (
            runner is not None
            and not discard
            and (finalize or frames_done > 0)
        )
        if should_finalize:
            export_result = runner.finish_streaming_export()
        elif runner is not None:
            runner._close_video_writer()

        if cls._cap is not None:
            cls._cap.release()
        previous = cls._tool
        cls._cap = None
        cls._running = False
        cls._video_path = None
        cls._video_id = None
        cls._tool = None
        cls._info = None
        cls._runner = None
        out_dir = cls._output_dir
        cls._output_dir = None
        cls._export_info = None

        if previous:
            print(f"[OFFLINE] stopped (was {previous})")
        result = {
            **cls.status(),
            "message": "Offline video session stopped",
            "active_tool": "idle",
        }
        if export_result:
            result["export"] = export_result
            result["message"] = f"Export complete → {export_result['output_dir']}"
            result["output_dir"] = export_result["output_dir"]
        elif out_dir:
            result["output_dir"] = str(out_dir)
        return result

    @classmethod
    def _read_frame(cls) -> tuple[bool, any]:
        if cls._cap is None or not cls._running:
            return False, None

        while True:
            ret, frame = cls._cap.read()
            if not ret:
                return False, None
            if cls._frame_idx % cls._stride == 0:
                idx = cls._frame_idx
                cls._frame_idx += 1
                return True, (frame, idx)
            cls._frame_idx += 1

    @classmethod
    def step(cls) -> dict:
        with cls._step_lock:
            if not cls._running or cls._runner is None:
                return {
                    "status": "idle",
                    "active_tool": "idle",
                    "message": "No offline video session — upload video and start with a tool",
                    "input_source": "idle",
                }

            ret, payload = cls._read_frame()
            if not ret:
                done = cls.stop(finalize=True)
                done["eof"] = True
                return done

            frame, frame_idx = payload
            t0 = time.perf_counter()
            try:
                result = cls._runner.stream_export_frame(
                    frame,
                    frame_idx,
                    cls._video_path.name if cls._video_path else "",
                )
            except Exception as exc:
                print(f"[OFFLINE] frame {frame_idx} failed: {exc}")
                done = cls.stop(finalize=True)
                done["error"] = str(exc)
                done["eof"] = True
                return done

            wall_ms = (time.perf_counter() - t0) * 1000.0
            if cls._first_frame_ms is None:
                cls._first_frame_ms = round(wall_ms, 2)
            cls._frames_processed += 1

            total_frames = cls._info["frame_count"] if cls._info else 0
            processed_target = max(1, (total_frames + cls._stride - 1) // cls._stride)
            result["total_frames"] = total_frames
            result["offline"] = {
                "video_id": cls._video_id,
                "frame_idx": frame_idx,
                "altitude_m": cls._altitude_m,
                "wall_ms": round(wall_ms, 2),
                "first_frame_ms": cls._first_frame_ms,
                "frames_processed": cls._frames_processed,
                "frames_target": processed_target,
                "output_dir": str(cls._output_dir),
                "overlay_video": cls._export_info.get("overlay_video")
                if cls._export_info
                else None,
            }
            result["camera"] = {"device": f"offline:{cls._video_id}"}
            return result

    @classmethod
    def delete_video(cls, video_id: str) -> dict:
        if cls._running and cls._video_id == video_id:
            cls.stop(finalize=True)
        path = cls.upload_dir() / f"{video_id}.mp4"
        if path.exists():
            path.unlink()
            return {"deleted": video_id}
        raise FileNotFoundError(video_id)
