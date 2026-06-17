"""Compose benchmark frames with flood grid + per-frame metrics for review videos."""

from __future__ import annotations

import cv2
import numpy as np

from core.flood_grid import draw_grid_overlay
from core.model_selector import FloodModelSelector

_FLOOD_THRESHOLD = FloodModelSelector.FLOOD_RATIO_THRESHOLD

# BGR palette — consistent technical HUD
_C_BG = (18, 22, 28)
_C_BORDER = (60, 140, 180)
_C_TITLE = (200, 230, 255)
_C_LABEL = (140, 170, 200)
_C_VALUE = (235, 245, 255)
_C_DIM = (110, 125, 145)
_C_OK = (80, 220, 120)
_C_WARN = (80, 200, 255)
_C_ALERT = (80, 80, 255)
_C_ACCENT = (255, 200, 80)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_MONO = cv2.FONT_HERSHEY_DUPLEX


def _flood_status(flood_ratio: float) -> tuple[str, bool]:
    ratio = float(flood_ratio or 0)
    flooded = ratio >= _FLOOD_THRESHOLD
    return ("FLOODED" if flooded else "DRY", flooded)


def _text_size(text: str, scale: float, thickness: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    return w, h


def _draw_filled_rect(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    alpha: float = 0.82,
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _C_BG, -1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), _C_BORDER, 1)
    return cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0)


def _put(
    frame: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.42,
    color: tuple[int, int, int] = _C_VALUE,
    thickness: int = 1,
    font=_FONT,
) -> None:
    cv2.putText(frame, text, xy, font, scale, color, thickness, cv2.LINE_AA)


def _draw_kv_rows(
    frame: np.ndarray,
    x: int,
    y: int,
    rows: list[tuple[str, str, tuple[int, int, int] | None]],
    *,
    label_w: int = 108,
    row_h: int = 18,
    scale: float = 0.42,
) -> int:
    """Aligned label | value rows. Returns y after last row."""
    for label, value, vcolor in rows:
        _put(frame, label, (x, y), scale=scale, color=_C_LABEL, thickness=1)
        _put(
            frame,
            value,
            (x + label_w, y),
            scale=scale,
            color=vcolor or _C_VALUE,
            thickness=1,
        )
        y += row_h
    return y


def _draw_section_title(frame: np.ndarray, x: int, y: int, title: str) -> int:
    _put(frame, title, (x, y), scale=0.38, color=_C_TITLE, thickness=1)
    tw, _ = _text_size(title, 0.38)
    cv2.line(frame, (x, y + 4), (x + tw, y + 4), _C_BORDER, 1)
    return y + 16


def _draw_header_bar(
    frame: np.ndarray,
    *,
    mode: str,
    frame_idx: int,
    altitude: float,
    placement: str,
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 26
    out = _draw_filled_rect(frame, 0, 0, w, bar_h, alpha=0.88)
    _put(out, "BENCHMARK REVIEW", (8, 18), scale=0.45, color=_C_ACCENT, thickness=1)
    fields = [
        ("MODE", mode.upper()),
        ("FRAME", f"{frame_idx:05d}"),
        ("ALT", f"{altitude:.0f}m"),
        ("VIEW", placement.upper()),
    ]
    x = 200
    for key, val in fields:
        _put(out, f"{key}:", (x, 18), scale=0.38, color=_C_LABEL)
        x += 42
        _put(out, val, (x, 18), scale=0.38, color=_C_VALUE)
        x += 72
    return out


def _draw_detection_panel(
    frame: np.ndarray,
    *,
    display_label: str,
    flood_ratio: float,
    seg_active: bool,
    primary: str,
    raw_label: str,
    human_count: int | None,
    gps_text: str | None,
    mode: str,
    seg_skipped: bool,
    model_switch: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0 = 0, 26
    panel_w = 272
    panel_h = min(210, h - 70)
    out = _draw_filled_rect(frame, x0, y0, x0 + panel_w, y0 + panel_h, alpha=0.78)

    x, y = 10, y0 + 18
    y = _draw_section_title(out, x, y, "DETECTION")
    status_color = _C_ALERT if display_label == "FLOODED" else _C_OK
    y = _draw_kv_rows(
        out,
        x,
        y,
        [
            ("STATUS", display_label, status_color),
            ("FLOOD_RATIO", f"{flood_ratio:.3f}", status_color),
            ("SEGMENTATION", "ACTIVE" if seg_active else "STANDBY", _C_WARN if seg_active else _C_DIM),
            ("PRIMARY", primary or "n/a", _C_VALUE),
            ("SEG_SKIP", "YES" if seg_skipped else "NO", _C_DIM),
            ("MODEL_SW", "YES" if model_switch else "NO", _C_DIM),
        ],
    )

    resnet_says_flood = raw_label.lower().startswith("flood")
    if raw_label and resnet_says_flood != (display_label == "FLOODED"):
        y += 4
        y = _draw_section_title(out, x, y, "CLASSIFIER")
        _draw_kv_rows(
            out,
            x,
            y,
            [("RESNET_RAW", raw_label.upper(), _C_DIM)],
        )

    if mode in ("human", "combined") and human_count is not None:
        y += 4
        y = _draw_section_title(out, x, y, "RESCUE")
        y = _draw_kv_rows(
            out,
            x,
            y,
            [("HUMANS", str(human_count), _C_WARN if human_count else _C_DIM)],
        )

    if gps_text:
        y += 4
        y = _draw_section_title(out, x, y, "LOCALIZATION")
        gps_val = gps_text.replace("GPS:", "").strip()
        _put(out, gps_val, (x, y), scale=0.36, color=_C_WARN)

    return out


def _draw_perf_panel(
    frame: np.ndarray,
    *,
    fps: float,
    latency_ms: float,
    infer_ms: float,
    clf_ms: float,
    seg_ms: float,
    mem_mb: float,
    cpu_pct: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w = 188
    x1 = w
    x0 = x1 - panel_w
    y0, y1 = 26, 26 + 168
    out = _draw_filled_rect(frame, x0, y0, x1, y1, alpha=0.78)

    x, y = x0 + 10, y0 + 18
    y = _draw_section_title(out, x, y, "PERFORMANCE")

    fps_color = _C_OK if fps >= 20 else _C_WARN if fps >= 10 else _C_ALERT
    lat_color = _C_OK if latency_ms < 50 else _C_WARN if latency_ms < 100 else _C_ALERT

    _draw_kv_rows(
        out,
        x,
        y,
        [
            ("FPS", f"{fps:6.1f}", fps_color),
            ("LATENCY", f"{latency_ms:5.0f} ms", lat_color),
            ("INFER", f"{infer_ms:5.0f} ms", _C_VALUE),
            ("CLF", f"{clf_ms:5.0f} ms", _C_VALUE),
            ("SEG", f"{seg_ms:5.0f} ms", _C_VALUE),
            ("MEM", f"{mem_mb:5.0f} MB", _C_VALUE),
            ("CPU", f"{cpu_pct:4.0f} %", _C_VALUE),
        ],
        label_w=88,
    )
    return out


def _draw_footer_bar(
    frame: np.ndarray,
    *,
    display_label: str,
    flood_ratio: float,
    human_count: int | None,
    mode: str,
    warmup: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    bar_h = 36
    y0 = h - bar_h
    flooded = display_label == "FLOODED"
    accent = _C_ALERT if flooded else _C_OK

    out = frame.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), _C_BG, -1)
    cv2.line(overlay, (0, y0), (w, y0), accent, 2)
    out = cv2.addWeighted(overlay, 0.86, out, 0.14, 0)

    segments = [
        f"FLOOD: {display_label}",
        f"RATIO: {flood_ratio:.1%}",
    ]
    if mode in ("human", "combined") and human_count is not None:
        segments.append(f"HUMANS: {human_count}")
    if warmup:
        segments.append("WARMUP")

    text = "  |  ".join(segments)
    _put(out, text, (10, h - 10), scale=0.48, color=accent, thickness=1)

    # Ratio bar (technical progress indicator)
    bar_x0, bar_x1 = w - 130, w - 12
    bar_y0, bar_y1 = y0 + 10, y0 + 22
    cv2.rectangle(out, (bar_x0, bar_y0), (bar_x1, bar_y1), _C_DIM, 1)
    fill_w = int((bar_x1 - bar_x0 - 2) * min(1.0, flood_ratio))
    if fill_w > 0:
        cv2.rectangle(out, (bar_x0 + 1, bar_y0 + 1), (bar_x0 + 1 + fill_w, bar_y1 - 1), accent, -1)
    _put(out, "WATER", (bar_x0, bar_y0 - 2), scale=0.32, color=_C_LABEL)

    return out


def _draw_cell_ratios(
    frame: np.ndarray,
    cell_ratios: list[list[float]],
    grid_size: int = 4,
) -> np.ndarray:
    out = frame
    h, w = out.shape[:2]
    cell_h, cell_w = h // grid_size, w // grid_size
    for row_i, row in enumerate(cell_ratios):
        for col_i, ratio in enumerate(row):
            label = f"{ratio:.2f}"
            tw, th = _text_size(label, 0.38, 1)
            cx = col_i * cell_w + (cell_w - tw) // 2
            cy = row_i * cell_h + (cell_h + th) // 2
            color = _C_ALERT if ratio >= 0.30 else _C_DIM
            # Small backing plate for readability
            cv2.rectangle(
                out,
                (cx - 2, cy - th - 2),
                (cx + tw + 2, cy + 2),
                (0, 0, 0),
                -1,
            )
            _put(out, label, (cx, cy), scale=0.38, color=color, font=_FONT_MONO)
    return out


def _draw_human_boxes(frame: np.ndarray, humans: list[dict]) -> np.ndarray:
    """Bounding boxes only — no duplicate HUD text."""
    out = frame
    for i, h in enumerate(humans):
        x1, y1, x2, y2 = h["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), _C_OK, 2)
        tag = f"P{i + 1}:{h['confidence']:.2f}"
        tw, th = _text_size(tag, 0.38)
        ty = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), _C_BG, -1)
        _put(out, tag, (x1 + 2, ty), scale=0.38, color=_C_OK)
    return out


def _draw_human_boxes_plain(frame: np.ndarray, humans: list[dict]) -> np.ndarray:
    """Human detections without text labels (export / dashboard video)."""
    out = frame
    for h in humans:
        x1, y1, x2, y2 = h["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), _C_OK, 2)
    return out


def compose_export_frame(
    frame_bgr: np.ndarray,
    *,
    mode: str,
    mask: np.ndarray | None = None,
    humans: list[dict] | None = None,
    drone_altitude_m: float = 50.0,
) -> tuple[np.ndarray, dict | None]:
    """Flood mask + 4×4 grid + red hotspot dot; no metrics HUD (offline export/preview)."""
    out = frame_bgr.copy()
    grid_analysis = None

    if mode in ("flood", "combined") and mask is not None:
        out, grid_analysis = draw_grid_overlay(
            out,
            mask,
            annotate_gps=False,
            drone_altitude_m=drone_altitude_m,
        )

    if humans and mode in ("human", "combined"):
        out = _draw_human_boxes_plain(out, humans)

    return out, grid_analysis


def compose_benchmark_frame(
    frame_bgr: np.ndarray,
    *,
    mode: str,
    metrics: dict,
    geom_meta: dict,
    frame_idx: int,
    mask: np.ndarray | None = None,
    humans: list[dict] | None = None,
    show_grid: bool = True,
) -> np.ndarray:
    """Flood grid + human boxes + structured benchmark HUD."""
    out = frame_bgr.copy()
    grid_analysis = None

    flood_ratio = float(metrics.get("flood_ratio", 0) or 0)
    display_label, seg_active = _flood_status(flood_ratio)
    raw_label = metrics.get("raw_classification_label", "")

    if show_grid and mask is not None and mode in ("flood", "combined") and seg_active:
        out, grid_analysis = draw_grid_overlay(out, mask, annotate_gps=False)
        if grid_analysis and grid_analysis.get("cell_ratios"):
            out = _draw_cell_ratios(out, grid_analysis["cell_ratios"])

    if humans and mode in ("human", "combined"):
        out = _draw_human_boxes(out, humans)

    altitude = float(geom_meta.get("sim_altitude_m", 0))
    placement = geom_meta.get("placement", "letterbox")
    human_count = metrics.get("human_count") if mode in ("human", "combined") else None
    gps_text = grid_analysis.get("gps_text") if grid_analysis else None

    out = _draw_header_bar(
        out,
        mode=mode,
        frame_idx=frame_idx,
        altitude=altitude,
        placement=placement,
    )
    out = _draw_detection_panel(
        out,
        display_label=display_label,
        flood_ratio=flood_ratio,
        seg_active=seg_active,
        primary=metrics.get("primary_model", "n/a"),
        raw_label=raw_label,
        human_count=human_count,
        gps_text=gps_text,
        mode=mode,
        seg_skipped=bool(metrics.get("segmentation_skipped", False)),
        model_switch=bool(metrics.get("model_switch_occurred", False)),
    )
    out = _draw_perf_panel(
        out,
        fps=float(metrics.get("fps", 0) or 0),
        latency_ms=float(metrics.get("total_latency_ms", 0) or 0),
        infer_ms=float(metrics.get("total_inference_ms", 0) or 0),
        clf_ms=float(metrics.get("classification_ms", 0) or 0),
        seg_ms=float(metrics.get("segmentation_ms", 0) or 0),
        mem_mb=float(metrics.get("memory_mb", 0) or 0),
        cpu_pct=float(metrics.get("cpu_percent", 0) or 0),
    )
    out = _draw_footer_bar(
        out,
        display_label=display_label,
        flood_ratio=flood_ratio,
        human_count=human_count,
        mode=mode,
        warmup=bool(metrics.get("warmup", False)),
    )

    return out
