#!/usr/bin/env python3
"""
Standalone real-time human detection with YOLOv8n (COCO class: person).

Usage:
  cd model_server
  python3 tools/human_detection_live.py              # live window (needs display)
  python3 tools/human_detection_live.py --frames 50  # headless test, print stats
  python3 tools/human_detection_live.py --save out.jpg --frames 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO

from core.camera_stream import CameraStream

PERSON_CLASS_ID = 0
MODEL_PATH = ROOT / "yolov8n.pt"


def parse_humans(result, conf_min: float):
    humans = []
    if result.boxes is None or len(result.boxes) == 0:
        return humans

    names = result.names
    for box in result.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id != PERSON_CLASS_ID or conf < conf_min:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        humans.append(
            {
                "label": names.get(cls_id, "person"),
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return humans


def draw_humans(frame, humans):
    out = frame.copy()
    for h in humans:
        x1, y1, x2, y2 = h["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"person {h['confidence']:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        out,
        f"humans: {len(humans)}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="YOLOv8n live human detection")
    parser.add_argument("--frames", type=int, default=0, help="Stop after N frames (0 = until q)")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--save", type=str, default="", help="Save last annotated frame to path")
    parser.add_argument("--device", type=str, default=os.environ.get("CAMERA_DEVICE", ""))
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print(f"[YOLO] Loading {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    camera = CameraStream(device=args.device or None)
    print(f"[CAMERA] {camera.device_path}")

    frame_idx = 0
    t0 = time.perf_counter()
    last_out = None

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                print("[WARN] No frame")
                continue

            t_inf = time.perf_counter()
            results = model.predict(
                frame,
                conf=args.conf,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                device=0,
            )
            infer_ms = (time.perf_counter() - t_inf) * 1000.0

            humans = parse_humans(results[0], args.conf)
            last_out = draw_humans(frame, humans)
            frame_idx += 1

            status = (
                f"frame={frame_idx} humans={len(humans)} "
                f"infer={infer_ms:.1f}ms cam={camera.device_path}"
            )
            if humans:
                for i, h in enumerate(humans):
                    status += f" | #{i+1} conf={h['confidence']}"
            print(status)

            if os.environ.get("DISPLAY"):
                cv2.imshow("YOLOv8n Human Detection", last_out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif args.frames <= 0:
                time.sleep(0.03)

            if args.frames > 0 and frame_idx >= args.frames:
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    fps = frame_idx / elapsed if elapsed > 0 else 0.0
    print(f"[DONE] {frame_idx} frames, avg_fps={fps:.2f}")

    if args.save and last_out is not None:
        cv2.imwrite(args.save, last_out)
        print(f"[SAVE] {args.save}")


if __name__ == "__main__":
    main()
