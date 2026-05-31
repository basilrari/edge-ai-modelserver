import base64
import glob
import os
import subprocess
import time

import cv2
import psutil
import torch

from core.cuda_runtime import require_cuda
from core.model_manager import ModelManager
from core.inference_engine import InferenceEngine
from core.metrics_logger import MetricsLogger


class LivePipeline:

    def __init__(
        self,
        use_test_image=True,
        test_image_path="test.png",
        return_frame=True,
        log_metrics=True,
        warmup_iterations=20,
    ):
        self.use_test_image = use_test_image
        self.test_image_path = test_image_path
        self.return_frame = return_frame
        self.log_metrics = log_metrics

        self.device = require_cuda()
        print(f"[DEVICE] {torch.cuda.get_device_name(0)}")

        self.model_manager = ModelManager()
        self.engine = InferenceEngine()
        self.logger = MetricsLogger(file_path="logs/metrics_cuda.csv")

        print("[LIVE PIPELINE] Loading models on GPU...")
        t0 = time.perf_counter()
        self.clf_model = self.model_manager.load_flood_classifier()
        self.clf_load_time = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        self.seg_model = self.model_manager.load_flood_segmenter()
        self.seg_load_time = (time.perf_counter() - t1) * 1000

        print(f"[BENCHMARK] clf load: {self.clf_load_time:.2f} ms")
        print(f"[BENCHMARK] seg load: {self.seg_load_time:.2f} ms")

        self._test_frame = None
        self.cap = None

        if self.use_test_image:
            self._test_frame = cv2.imread(self.test_image_path)
            if self._test_frame is None:
                raise FileNotFoundError(
                    f"test image not found: {self.test_image_path}"
                )
            print("[MODE] TEST IMAGE (cached, deterministic benchmarking)")
        else:
            print("[CAMERA] initializing...")
            self.cap = self._create_camera()
            if self.cap is None:
                raise RuntimeError("[CAMERA] No working camera found")
            print("[CAMERA] ready")

        warmup_frame = self._test_frame if self.use_test_image else self._read_camera_frame()
        if warmup_frame is not None:
            self.engine.warmup(
                self.clf_model,
                self.seg_model,
                warmup_frame,
                iterations=warmup_iterations,
            )

        self.inference_frame_count = 0
        self.inference_ms_accum = 0.0
        self.inference_fps_start = time.perf_counter()
        self.fps = 0.0
        self.frame_count = 0
        self.warmup_frames = warmup_iterations

        self.process = psutil.Process(os.getpid())
        psutil.cpu_percent(interval=None)

        self.peak_memory_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_power_w = 0.0
        self._last_power_w = 0.0
        self._power_sample_interval = 30

    def _get_memory_mb(self):
        return self.process.memory_info().rss / (1024 * 1024)

    def _read_power_sysfs_mw(self):
        patterns = [
            "/sys/bus/i2c/devices/*/iio:device*/in_power*_input",
            "/sys/class/hwmon/hwmon*/power1_input",
        ]
        for pattern in patterns:
            for path in glob.glob(pattern):
                try:
                    with open(path, encoding="utf-8") as f:
                        return int(f.read().strip())
                except (OSError, ValueError):
                    continue
        return None

    def _sample_power_watts(self):
        milliwatts = self._read_power_sysfs_mw()
        if milliwatts is not None:
            return round(milliwatts / 1000.0, 2)

        proc = None
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "500"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            line = proc.stdout.readline() if proc.stdout else ""
            for marker in ("VDD_IN ", "POM_5V_IN"):
                if marker in line:
                    token = line.split(marker)[1].split()[0]
                    token = token.split("/")[0].replace("=", "").replace("mW", "").strip()
                    return round(float(token) / 1000.0, 2)
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
        finally:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=0.2)

        return self._last_power_w

    def _get_power_watts(self):
        if self.frame_count % self._power_sample_interval == 0:
            self._last_power_w = self._sample_power_watts()
        return self._last_power_w

    def _create_camera(self):
        gst_pipeline = (
            "v4l2src device=/dev/video0 ! "
            "video/x-raw, width=640, height=480, framerate=30/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened() and cap.read()[0]:
            return cap

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if cap.isOpened() and cap.read()[0]:
            return cap
        return None

    def _read_camera_frame(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def _get_frame(self):
        if self.use_test_image:
            return self._test_frame
        return self._read_camera_frame()

    def process_frame(self):
        frame = self._get_frame()
        if frame is None:
            return None

        self.frame_count += 1

        clf_result = self.engine.run_classification(self.clf_model, frame)
        classification_ms = clf_result.pop("inference_ms")

        seg_result = self.engine.run_segmentation(self.seg_model, frame)
        segmentation_ms = seg_result.pop("inference_ms")

        total_inference_ms = classification_ms + segmentation_ms

        self.inference_frame_count += 1
        self.inference_ms_accum += total_inference_ms

        elapsed = time.perf_counter() - self.inference_fps_start
        if elapsed >= 1.0 and self.inference_ms_accum > 0:
            self.fps = (self.inference_frame_count * 1000.0) / self.inference_ms_accum
            self.inference_frame_count = 0
            self.inference_ms_accum = 0.0
            self.inference_fps_start = time.perf_counter()

        is_warmup = self.frame_count <= self.warmup_frames
        flood_ratio = float(seg_result.get("flood_ratio", 0))

        memory_mb = self._get_memory_mb()
        cpu_percent = psutil.cpu_percent(interval=None)
        power_w = self._get_power_watts()

        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
        self.peak_power_w = max(self.peak_power_w, power_w)

        classification_ms = round(classification_ms, 2)
        segmentation_ms = round(segmentation_ms, 2)
        total_latency_ms = round(total_inference_ms, 2)
        model_switch_latency_ms = round(classification_ms + segmentation_ms, 2)

        metrics = {
            "fps": round(self.fps, 2),
            "instant_fps": round(
                1000.0 / total_inference_ms if total_inference_ms > 0 else 0.0, 2
            ),
            "total_inference_ms": total_latency_ms,
            "total_latency_ms": total_latency_ms,
            "classification_ms": classification_ms,
            "segmentation_ms": segmentation_ms,
            "model_switch_latency_ms": model_switch_latency_ms,
            "memory_mb": round(memory_mb, 2),
            "cpu_percent": round(cpu_percent, 2),
            "power_w": round(power_w, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "peak_power_w": round(self.peak_power_w, 2),
            "clf_load_ms": round(self.clf_load_time, 2),
            "seg_load_ms": round(self.seg_load_time, 2),
            "flood_ratio": round(flood_ratio, 4),
            "warmup": is_warmup,
        }

        if self.log_metrics and not is_warmup:
            self.logger.log(metrics)

        result = {
            "classification": clf_result,
            "segmentation": seg_result,
            "metrics": metrics,
        }

        if self.return_frame:
            _, buffer = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            )
            result["frame"] = base64.b64encode(buffer).decode("utf-8")

        return result

    def release(self):
        if self.cap:
            self.cap.release()
        del self.clf_model
        del self.seg_model
        torch.cuda.empty_cache()
        print("[LIVE PIPELINE] released")
