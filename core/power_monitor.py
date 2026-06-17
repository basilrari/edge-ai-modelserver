"""Jetson board power via tegrastats VDD_IN (non-blocking background sampling)."""

from __future__ import annotations

import re
import subprocess
import threading
import time

from core.perf_config import ASYNC_POWER

_VDD_RE = re.compile(r"VDD_IN\s+(\d+)mW")

_instance: "PowerMonitor | None" = None
_instance_lock = threading.Lock()


def get_power_monitor() -> "PowerMonitor":
    """Process-wide monitor (one tegrastats sampler for live + offline paths)."""
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = PowerMonitor()
            _instance.calibrate_idle()
        return _instance


class PowerMonitor:
    def __init__(self):
        self.idle_power_w: float | None = None
        self.last_inference_power_w: float = 0.0
        self.peak_inference_power_w: float = 0.0
        self.peak_extra_power_w: float = 0.0
        self._cached: dict = {
            "idle_power_w": 0.0,
            "inference_power_w": 0.0,
            "extra_power_w": 0.0,
            "peak_inference_power_w": 0.0,
            "peak_extra_power_w": 0.0,
            "power_source": "tegrastats_vdd_in",
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if ASYNC_POWER:
            self._thread = threading.Thread(
                target=self._sample_loop, name="power-monitor", daemon=True
            )
            self._thread.start()

    def _read_vdd_in_mw(self) -> float | None:
        proc = None
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "100"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if not proc.stdout:
                return None
            proc.stdout.readline()
            line = proc.stdout.readline()
            match = _VDD_RE.search(line or "")
            if match:
                return float(match.group(1))
        except (subprocess.SubprocessError, ValueError, OSError):
            return None
        finally:
            if proc is not None:
                proc.kill()
                try:
                    proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        return None

    def _sample_watts(self) -> float:
        milliwatts = self._read_vdd_in_mw()
        if milliwatts is not None and milliwatts > 0:
            return round(milliwatts / 1000.0, 2)
        return self.last_inference_power_w

    def _update_cache(self, power: float) -> dict:
        if self.idle_power_w is None:
            self.calibrate_idle(samples=3)

        idle = self.idle_power_w or 0.0
        extra = round(max(0.0, power - idle), 2)
        self.last_inference_power_w = power
        self.peak_inference_power_w = max(self.peak_inference_power_w, power)
        self.peak_extra_power_w = max(self.peak_extra_power_w, extra)

        payload = {
            "idle_power_w": idle,
            "inference_power_w": power,
            "extra_power_w": extra,
            "peak_inference_power_w": round(self.peak_inference_power_w, 2),
            "peak_extra_power_w": round(self.peak_extra_power_w, 2),
            "power_source": "tegrastats_vdd_in",
        }
        with self._lock:
            self._cached = payload
        return payload

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            value = self._sample_watts()
            if value > 0:
                self._update_cache(value)
            self._stop.wait(0.35)

    def calibrate_idle(self, samples: int = 5) -> float:
        if self.idle_power_w is not None:
            return self.idle_power_w

        readings = []
        for _ in range(samples):
            value = self._sample_watts()
            if value > 0:
                readings.append(value)
            time.sleep(0.1)

        if readings:
            self.idle_power_w = round(min(readings), 2)
        else:
            self.idle_power_w = 0.0
        return self.idle_power_w

    def record_inference_power(self) -> dict:
        """Return latest power reading without blocking inference."""
        if ASYNC_POWER:
            with self._lock:
                return dict(self._cached)

        if self.idle_power_w is None:
            self.calibrate_idle()

        readings = []
        for _ in range(2):
            value = self._sample_watts()
            if value > 0:
                readings.append(value)

        power = round(max(readings), 2) if readings else self.last_inference_power_w
        return self._update_cache(power)

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
