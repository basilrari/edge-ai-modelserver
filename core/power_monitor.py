"""Jetson board power via tegrastats VDD_IN (idle baseline vs inference peak)."""

from __future__ import annotations

import re
import subprocess
import time

_VDD_RE = re.compile(r"VDD_IN\s+(\d+)mW")


class PowerMonitor:
    def __init__(self):
        self.idle_power_w: float | None = None
        self.last_inference_power_w: float = 0.0
        self.peak_inference_power_w: float = 0.0
        self.peak_extra_power_w: float = 0.0

    def _read_vdd_in_mw(self) -> float | None:
        proc = None
        try:
            proc = subprocess.Popen(
                ["tegrastats", "--interval", "200"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if not proc.stdout:
                return None
            # Skip first line (often stale); use the second reading.
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
                    proc.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        return None

    def _sample_watts(self) -> float:
        milliwatts = self._read_vdd_in_mw()
        if milliwatts is not None and milliwatts > 0:
            return round(milliwatts / 1000.0, 2)
        return self.last_inference_power_w

    def calibrate_idle(self, samples: int = 5) -> float:
        """Baseline at rest: use the minimum VDD_IN before GPU inference."""
        if self.idle_power_w is not None:
            return self.idle_power_w

        readings = []
        for _ in range(samples):
            value = self._sample_watts()
            if value > 0:
                readings.append(value)
            time.sleep(0.15)

        if readings:
            self.idle_power_w = round(min(readings), 2)
        else:
            self.idle_power_w = 0.0
        return self.idle_power_w

    def record_inference_power(self) -> dict:
        """
        Sample after GPU work: take the peak of a few tegrastats readings
        so inference power reflects load, not idle dips.
        """
        if self.idle_power_w is None:
            self.calibrate_idle()

        readings = []
        for _ in range(3):
            value = self._sample_watts()
            if value > 0:
                readings.append(value)
            time.sleep(0.05)

        power = round(max(readings), 2) if readings else self.last_inference_power_w
        self.last_inference_power_w = power
        self.peak_inference_power_w = max(self.peak_inference_power_w, power)

        idle = self.idle_power_w or 0.0
        extra = round(max(0.0, power - idle), 2)
        self.peak_extra_power_w = max(self.peak_extra_power_w, extra)

        return {
            "idle_power_w": idle,
            "inference_power_w": power,
            "extra_power_w": extra,
            "peak_inference_power_w": round(self.peak_inference_power_w, 2),
            "peak_extra_power_w": round(self.peak_extra_power_w, 2),
            "power_source": "tegrastats_vdd_in",
        }
