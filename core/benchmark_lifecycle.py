"""Track model load/reload and first-frame vs steady-state latency for offline runs."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


@dataclass
class BenchmarkLifecycle:
    active_tool: str = ""
    benchmark_mode: str = ""

    clf_load_ms: float = 0.0
    seg_load_ms: float = 0.0
    human_load_ms: float = 0.0
    model_load_total_ms: float = 0.0
    model_load_events: int = 0
    model_reload_events: int = 0

    unload_events: list[dict] = field(default_factory=list)
    first_frame_latency_ms: float | None = None
    _latencies: list[float] = field(default_factory=list)

    def record_load(
        self,
        *,
        clf_ms: float,
        seg_ms: float,
        human_ms: float,
    ) -> None:
        self.model_load_events += 1
        if self.model_load_events > 1:
            self.model_reload_events += 1
        self.clf_load_ms = round(clf_ms, 2)
        self.seg_load_ms = round(seg_ms, 2)
        self.human_load_ms = round(human_ms, 2)
        self.model_load_total_ms = round(clf_ms + seg_ms + human_ms, 2)

    def record_unload(self, model_name: str, unload_ms: float) -> None:
        self.unload_events.append(
            {"model": model_name, "unload_ms": round(unload_ms, 2)}
        )

    def record_frame_latency(self, wall_ms: float, *, is_warmup: bool) -> None:
        if is_warmup:
            return
        if self.first_frame_latency_ms is None:
            self.first_frame_latency_ms = round(wall_ms, 2)
        self._latencies.append(wall_ms)

    def summary(self) -> dict:
        lat = self._latencies
        steady = lat[1:] if len(lat) > 1 else lat
        return {
            "active_tool": self.active_tool,
            "benchmark_mode": self.benchmark_mode,
            "model_load_events": self.model_load_events,
            "model_reload_events": self.model_reload_events,
            "clf_load_ms": self.clf_load_ms,
            "seg_load_ms": self.seg_load_ms,
            "human_load_ms": self.human_load_ms,
            "model_load_total_ms": self.model_load_total_ms,
            "unload_events": self.unload_events,
            "first_frame_latency_ms": self.first_frame_latency_ms,
            "steady_latency_ms": {
                "mean": round(statistics.mean(steady), 2) if steady else None,
                "p50": round(statistics.median(steady), 2) if steady else None,
                "p95": round(
                    sorted(steady)[max(0, int(len(steady) * 0.95) - 1)], 2
                )
                if steady
                else None,
                "frames": len(steady),
            },
        }
