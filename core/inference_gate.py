"""Single-flight GPU inference — prevents WebSocket/API request pile-up."""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")

_lock = threading.Lock()
_busy = False
_last_latency_ms: float = 0.0


def is_busy() -> bool:
    return _busy


def last_latency_ms() -> float:
    return _last_latency_ms


def run_inference_gated(fn: Callable[[], T]) -> T | None:
    global _busy, _last_latency_ms
    if not _lock.acquire(blocking=False):
        return None
    _busy = True
    try:
        import time

        t0 = time.perf_counter()
        result = fn()
        _last_latency_ms = (time.perf_counter() - t0) * 1000.0
        return result
    finally:
        _busy = False
        _lock.release()
