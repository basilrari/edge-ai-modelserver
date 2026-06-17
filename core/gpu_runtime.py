"""CUDA stream helpers for overlapping flood and human inference."""

from __future__ import annotations

import contextlib
import threading

import torch

from core.perf_config import CUDNN_BENCHMARK

_streams_lock = threading.Lock()
_flood_stream: torch.cuda.Stream | None = None
_human_stream: torch.cuda.Stream | None = None
_init_done = False


def _init_cuda_runtime() -> None:
    global _init_done
    if _init_done:
        return
    _init_done = True
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _make_stream() -> torch.cuda.Stream | None:
    if not torch.cuda.is_available():
        return None
    _init_cuda_runtime()
    return torch.cuda.Stream()


def get_flood_stream() -> torch.cuda.Stream | None:
    global _flood_stream
    with _streams_lock:
        if _flood_stream is None:
            _flood_stream = _make_stream()
        return _flood_stream


def get_human_stream() -> torch.cuda.Stream | None:
    global _human_stream
    with _streams_lock:
        if _human_stream is None:
            _human_stream = _make_stream()
        return _human_stream


@contextlib.contextmanager
def cuda_stream(stream: torch.cuda.Stream | None):
    if stream is None or not torch.cuda.is_available():
        yield
        return
    with torch.cuda.stream(stream):
        yield


def sync_all() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
