"""Load / unload / model-switch timing (same profilers as ModelManager)."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from core.context_evaluator import ContextEvaluator
from core.model_manager import ModelManager
from core.model_selector import FloodModelSelector


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def benchmark_load_unload() -> dict:
    """Cold load and unload times for each model slot."""
    results = {}

    def _fresh_manager() -> ModelManager:
        m = ModelManager()
        m.models = {}
        return m

    for name, load_fn in (
        ("flood_classifier", lambda m: m.load_flood_classifier()),
        ("flood_segmenter", lambda m: m.load_flood_segmenter()),
        ("human_detector", lambda m: m.load_human_detector()),
    ):
        manager = _fresh_manager()
        gc.collect()
        torch.cuda.empty_cache()

        t0 = time.perf_counter()
        load_fn(manager)
        load_ms = _ms(t0)

        unload_ms = manager.unload_model(name) * 1000.0

        results[name] = {
            "load_ms": round(load_ms, 2),
            "unload_ms": round(unload_ms, 2),
            "reload_ms": None,
        }

        t1 = time.perf_counter()
        load_fn(manager)
        results[name]["reload_ms"] = round(_ms(t1), 2)

    return results


def benchmark_model_switch(iterations: int = 50) -> dict:
    """
    Time logical primary-model switches (ResNet18 ↔ DeepLab) as flood_ratio changes.
    Does not unload weights — matches production TaskSession behavior.
    """
    selector = FloodModelSelector()
    context_evaluator = ContextEvaluator()
    ratios = [0.05, 0.18, 0.22, 0.35, 0.12, 0.28, 0.08, 0.25] * (iterations // 8 + 1)

    switch_events = []
    switch_ms_list = []

    for i, ratio in enumerate(ratios[:iterations]):
        context = context_evaluator.get_context()
        context["flood_ratio"] = ratio
        context["battery"] = 100
        context["cpu_usage"] = 40

        t0 = time.perf_counter()
        pre = selector.select_models(context)
        switches = selector.apply_selection(pre)
        elapsed = _ms(t0)
        switch_ms_list.append(elapsed)

        if switches.get("primary"):
            switch_events.append(
                {
                    "step": i,
                    "flood_ratio": ratio,
                    "from": switches["primary"]["from"],
                    "to": switches["primary"]["to"],
                    "latency_ms": round(elapsed, 3),
                }
            )

    return {
        "iterations": len(ratios[:iterations]),
        "switch_count": len(switch_events),
        "switch_latency_ms": {
            "mean": round(sum(switch_ms_list) / len(switch_ms_list), 3),
            "max": round(max(switch_ms_list), 3),
            "min": round(min(switch_ms_list), 3),
        },
        "events": switch_events[:20],
        "note": (
            "Primary switch is orchestration only (models stay loaded). "
            "Unload/reload times are in load_unload section."
        ),
    }


def benchmark_manager_switch() -> dict:
    """Physical unload old + load new via ModelManager.switch_model (if used)."""
    manager = ModelManager()
    manager.load_flood_classifier()

    t0 = time.perf_counter()
    manager.switch_model(
        "flood_classifier",
        "flood_segmenter",
        manager.load_flood_segmenter,
    )
    switch_physical_ms = _ms(t0)

    return {
        "flood_classifier_to_segmenter_ms": round(switch_physical_ms, 2),
        "note": "Full unload classifier + load segmenter (worst-case swap).",
    }


def run_system_benchmark(output_path: Path) -> dict:
    print("[SYSTEM BENCHMARK] load / unload ...")
    load_unload = benchmark_load_unload()

    print("[SYSTEM BENCHMARK] logical model switch ...")
    logical_switch = benchmark_model_switch()

    print("[SYSTEM BENCHMARK] physical model swap ...")
    physical_switch = benchmark_manager_switch()

    payload = {
        "load_unload": load_unload,
        "logical_primary_switch": logical_switch,
        "physical_model_swap": physical_switch,
        "live_pipeline_fields": [
            "clf_load_ms",
            "seg_load_ms",
            "model_switch_latency_ms",
            "classification_ms",
            "segmentation_ms",
            "total_latency_ms",
            "fps",
            "memory_mb",
            "cpu_percent",
            "power_w",
            "peak_power_w",
            "flood_ratio",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SYSTEM BENCHMARK] wrote {output_path}")
    return payload
