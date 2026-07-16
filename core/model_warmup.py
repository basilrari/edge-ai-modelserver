"""Preload and warmup GPU models for the active tool(s) only."""

from __future__ import annotations

import numpy as np

_warmed_tools: frozenset[str] = frozenset()


def warmup_for_tools(active_tools: set[str]) -> None:
    """Warm shared model singletons once per tool-set (no duplicate ModelManager)."""
    global _warmed_tools
    sig = frozenset(active_tools)
    if not sig or sig.issubset(_warmed_tools):
        return

    from core.gpu_runtime import sync_all
    from core.inference_engine import InferenceEngine
    from core.trt_runner import TrtFloodClassifier, TrtFloodSegmenter

    need_flood = "detect_flood" in active_tools
    need_human = "detect_human" in active_tools
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    if need_flood:
        from tools.detect_flood import _components

        model_manager, engine, _, _, _ = _components()
        clf = model_manager.load_flood_classifier()
        seg = model_manager.load_flood_segmenter()

        if isinstance(clf, TrtFloodClassifier):
            clf.run_classification(dummy)
        else:
            engine.run_classification(clf, dummy)

        if isinstance(seg, TrtFloodSegmenter):
            seg.run_segmentation(dummy)
        else:
            engine.run_segmentation(seg, dummy)

        if not isinstance(clf, TrtFloodClassifier) or not isinstance(
            seg, TrtFloodSegmenter
        ):
            engine.warmup(clf, seg, dummy, iterations=2)

    if need_human:
        from tools.detect_human import _get_model_manager

        manager = _get_model_manager()
        model = manager.load_human_detector()
        model.predict(
            dummy,
            conf=0.4,
            classes=list(manager.human_class_ids),
            verbose=False,
            device=0,
            imgsz=manager.human_imgsz,
            half=True,
        )

    sync_all()
    _warmed_tools = _warmed_tools | sig
    parts = []
    if need_flood:
        from tools.detect_flood import _components

        mm, _, _, _, _ = _components()
        parts.append(f"clf={mm.clf_backend} seg={mm.seg_backend}")
    if need_human:
        from tools.detect_human import _get_model_manager

        parts.append(
            f"human={_get_model_manager().human_backend} "
            f"tier={_get_model_manager().human_tier}"
        )
    print(f"[WARMUP] Ready ({', '.join(parts)}) tools={sorted(active_tools)}")


def warmup_all_models() -> None:
    """Legacy entry — warm everything (CLI / offline benchmarks)."""
    warmup_for_tools({"detect_flood", "detect_human"})
