"""Runtime human detector tier with context-aware auto switching and optional force override."""

from __future__ import annotations

import os

from core.human_model_selector import HumanModelSelector
from core.perf_config import YOLO_IMGSZ, YOLO_ROBUST_IMGSZ

VALID_TIERS = frozenset({"lightweight", "robust"})

_current_tier = "lightweight"
_force_tier: str | None = None
_inference_active = False
_last_switch: dict | None = None
_last_selection: dict | None = None
_selector = HumanModelSelector()


def get_selector() -> HumanModelSelector:
    return _selector


def get_tier() -> str:
    return _current_tier


def is_inference_active() -> bool:
    return _inference_active


def tier_metadata(tier: str | None = None) -> dict:
    tier = tier or _current_tier
    if tier == "robust":
        return {
            "tier": "robust",
            "detector_key": "yolo11s_visdrone_human_1280",
            "detector_label": "YOLO11s VisDrone",
            "imgsz": YOLO_ROBUST_IMGSZ,
            "human_label": "human",
            "class_ids": [0, 1],
            "description": "Aerial small-human specialist @ 1280px",
        }
    return {
        "tier": "lightweight",
        "detector_key": "yolov8n",
        "detector_label": "YOLOv8n",
        "imgsz": YOLO_IMGSZ,
        "human_label": "human",
        "class_ids": [0],
        "description": "Fast patrol detector @ 320px",
    }


def get_status() -> dict:
    mode = "forced" if _force_tier else "auto"
    meta = tier_metadata() if _inference_active else None
    return {
        "mode": mode,
        "force_tier": _force_tier,
        "inference_active": _inference_active,
        "tier": _current_tier if _inference_active else None,
        "detector_key": meta["detector_key"] if meta else None,
        "detector_label": meta["detector_label"] if meta else None,
        "imgsz": meta["imgsz"] if meta else None,
        "human_label": "human",
        "available_tiers": sorted(VALID_TIERS),
        "available_modes": ["auto", "forced"],
        "last_switch": _last_switch,
        "last_selection": _last_selection,
        "status_note": (
            "Context-aware switching active"
            if _inference_active
            else "Idle — activate detect_human or combined to start"
        ),
    }


def _unload_human_detector() -> None:
    from tools.detect_human import _get_model_manager

    manager = _get_model_manager()
    if "human_detector" in manager.models:
        manager.unload_model("human_detector")


def _reset_human_warmup() -> None:
    import core.model_warmup as mw

    mw._warmed_tools = frozenset(
        t for t in mw._warmed_tools if t != "detect_human"
    )


def _apply_tier(tier: str, reason: str = "manual") -> dict:
    global _current_tier, _last_switch

    normalized = tier.strip().lower()
    if normalized not in VALID_TIERS:
        raise ValueError(f"unknown tier: {tier}")

    previous = _current_tier
    if normalized == previous:
        return {"changed": False, "from": previous, "to": normalized}

    _unload_human_detector()
    _reset_human_warmup()
    _current_tier = normalized
    _selector.active_tier = normalized
    _last_switch = {
        "from": previous,
        "to": normalized,
        "from_label": tier_metadata(previous)["detector_label"],
        "to_label": tier_metadata(normalized)["detector_label"],
        "reason": reason,
    }
    print(
        f"[HUMAN TIER] {previous} → {normalized} "
        f"({_last_switch['from_label']} → {_last_switch['to_label']}) [{reason}]"
    )
    return {
        "changed": True,
        "from": previous,
        "to": normalized,
        "switch": _last_switch,
    }


def begin_inference() -> None:
    global _inference_active
    _inference_active = True


def reset_idle() -> None:
    """Called when detection stops — no tier switching until next activation."""
    global _inference_active, _last_selection
    _inference_active = False
    _last_selection = None
    _selector.reset()
    _unload_human_detector()
    _reset_human_warmup()


def resolve_tier_for_context(context: dict) -> dict:
    """
    Pick tier from context (or force override), swap weights if needed.
    Only acts when human inference is active.
    """
    global _last_selection

    begin_inference()

    if _force_tier:
        selected_tier = _force_tier
        metadata = {
            "decision_basis": {"force_tier": _force_tier},
            "selected_tier": selected_tier,
            "mode": "forced",
        }
        switches = {}
    else:
        selection = _selector.select_tier(context)
        selected_tier = selection["tier"]
        metadata = selection["metadata"]
        metadata["mode"] = "auto"
        switches = _selector.apply_selection(selection)

    swap = _apply_tier(selected_tier, reason=metadata.get("mode", "auto"))
    if switches.get("tier"):
        switches["tier"]["reason"] = metadata.get("mode", "auto")

    result = {
        **tier_metadata(selected_tier),
        "mode": metadata.get("mode", "auto"),
        "metadata": metadata,
        "tier_switches": switches,
        **swap,
    }
    _last_selection = result
    return result


def set_force_tier(tier: str | None) -> dict:
    global _force_tier

    if tier is None or str(tier).strip().lower() in ("", "auto", "none"):
        _force_tier = None
        return {
            "mode": "auto",
            "force_tier": None,
            "message": "Human detector set to context-aware auto switching",
        }

    normalized = str(tier).strip().lower()
    if normalized not in VALID_TIERS:
        raise ValueError(f"unknown tier: {tier}")

    _force_tier = normalized
    if _inference_active:
        swap = _apply_tier(normalized, reason="forced")
        return {
            "mode": "forced",
            "force_tier": normalized,
            "message": f"Human detector forced to {normalized}",
            **swap,
        }
    return {
        "mode": "forced",
        "force_tier": normalized,
        "message": f"Force tier set to {normalized} — applies on next human inference",
    }


def set_tier(tier: str) -> dict:
    """Manual tier set (force override)."""
    result = set_force_tier(tier)
    return {**result, **tier_metadata(_force_tier or tier)}
