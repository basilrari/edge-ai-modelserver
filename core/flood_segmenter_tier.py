"""Runtime flood segmenter tier with context-aware switching and optional force override."""

from __future__ import annotations

from core.flood_seg_model_selector import FloodSegModelSelector

VALID_TIERS = frozenset({"lightweight", "robust"})

_current_tier = "lightweight"
_force_tier: str | None = None
_inference_active = False
_last_switch: dict | None = None
_last_selection: dict | None = None
_selector = FloodSegModelSelector()


def get_selector() -> FloodSegModelSelector:
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
            "segmenter_key": "deeplabv3plus_floodnet",
            "segmenter_label": "DeepLabv3+ FloodNet",
            "weights_file": "best_model_robust_floodnet.pth",
            "description": "FloodNet-trained robust segmenter @ 256px",
        }
    return {
        "tier": "lightweight",
        "segmenter_key": "deeplabv3plus",
        "segmenter_label": "DeepLabv3+",
        "weights_file": "best_model.pth",
        "description": "Project lightweight segmenter @ 256px",
    }


def get_status() -> dict:
    mode = "forced" if _force_tier else "auto"
    meta = tier_metadata() if _inference_active else None
    return {
        "mode": mode,
        "force_tier": _force_tier,
        "inference_active": _inference_active,
        "tier": _current_tier if _inference_active else None,
        "segmenter_key": meta["segmenter_key"] if meta else None,
        "segmenter_label": meta["segmenter_label"] if meta else None,
        "available_tiers": sorted(VALID_TIERS),
        "available_modes": ["auto", "forced"],
        "last_switch": _last_switch,
        "last_selection": _last_selection,
        "training_report": "training_report.json",
        "status_note": (
            "Context-aware flood segmentation tier active"
            if _inference_active
            else "Idle — activate detect_flood or combined to start"
        ),
    }


def _unload_flood_segmenter() -> None:
    from tools.detect_flood import _get_model_manager

    manager = _get_model_manager()
    if "flood_segmenter" in manager.models:
        manager.unload_model("flood_segmenter")
    manager.seg_cache_sig = None


def _reset_flood_warmup() -> None:
    import core.model_warmup as mw

    mw._warmed_tools = frozenset(
        t for t in mw._warmed_tools if t != "detect_flood"
    )


def _apply_tier(tier: str, reason: str = "manual") -> dict:
    global _current_tier, _last_switch

    normalized = tier.strip().lower()
    if normalized not in VALID_TIERS:
        raise ValueError(f"unknown tier: {tier}")

    previous = _current_tier
    if normalized == previous:
        return {"changed": False, "from": previous, "to": normalized}

    _unload_flood_segmenter()
    _reset_flood_warmup()
    _current_tier = normalized
    _selector.active_tier = normalized
    _last_switch = {
        "from": previous,
        "to": normalized,
        "from_label": tier_metadata(previous)["segmenter_label"],
        "to_label": tier_metadata(normalized)["segmenter_label"],
        "reason": reason,
    }
    print(
        f"[FLOOD SEG TIER] {previous} → {normalized} "
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
    global _inference_active, _last_selection
    _inference_active = False
    _last_selection = None
    _selector.reset()
    _unload_flood_segmenter()
    _reset_flood_warmup()


def resolve_tier_for_context(context: dict) -> dict:
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
            "message": "Flood segmenter set to context-aware auto switching",
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
            "message": f"Flood segmenter forced to {normalized}",
            **swap,
        }
    return {
        "mode": "forced",
        "force_tier": normalized,
        "message": f"Force tier set to {normalized} — applies on next flood segmentation",
    }


def set_tier(tier: str) -> dict:
    result = set_force_tier(tier)
    return {**result, **tier_metadata(_force_tier or tier)}
