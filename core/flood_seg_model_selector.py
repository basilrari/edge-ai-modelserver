"""Context-aware selection between lightweight and robust flood segmenters.

Design (mirrors human dual-tier):
  - Lightweight: project best_model.pth — tuned to deployment camera/domain; used for
    dry scenes, periodic SEG_INTERVAL refresh, and low-power fallback.
  - Robust: FloodNet best_model_robust_floodnet.pth — better generalization on unseen
    post-disaster aerial footage (test IoU 0.76).

Only one segmenter is loaded at a time. ResNet18 always classifies first; SMART_SEGMENT
decides whether to run DeepLab this frame; tier selector picks which DeepLab weights.
"""

from core.segment_policy import CLF_FLOOD_CLASS


class FloodSegModelSelector:
    """
    Tier orchestration for DeepLab flood segmenters:
      - lightweight: project-trained best_model.pth
      - robust: FloodNet-trained best_model_robust_floodnet.pth
    """

    LIGHTWEIGHT = "lightweight"
    ROBUST = "robust"

    FLOOD_RATIO_ROBUST = 0.20
    FLOOD_HYSTERESIS = 0.12
    PRIORITY_HIGH = 0.9
    VISIBILITY_LOW = 0.4
    BATTERY_FORCE_LIGHT = 20.0
    CPU_FORCE_LIGHT = 90.0

    def __init__(self):
        self.active_tier = self.LIGHTWEIGHT

    def select_tier(self, context: dict) -> dict:
        tier = self._choose_tier(context)
        return {
            "tier": tier,
            "metadata": self.get_selection_metadata(context, tier),
        }

    def _choose_tier(self, context: dict) -> str:
        battery = float(context.get("battery", 100))
        cpu_usage = float(context.get("cpu_usage", 0))
        if battery < self.BATTERY_FORCE_LIGHT or cpu_usage > self.CPU_FORCE_LIGHT:
            return self.LIGHTWEIGHT

        priority = float(context.get("priority", context.get("mission_priority", 0.5)))
        visibility = float(context.get("visibility", 0.5))
        flood_ratio = float(context.get("flood_ratio", 0.0))
        clf_index = context.get("clf_class_index")
        raw_label = str(context.get("classification_label", "")).lower()

        if priority >= self.PRIORITY_HIGH:
            return self.ROBUST
        if visibility < self.VISIBILITY_LOW:
            return self.ROBUST
        if flood_ratio >= self.FLOOD_RATIO_ROBUST:
            return self.ROBUST
        if clf_index == CLF_FLOOD_CLASS:
            return self.ROBUST
        if raw_label in ("flood", "flooded"):
            return self.ROBUST
        if (
            self.active_tier == self.ROBUST
            and flood_ratio >= self.FLOOD_HYSTERESIS
        ):
            return self.ROBUST

        return self.LIGHTWEIGHT

    def apply_selection(self, selection: dict) -> dict:
        switches = {}
        new_tier = selection["tier"]
        if new_tier != self.active_tier:
            switches["tier"] = {
                "from": self.active_tier,
                "to": new_tier,
            }
            self.active_tier = new_tier
            print(
                f"[FLOOD SEG TIER] "
                f"{switches['tier']['from']} → {switches['tier']['to']}"
            )
        return switches

    def reset(self) -> None:
        self.active_tier = self.LIGHTWEIGHT

    def get_selection_metadata(self, context: dict, tier: str) -> dict:
        return {
            "decision_basis": {
                "battery": context.get("battery"),
                "cpu_usage": context.get("cpu_usage"),
                "priority": context.get("priority", context.get("mission_priority")),
                "visibility": context.get("visibility"),
                "flood_ratio": context.get("flood_ratio"),
                "clf_class_index": context.get("clf_class_index"),
                "classification_label": context.get("classification_label"),
            },
            "selected_tier": tier,
        }
