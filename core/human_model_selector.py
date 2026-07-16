"""Context-aware selection between lightweight and robust human detectors."""


class HumanModelSelector:
    """
    Adaptive tier orchestration (mirrors FloodModelSelector pattern):
      - lightweight: YOLOv8n @ 320
      - robust: YOLO11s VisDrone @ 1280
    """

    LIGHTWEIGHT = "lightweight"
    ROBUST = "robust"

    ALTITUDE_ROBUST_M = 60.0
    ALTITUDE_BORDERLINE_M = 45.0
    PRIORITY_HIGH = 0.9
    PRIORITY_MEDIUM = 0.7
    VISIBILITY_LOW = 0.4
    FLOOD_RATIO_ROBUST = 0.20
    FLOOD_HYSTERESIS = 0.12
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

        altitude = float(context.get("altitude", 50))
        priority = float(context.get("priority", context.get("mission_priority", 0.5)))
        visibility = float(context.get("visibility", 0.5))
        flood_ratio = float(context.get("flood_ratio", 0.0))
        escalate = bool(context.get("human_escalate", False))

        if priority >= self.PRIORITY_HIGH:
            return self.ROBUST
        if altitude >= self.ALTITUDE_ROBUST_M:
            return self.ROBUST
        if visibility < self.VISIBILITY_LOW:
            return self.ROBUST
        if flood_ratio >= self.FLOOD_RATIO_ROBUST:
            return self.ROBUST
        if escalate:
            return self.ROBUST
        if (
            priority >= self.PRIORITY_MEDIUM
            and altitude >= self.ALTITUDE_BORDERLINE_M
        ):
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
                f"[HUMAN MODEL SWITCH] "
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
                "altitude": context.get("altitude"),
                "priority": context.get("priority", context.get("mission_priority")),
                "visibility": context.get("visibility"),
                "flood_ratio": context.get("flood_ratio"),
                "human_escalate": context.get("human_escalate", False),
            },
            "selected_tier": tier,
        }
