class FloodModelSelector:
    """
    Adaptive orchestration for the two deployed flood models only:
      - resnet18 (classification)
      - deeplabv3plus (segmentation)

    Switching changes which model leads the flood decision after each inference.
    Both models stay loaded; inference always runs both unless low-power mode.
    """

    RESNET18 = "resnet18"
    DEEPLAB = "deeplabv3plus"
    FLOOD_RATIO_THRESHOLD = 0.20

    def __init__(self):
        self.primary_model = self.RESNET18
        self.current_models = {
            "classifier": self.RESNET18,
            "segmenter": self.DEEPLAB,
        }

    def select_models(self, context):
        primary = self._choose_primary(context)
        run_segmenter = self._should_run_segmenter(context)

        selection = {
            "classifier": self.RESNET18,
            "segmenter": self.DEEPLAB,
            "primary_model": primary,
            "run_classifier": True,
            "run_segmenter": run_segmenter,
            "metadata": self.get_selection_metadata(context, primary, run_segmenter),
        }
        return selection

    def _choose_primary(self, context):
        flood_ratio = float(context.get("flood_ratio", 0.0))
        if flood_ratio >= self.FLOOD_RATIO_THRESHOLD:
            return self.DEEPLAB
        return self.RESNET18

    def is_segmentation_active(self, flood_ratio: float) -> bool:
        return float(flood_ratio) >= self.FLOOD_RATIO_THRESHOLD

    def _should_run_segmenter(self, context):
        battery = float(context.get("battery", 100))
        cpu_usage = float(context.get("cpu_usage", 0))
        if battery < 15 or cpu_usage > 90:
            return False
        return True

    def apply_selection(self, selection):
        switches = {}
        new_primary = selection["primary_model"]
        if new_primary != self.primary_model:
            switches["primary"] = {
                "from": self.primary_model,
                "to": new_primary,
            }
            self.primary_model = new_primary
            print(
                f"[MODEL SWITCH] primary: "
                f"{switches['primary']['from']} → {switches['primary']['to']}"
            )
        return switches

    def check_switch_needed(self, _current, selected):
        return self.apply_selection(selected)

    def update_current_models(self, selected):
        self.primary_model = selected.get("primary_model", self.primary_model)

    def get_selection_metadata(self, context, primary, run_segmenter):
        return {
            "decision_basis": {
                "battery": context.get("battery"),
                "cpu_usage": context.get("cpu_usage"),
                "flood_ratio": context.get("flood_ratio"),
                "classification_label": context.get("classification_label"),
            },
            "selected_models": {
                "classifier": self.RESNET18,
                "segmenter": self.DEEPLAB,
                "primary_model": primary,
                "run_segmenter": run_segmenter,
            },
        }
