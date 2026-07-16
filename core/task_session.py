"""Active detection tasks for LLM-driven tool switching (single or combined)."""

from tools.detect_combined import detect_flood_and_human
from tools.detect_flood import detect_flood
from tools.detect_human import detect_human

from core.human_detector_tier import get_status as human_detector_status

VALID_TOOLS = frozenset({"detect_flood", "detect_human"})
TOOL_ALIASES = {
    "detect_flood": "detect_flood",
    "flood_detect": "detect_flood",
    "detect_human": "detect_human",
    "human_detect": "detect_human",
    "detect_flood_and_human": "detect_combined",
    "detect_human_and_flood": "detect_combined",
    "detect_combined": "detect_combined",
    "flood_and_human": "detect_combined",
    "rescue": "detect_combined",
}


class TaskSession:
    _active_tools: set[str] = set()

    @classmethod
    def normalize_tool(cls, tool: str | None) -> str | None:
        if not tool:
            return None
        key = tool.strip()
        if key in TOOL_ALIASES:
            return TOOL_ALIASES[key]
        return key if key in VALID_TOOLS else None

    @classmethod
    def _is_stop_request(cls, value) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return value.strip().lower() in ("idle", "stop", "deactivate")
        if isinstance(value, (list, tuple, set)):
            return any(cls._is_stop_request(item) for item in value) and len(value) == 1
        return False

    @classmethod
    def _parse_tools(cls, request) -> set[str] | None:
        if request is None:
            return None
        if isinstance(request, str):
            if cls._is_stop_request(request):
                return set()
            normalized = cls.normalize_tool(request)
            if normalized == "detect_combined":
                return {"detect_flood", "detect_human"}
            if normalized in VALID_TOOLS:
                return {normalized}
            return None

        if not isinstance(request, dict):
            return None

        raw = request.get("tools")
        if raw is None:
            raw = request.get("tool")
        if raw is None:
            return None

        if cls._is_stop_request(raw):
            return set()

        if isinstance(raw, str):
            normalized = cls.normalize_tool(raw)
            if normalized == "detect_combined":
                return {"detect_flood", "detect_human"}
            if normalized in VALID_TOOLS:
                return {normalized}
            return None

        if isinstance(raw, (list, tuple, set)):
            parsed: set[str] = set()
            for item in raw:
                normalized = cls.normalize_tool(str(item))
                if normalized == "detect_combined":
                    parsed.update(VALID_TOOLS)
                elif normalized in VALID_TOOLS:
                    parsed.add(normalized)
            return parsed if parsed else None

        return None

    @classmethod
    def _active_tool_label(cls) -> str:
        if not cls._active_tools:
            return "idle"
        if cls._active_tools == VALID_TOOLS:
            return "detect_combined"
        return next(iter(cls._active_tools))

    @classmethod
    def get_status(cls) -> dict:
        label = cls._active_tool_label()
        return {
            "server": "running",
            "active_tool": label,
            "active_tools": sorted(cls._active_tools),
            "inference_enabled": bool(cls._active_tools),
            "human_detector": human_detector_status(),
        }

    @classmethod
    def _reset_tool_sessions(cls) -> None:
        from core.human_detector_tier import reset_idle
        from tools.detect_flood import reset_session as reset_flood_session
        from tools.detect_human import reset_session as reset_human_session

        reset_flood_session()
        reset_human_session()
        reset_idle()

    @classmethod
    def activate(cls, tools: set[str]) -> dict:
        previous = set(cls._active_tools)
        cls._active_tools = set(tools)
        if previous != cls._active_tools:
            cls._reset_tool_sessions()
            print(
                f"[TASK] switched {sorted(previous) or ['idle']} "
                f"→ {sorted(cls._active_tools) or ['idle']}"
            )
        return {"from": sorted(previous), "to": sorted(cls._active_tools)}

    @classmethod
    def deactivate(cls) -> dict:
        previous = set(cls._active_tools)
        cls._active_tools = set()
        if previous:
            cls._reset_tool_sessions()
            print(f"[TASK] stopped ({sorted(previous)} → idle)")
        return {"from": sorted(previous), "to": []}

    @classmethod
    def _run_active_impl(cls) -> dict:
        if cls._active_tools == VALID_TOOLS:
            return detect_flood_and_human()
        if cls._active_tools == {"detect_flood"}:
            result = detect_flood()
            result["task"] = "detect_flood"
            return result
        if cls._active_tools == {"detect_human"}:
            return detect_human()
        return {"error": f"unsupported tool set: {sorted(cls._active_tools)}"}

    @classmethod
    def run_active(cls) -> dict:
        if not cls._active_tools:
            return {
                "status": "idle",
                "active_tool": "idle",
                "active_tools": [],
                "message": "Model server ready — waiting for detect_flood / detect_human command",
            }

        from core.inference_gate import run_inference_gated
        from core.model_warmup import warmup_for_tools

        warmup_for_tools(cls._active_tools)

        result = run_inference_gated(cls._run_active_impl)
        if result is None:
            label = cls._active_tool_label()
            return {
                "skipped": True,
                "active_tool": label,
                "active_tools": sorted(cls._active_tools),
                "message": "Previous inference still running",
            }
        return result

    @classmethod
    def run_tool_request(cls, request) -> dict:
        tools = cls._parse_tools(request)
        if tools is None:
            tool_hint = request if isinstance(request, str) else request.get("tool") or request.get("tools")
            return {"error": f"unknown tool: {tool_hint}", **cls.get_status()}

        if not tools:
            switch = cls.deactivate()
            return {
                **cls.get_status(),
                "task_switch": switch,
                "message": "Detection stopped",
            }

        switch = cls.activate(tools)
        from core.model_warmup import warmup_for_tools

        warmup_for_tools(cls._active_tools)
        label = cls._active_tool_label()
        from core.inference_gate import run_inference_gated

        infer = run_inference_gated(cls._run_active_impl)
        result = {
            **cls.get_status(),
            "active_tool": label,
            "active_tools": sorted(cls._active_tools),
            "message": f"{label} active",
        }
        if infer and not infer.get("skipped"):
            result.update(infer)
            result["active_tool"] = label
            result["active_tools"] = sorted(cls._active_tools)
        if switch["from"] != switch["to"]:
            result["task_switch"] = switch
        return result

    @classmethod
    def run_tool(cls, tool: str) -> dict:
        return cls.run_tool_request({"tool": tool})
