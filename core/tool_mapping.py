"""Map live LLM tools to offline benchmark modes (single source of truth)."""

from __future__ import annotations

# Live API tool names (TaskSession / POST /tool)
LIVE_TOOLS = frozenset({"detect_flood", "detect_human", "detect_combined", "idle"})

# Offline benchmark runner modes
BENCHMARK_MODES = frozenset({"flood", "human", "combined"})

_TOOL_TO_MODE = {
    "detect_flood": "flood",
    "detect_human": "human",
    "detect_combined": "combined",
    "flood": "flood",
    "human": "human",
    "combined": "combined",
}

_MODE_TO_TOOL = {
    "flood": "detect_flood",
    "human": "detect_human",
    "combined": "detect_combined",
}


def tool_to_mode(tool: str | None) -> str | None:
    if not tool:
        return None
    key = tool.strip().lower()
    return _TOOL_TO_MODE.get(key)


def mode_to_tool(mode: str | None) -> str | None:
    if not mode:
        return None
    key = mode.strip().lower()
    return _MODE_TO_TOOL.get(key)


def normalize_tool(tool: str | None) -> str | None:
    mode = tool_to_mode(tool)
    return mode_to_tool(mode) if mode else None


def normalize_mode(mode: str | None) -> str | None:
    if not mode:
        return None
    key = mode.strip().lower()
    return key if key in BENCHMARK_MODES else tool_to_mode(key)


def tools_for_modes(modes: tuple[str, ...]) -> list[str]:
    out = []
    for mode in modes:
        tool = mode_to_tool(normalize_mode(mode) or "")
        if tool and tool not in out:
            out.append(tool)
    return out
