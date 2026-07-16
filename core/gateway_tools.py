"""Map edge-ai-gateway model tool names to model-server tools (unchanged API)."""

from __future__ import annotations

FLOOD_GATEWAY_TOOLS = frozenset({"flood_seg", "flood_class"})
HUMAN_GATEWAY_TOOLS = frozenset({"human_detect"})


def model_tasks_from_infer_response(response: dict) -> list[str]:
    """Collect gateway model tool names from an /infer response."""
    names: list[str] = []
    tools = response.get("tools")
    if isinstance(tools, list):
        for step in tools:
            if isinstance(step, dict) and step.get("category") == "model":
                name = str(step.get("name", "")).strip()
                if name:
                    names.append(name)
    if not names and response.get("category") == "model":
        name = str(response.get("tool_name", "")).strip()
        if name:
            names.append(name)
    return names


def map_gateway_models_to_tool(model_names: list[str]) -> str | None:
    """
    Translate gateway model tools → model server tool.
    Returns detect_flood | detect_human | detect_combined, or None.
    """
    if not model_names:
        return None
    normalized = {n.strip().lower() for n in model_names if n}
    has_human = bool(normalized & HUMAN_GATEWAY_TOOLS)
    has_flood = bool(normalized & FLOOD_GATEWAY_TOOLS)
    if has_human and has_flood:
        return "detect_combined"
    if has_human:
        return "detect_human"
    if has_flood:
        return "detect_flood"
    return None


def is_llm_infer_failure(gateway_response: dict) -> bool:
    action = str(gateway_response.get("action_taken") or "")
    if "llm_http_failed" in action or "llm_parse_failed" in action:
        return True
    trace = gateway_response.get("debug_trace") or []
    if isinstance(trace, list):
        for line in trace:
            if "llm_http_transport_failed" in str(line):
                return True
    return False


def local_prompt_to_tool(prompt: str) -> str | None:
    """
    Keyword fallback when the LLM server is offline.
    Maps natural phrases to existing model-server tools only.
    """
    text = (prompt or "").lower()
    if not text.strip():
        return None

    human_kw = (
        "human", "humans", "people", "person", "persons",
        "survivor", "survivors", "victim", "stuck",
    )
    flood_kw = (
        "flood", "flooded", "flooding", "water", "inundat", "submerged",
    )
    has_human = any(k in text for k in human_kw)
    has_flood = any(k in text for k in flood_kw)

    if has_human and has_flood:
        return "detect_combined"
    if has_human:
        return "detect_human"
    if has_flood:
        return "detect_flood"
    return None


def infer_plan_summary(response: dict) -> dict:
    """Human-readable summary for dashboard / logs."""
    tasks = response.get("tools") if isinstance(response.get("tools"), list) else []
    drone_steps = [
        t.get("name") for t in tasks
        if isinstance(t, dict) and t.get("category") == "drone"
    ]
    model_names = model_tasks_from_infer_response(response)
    mapped = map_gateway_models_to_tool(model_names)
    category = response.get("category")
    tool_name = response.get("tool_name")
    is_none = (
        category == "none"
        or (
            not model_names
            and not drone_steps
            and mapped is None
            and category not in ("model", "drone")
        )
    )
    return {
        "drone_steps": drone_steps,
        "gateway_model_tools": model_names,
        "model_server_tool": mapped,
        "category": category,
        "tool_name": tool_name,
        "is_none": bool(is_none and not model_names and not drone_steps),
        "action_taken": response.get("action_taken"),
        "drone_error": response.get("drone_error"),
    }
