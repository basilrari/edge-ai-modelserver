from tools.detect_human import detect_human
from tools.detect_flood import detect_flood

TOOL_MAP = {

"detect_human": detect_human,
"detect_flood": detect_flood

}

def run_tool(tool):

    if tool not in TOOL_MAP:

        return {"error": "unknown tool"}

    return TOOL_MAP[tool]()