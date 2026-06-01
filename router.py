from core.task_session import TaskSession


def run_tool(request):
    if isinstance(request, dict):
        return TaskSession.run_tool_request(request)
    return TaskSession.run_tool(str(request))
