import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import asyncio
import json

from core.inference_gate import is_busy, last_latency_ms
from core.offline_video_session import OfflineVideoSession
from core.perf_config import WS_MIN_INTERVAL
from core.task_session import TaskSession
from core.tool_mapping import normalize_tool
from router import run_tool

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "model server running", **TaskSession.get_status()}


@app.get("/status")
def status():
    return TaskSession.get_status()


@app.post("/tool")
def execute_tool(request: dict):
    return run_tool(request)


@app.post("/detect_combined")
def detect_combined_api():
    TaskSession.run_tool_request({"tools": ["detect_flood", "detect_human"]})
    return TaskSession.run_active()


@app.post("/detect_flood")
def detect_flood_api():
    TaskSession.run_tool_request({"tool": "detect_flood"})
    return TaskSession.run_active()


@app.post("/detect_human")
def detect_human_api():
    TaskSession.run_tool_request({"tool": "detect_human"})
    return TaskSession.run_active()


@app.post("/stop")
def stop_detection():
    return TaskSession.run_tool("idle")


@app.get("/offline/videos")
def offline_video_list():
    return {"videos": OfflineVideoSession.list_videos()}


@app.post("/offline/video/upload")
async def offline_video_upload(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return {"error": "empty upload"}
    name = file.filename or "upload.mp4"
    if not name.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        return {"error": "supported formats: mp4, mov, avi, mkv"}
    return OfflineVideoSession.save_upload(name, data)


@app.post("/offline/session/start")
def offline_session_start(request: dict):
    video_id = request.get("video_id")
    tool = request.get("tool")
    if not video_id or not tool:
        return {"error": "video_id and tool required"}
    normalized = normalize_tool(tool)
    if not normalized or normalized == "idle":
        return {"error": f"invalid tool: {tool}"}
    TaskSession.deactivate()
    return OfflineVideoSession.start(
        video_id=video_id,
        tool=normalized,
        altitude_m=float(request.get("altitude_m", 50)),
        stride=int(request.get("stride", 1)),
    )


@app.post("/offline/session/step")
def offline_session_step():
    if OfflineVideoSession.status()["running"]:
        return OfflineVideoSession.step()
    return {
        "status": "idle",
        "active_tool": "idle",
        "message": "Start an offline session first",
        "input_source": "idle",
    }


@app.post("/offline/session/stop")
def offline_session_stop(request: dict | None = None):
    finalize = True
    if isinstance(request, dict):
        finalize = bool(request.get("finalize", True))
    return OfflineVideoSession.stop(finalize=finalize)


@app.get("/offline/session/status")
def offline_session_status():
    return OfflineVideoSession.status()


@app.get("/offline/export/{video_id}")
def offline_export_info(video_id: str):
    return OfflineVideoSession.get_export(video_id)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/adaptive", response_class=HTMLResponse)
async def adaptive_dashboard(request: Request):
    return templates.TemplateResponse(
        "adaptive_framework.html", {"request": request}
    )


@app.on_event("startup")
def startup_event():
    print("[SYSTEM] Model server ready (idle — awaiting LLM tool command)")
    print("[SYSTEM] Tools: detect_flood | detect_human | both | idle/stop")
    print("[SYSTEM] Perf: PARALLEL_COMBINED ASYNC_POWER USE_TENSORRT (see core/perf_config.py)")


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    async def _send_idle_state() -> None:
        await websocket.send_text(
            json.dumps(
                {
                    "status": "idle",
                    "active_tool": "idle",
                    "active_tools": [],
                    "message": (
                        "Model server ready — waiting for detect_flood / "
                        "detect_human command"
                    ),
                }
            )
        )

    try:
        while True:
            if not TaskSession.get_status()["inference_enabled"]:
                await _send_idle_state()
                await asyncio.sleep(2.0)
                continue

            if is_busy():
                await asyncio.sleep(0.02)
                continue

            result = await asyncio.to_thread(TaskSession.run_active)
            if result and not result.get("skipped"):
                payload = {**result, **TaskSession.get_status()}
                await websocket.send_text(json.dumps(payload))

            delay = max(WS_MIN_INTERVAL, last_latency_ms() / 1000.0 * 0.15)
            await asyncio.sleep(delay)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print("[WS ERROR]", e)


@app.on_event("shutdown")
def shutdown_event():
    TaskSession.deactivate()
    print("[SYSTEM] Model server stopped")
