import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import json
import asyncio

from core.task_session import TaskSession
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
    return TaskSession.run_tool_request({"tools": ["detect_flood", "detect_human"]})


@app.post("/detect_flood")
def detect_flood_api():
    return TaskSession.run_tool("detect_flood")


@app.post("/detect_human")
def detect_human_api():
    return TaskSession.run_tool("detect_human")


@app.post("/stop")
def stop_detection():
    return TaskSession.run_tool("idle")


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


@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    try:
        while True:
            result = TaskSession.run_active()
            if result:
                payload = {**result, **TaskSession.get_status()}
                await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.2 if TaskSession.get_status()["inference_enabled"] else 1.0)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print("[WS ERROR]", e)


@app.on_event("shutdown")
def shutdown_event():
    TaskSession.deactivate()
    print("[SYSTEM] Model server stopped")
