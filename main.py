import os

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import json
import asyncio

from router import run_tool
from tools.detect_flood import detect_flood

app = FastAPI()

# -------------------------
# Static & Templates
# -------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/health")
def health():
    return {"status": "model server running"}


# =====================================================
# TOOL API (UNCHANGED)
# =====================================================
@app.post("/tool")
def execute_tool(request: dict):
    tool = request.get("tool")
    return run_tool(tool)


# =====================================================
# DASHBOARD
# =====================================================
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/adaptive", response_class=HTMLResponse)
async def adaptive_dashboard(request: Request):
    return templates.TemplateResponse(request, "adaptive_framework.html")


# =====================================================
# FLOOD DETECTION API (adaptive model selection + inference)
# =====================================================
@app.post("/detect_flood")
def detect_flood_api():
    return detect_flood()


# =====================================================
# STARTUP — warm GPU models once
# =====================================================
@app.on_event("startup")
def startup_event():
    print("[SYSTEM] Warming adaptive flood detection...")
    try:
        result = detect_flood()
        if result.get("error"):
            print("[ERROR] Warmup failed:", result["error"])
        else:
            models = result.get("active_models", {})
            print(
                "[SYSTEM] Ready — classifier=%s segmenter=%s"
                % (models.get("classifier"), models.get("segmenter"))
            )
    except Exception as e:
        print("[ERROR] Startup warmup failed:", e)


# =====================================================
# WEBSOCKET STREAM (REAL-TIME MODE)
# =====================================================
@app.websocket("/ws/live")
async def live_stream(websocket: WebSocket):

    await websocket.accept()
    print("[WS] Client connected")

    try:
        while True:
            result = detect_flood()
            if result and "error" not in result:
                await websocket.send_text(json.dumps(result))
            await asyncio.sleep(0.2)

    except WebSocketDisconnect:
        print("[WS] Client disconnected")

    except Exception as e:
        print("[WS ERROR]", e)


# =====================================================
# SHUTDOWN
# =====================================================
@app.on_event("shutdown")
def shutdown_event():
    print("[SYSTEM] Flood detection API stopped")
