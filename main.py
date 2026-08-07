import faulthandler
import os

faulthandler.enable()  # print thread stacks on SIGSEGV (native crash diagnostics)
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")

from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import asyncio
import json

from core.inference_gate import is_busy, last_latency_ms
from core.gateway_client import (
    gateway_base_url,
    get_status as gateway_get_status,
    infer_prompt,
    llm_base_url,
    llm_reachable,
)
from core.gateway_tools import (
    infer_plan_summary,
    is_llm_infer_failure,
    local_prompt_to_tool,
    map_gateway_models_to_tool,
    model_tasks_from_infer_response,
)
from core.offline_video_session import OfflineVideoSession
from core.perf_config import WS_MIN_INTERVAL
from core.task_session import TaskSession
from core.tool_mapping import normalize_tool
from router import run_tool
from pydantic import BaseModel


class WebRtcOffer(BaseModel):
    sdp: str
    type: Literal["offer"] = "offer"

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health():
    return {"status": "model server running", **TaskSession.get_status()}


@app.get("/camera/status")
def camera_status_api():
    from core.webrtc_live import camera_status

    return camera_status()


@app.get("/camera/snapshot")
def camera_snapshot():
    """Latest raw camera frame as JPEG — isolates capture from the WebRTC path."""
    import cv2
    from fastapi.responses import Response

    from core.shared_camera import get_frame

    frame = get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="no camera frame available")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="JPEG encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.post("/camera/webrtc/offer")
async def camera_webrtc_offer(body: WebRtcOffer):
    from core.webrtc_live import handle_offer

    try:
        return await handle_offer(body.sdp, body.type)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.get("/status")
def status():
    return TaskSession.get_status()


@app.get("/human/detector")
def human_detector_info():
    from core.human_detector_tier import get_status as human_detector_status

    return human_detector_status()


@app.post("/human/detector-tier")
def set_human_detector_tier(request: dict):
    from core.human_detector_tier import set_force_tier, set_tier

    mode = (request.get("mode") or "").strip().lower()
    if mode == "auto":
        result = set_force_tier(None)
        return {**TaskSession.get_status(), **result}

    tier = request.get("tier") or request.get("force_tier")
    if not tier:
        return {
            "error": "tier required (lightweight/robust) or mode=auto",
            **TaskSession.get_status(),
        }
    try:
        if request.get("force", True):
            result = set_force_tier(tier)
        else:
            result = set_tier(tier)
    except ValueError as exc:
        return {"error": str(exc), **TaskSession.get_status()}
    return {**TaskSession.get_status(), **result}


@app.get("/flood/segmenter")
def flood_segmenter_info():
    from core.flood_segmenter_tier import get_status as flood_segmenter_status

    return flood_segmenter_status()


@app.post("/flood/segmenter-tier")
def set_flood_segmenter_tier(request: dict):
    from core.flood_segmenter_tier import set_force_tier, set_tier

    mode = (request.get("mode") or "").strip().lower()
    if mode == "auto":
        result = set_force_tier(None)
        return {**TaskSession.get_status(), **result}

    tier = request.get("tier") or request.get("force_tier")
    if not tier:
        return {
            "error": "tier required (lightweight/robust) or mode=auto",
            **TaskSession.get_status(),
        }
    try:
        if request.get("force", True):
            result = set_force_tier(tier)
        else:
            result = set_tier(tier)
    except ValueError as exc:
        return {"error": str(exc), **TaskSession.get_status()}
    return {**TaskSession.get_status(), **result}


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


@app.get("/gateway/status")
def gateway_status():
    data = gateway_get_status()
    reachable = data.get("gateway_reachable", True) and "error" not in data
    llm_up = llm_reachable()
    return {
        "gateway_url": gateway_base_url(),
        "llm_url": llm_base_url(),
        "llm_reachable": llm_up,
        "reachable": reachable,
        **data,
    }


@app.post("/gateway/infer")
def gateway_infer(request: dict):
    prompt = (request.get("prompt") or "").strip()
    if not prompt:
        return {"error": "prompt required", "gateway_url": gateway_base_url()}

    outcome = infer_prompt(prompt)
    gateway_down = outcome.get("gateway_reachable") is False
    plan = None
    mapped = None
    fallback_used = False

    if gateway_down:
        mapped = local_prompt_to_tool(prompt)
        if mapped:
            fallback_used = True
            plan = {
                "model_server_tool": mapped,
                "fallback": "local_keywords",
                "gateway_unreachable": True,
                "gateway_model_tools": [],
                "drone_steps": [],
                "is_none": False,
            }
    else:
        plan = infer_plan_summary(outcome)
        mapped = plan.get("model_server_tool")
        if not mapped and is_llm_infer_failure(outcome):
            mapped = local_prompt_to_tool(prompt)
            if mapped:
                fallback_used = True
                plan = {
                    **plan,
                    "model_server_tool": mapped,
                    "fallback": "local_keywords",
                    "llm_offline": True,
                }

    tool_result = None
    if mapped and request.get("activate", True):
        if mapped == "detect_combined":
            tool_body = {"tools": ["detect_flood", "detect_human"]}
        else:
            tool_body = {"tool": mapped}
        tool_result = TaskSession.run_tool_request(tool_body)
        if tool_result and not tool_result.get("skipped") and not tool_result.get("error"):
            infer = TaskSession.run_active()
            if infer and not infer.get("skipped"):
                tool_result.update(infer)

    return {
        "gateway_url": gateway_base_url(),
        "llm_url": llm_base_url(),
        "llm_reachable": llm_reachable(),
        "prompt": prompt,
        "gateway": outcome if not gateway_down else None,
        "plan": plan,
        "model_server_tool": mapped,
        "fallback_used": fallback_used,
        "activated": mapped is not None and tool_result is not None and not tool_result.get("error"),
        "tool_result": tool_result,
        **({"error": outcome.get("error"), "gateway_reachable": False} if gateway_down else {}),
    }


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
    return templates.TemplateResponse(
        request,
        "index.html",
        {"gateway_url": gateway_base_url(), "llm_url": llm_base_url()},
    )


@app.get("/adaptive", response_class=HTMLResponse)
async def adaptive_dashboard(request: Request):
    return templates.TemplateResponse(
        "adaptive_framework.html", {"request": request}
    )


@app.on_event("startup")
def startup_event():
    print("[SYSTEM] Model server ready (idle — awaiting LLM tool command)")
    print("[SYSTEM] Tools: detect_flood | detect_human | both | idle/stop")
    print(f"[SYSTEM] Gateway proxy: {gateway_base_url()} (GATEWAY_URL to override)")
    print("[SYSTEM] WebRTC live view: POST /camera/webrtc/offer (SAR frontend /camera)")
    print("[SYSTEM] Perf: PARALLEL_COMBINED ASYNC_POWER USE_TENSORRT (see core/perf_config.py)")
    try:
        from core.gopro_preview import start_gopro_preview_if_needed

        start_gopro_preview_if_needed()
    except Exception as e:
        print(f"[CAMERA] GoPro preview failed: {e}")
    if os.environ.get("WEBRTC_WARMUP", "1") == "1":
        try:
            from core.webrtc_live import warmup_camera

            warmup_camera()
            print("[WEBRTC] Camera warmup OK")
        except Exception as e:
            print(f"[WEBRTC] Camera warmup skipped: {e}")


@app.on_event("shutdown")
async def shutdown_event_async():
    from core.gopro_preview import stop_gopro_preview_if_started
    from core.webrtc_live import close_all_peers

    await close_all_peers()
    stop_gopro_preview_if_started()
    TaskSession.deactivate()
    print("[SYSTEM] Model server stopped")


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
