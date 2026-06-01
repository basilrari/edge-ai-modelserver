let ws = null;
let activeTool = "idle";
let activeTools = [];

function isCombinedMode() {
    return activeTool === "detect_combined"
        || (activeTools.includes("detect_flood") && activeTools.includes("detect_human"));
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value ?? "—";
}

function setStatusBadge(el, systemStatus) {
    el.innerText = systemStatus || "NORMAL";
    if (systemStatus === "CRITICAL" || systemStatus === "ALERT") {
        el.className = "status alert";
    } else if (systemStatus === "WARNING") {
        el.className = "status alert";
        el.style.background = "#78350f";
    } else {
        el.className = "status ok";
    }
}

function updateTaskUI(tool, tools) {
    activeTool = tool || "idle";
    activeTools = Array.isArray(tools) ? tools : (tool && tool !== "idle" ? [tool] : []);
    if (activeTool === "detect_combined") {
        activeTools = ["detect_flood", "detect_human"];
    }

    const el = document.getElementById("active-task");
    const label = isCombinedMode() ? "detect_flood + detect_human" : activeTool;
    el.innerText = label;
    el.className =
        isCombinedMode() ? "mode-combined"
        : activeTool === "detect_flood" ? "mode-flood"
        : activeTool === "detect_human" ? "mode-human"
        : "mode-idle";

    const showFlood = activeTool === "detect_flood" || isCombinedMode();
    const showHuman = activeTool === "detect_human" || isCombinedMode();

    document.getElementById("idle-panel").classList.toggle("hidden", activeTool !== "idle");
    document.getElementById("flood-panel").classList.toggle("hidden", !showFlood);
    document.getElementById("human-panel").classList.toggle("hidden", !showHuman);
    document.getElementById("flood-metric-clf").classList.toggle("hidden", !showFlood);
    document.getElementById("flood-metric-seg").classList.toggle("hidden", !showFlood);
    document.getElementById("human-metric-det").classList.toggle("hidden", !showHuman);
    document.getElementById("power-section").classList.toggle("hidden", activeTool === "detect_human" && !showFlood);
}

function renderFlood(data) {
    const m = data.metrics || {};
    const p = data.power || {};
    const segActive = Boolean(data.segmentation_active ?? data.segmentation?.active);
    const classification = segActive ? "Flooded" : (data.classification?.label || "Non-Flooded");

    setText("classification", classification);
    setText("raw-classification", data.classification?.raw_label || "—");
    setText("flood_ratio", Number(data.segmentation?.flood_ratio ?? 0).toFixed(3));
    setText("primary-model", data.active_models?.primary || data.primary_model || "—");
    setText("overlay-mode", segActive ? "4×4 grid (DeepLab)" : "none (ResNet)");
    setText("gps-text", data.grid?.gps_text || "");

    const resnet = document.getElementById("chip-resnet");
    const deeplab = document.getElementById("chip-deeplab");
    if (resnet && deeplab) {
        resnet.classList.toggle("primary", !segActive);
        deeplab.classList.toggle("primary", segActive);
    }

    let switchText = "none";
    if (data.model_switches?.primary) {
        const s = data.model_switches.primary;
        switchText = `${s.from} → ${s.to}`;
    }
    setText("model-switch", switchText);

    setText("latency", m.total_latency_ms?.toFixed?.(1) ?? m.total_latency_ms);
    setText("fps", m.instant_fps ?? data.system?.fps ?? "—");
    setText("clf-ms", m.classification_ms);
    setText("seg-ms", m.segmentation_ms);
    setText("memory-mb", m.memory_mb);
    setText("cpu-pct", m.cpu_percent);
    setText("idle-power", p.idle_power_w ?? m.idle_power_w);
    setText("inference-power", p.inference_power_w ?? m.inference_power_w);
    setText("extra-power", p.extra_power_w ?? m.extra_power_w);
    setStatusBadge(document.getElementById("status"), data.system?.status);
}

function renderHuman(data) {
    const m = data.metrics || {};
    const humans = data.humans || data.detections || [];
    setText("human-count", data.human_count ?? humans.length);
    setText("overlay-mode", "YOLOv8n bounding boxes");
    setText("gps-text", "");

    const list = humans.map((h, i) =>
        `#${i + 1} conf=${h.confidence} bbox=[${h.bbox.join(",")}]`
    ).join(" | ");
    setText("human-list", list || "No persons in frame");

    setText("latency", m.total_latency_ms?.toFixed?.(1) ?? data.system?.latency_ms);
    setText("fps", m.instant_fps ?? data.system?.fps ?? "—");
    setText("det-ms", m.detection_ms);
    setStatusBadge(document.getElementById("status"), data.system?.status);
}

function renderIdle(data) {
    setText("overlay-mode", "none");
    setText("gps-text", data.message || "");
    setStatusBadge(document.getElementById("status"), "IDLE");
}

function renderCombined(data) {
    renderFlood(data);
    renderHuman(data.human_detection || data);
    setText("overlay-mode", data.overlay_mode || "flood grid + human boxes");
    setText("latency", data.metrics?.combined_latency_ms ?? data.system?.latency_ms);
    setText("fps", data.system?.fps ?? data.metrics?.instant_fps);
    setStatusBadge(document.getElementById("status"), data.system?.status);
}

function renderPayload(data) {
    if (data.error) throw new Error(data.error);

    const tool = data.active_tool || data.task || activeTool;
    const tools = data.active_tools || activeTools;
    updateTaskUI(tool, tools);

    if (tool === "detect_combined" || isCombinedMode()) renderCombined(data);
    else if (tool === "detect_flood") renderFlood(data);
    else if (tool === "detect_human") renderHuman(data);
    else renderIdle(data);

    setText("camera-device", data.camera?.device || "—");

    const frameB64 = data.frame_base64 || data.frame;
    if (frameB64) {
        document.getElementById("video").src = "data:image/jpeg;base64," + frameB64;
    }

    if (data.log || data.message) {
        const logBox = document.getElementById("logs");
        logBox.innerText += `[${new Date().toLocaleTimeString()}] ${data.log || data.message}\n`;
        logBox.scrollTop = logBox.scrollHeight;
        const lines = logBox.innerText.split("\n");
        if (lines.length > 50) logBox.innerText = lines.slice(-50).join("\n");
    }
}

async function fetchStatus() {
    try {
        const res = await fetch("/status");
        const data = await res.json();
        updateTaskUI(data.active_tool, data.active_tools);
    } catch (e) {
        console.error(e);
    }
}

async function pollActiveTask() {
    if (activeTool === "idle") return;
    const endpoint =
        isCombinedMode() ? "/detect_combined"
        : activeTool === "detect_human" ? "/detect_human"
        : "/detect_flood";
    try {
        const response = await fetch(endpoint, { method: "POST" });
        renderPayload(await response.json());
    } catch (err) {
        console.error(err);
    }
}

async function activateTool(tool) {
    const body = tool === "detect_combined"
        ? { tools: ["detect_flood", "detect_human"] }
        : { tool };
    try {
        const response = await fetch("/tool", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        renderPayload(await response.json());
    } catch (err) {
        console.error(err);
    }
}

function connectWebSocket() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/live`);
    ws.onmessage = (ev) => {
        try {
            renderPayload(JSON.parse(ev.data));
        } catch (e) {
            console.error(e);
        }
    };
    ws.onclose = () => setTimeout(connectWebSocket, 2000);
    ws.onerror = () => ws.close();
}

fetchStatus();
connectWebSocket();
setInterval(fetchStatus, 2000);
setInterval(pollActiveTask, 2500);
