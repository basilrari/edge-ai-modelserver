let ws = null;
let activeTool = "idle";
let activeTools = [];
let wsConnected = false;
let inputSource = "camera";
let offlineRunning = false;
let offlineStepBusy = false;

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
    document.getElementById("power-section").classList.toggle("hidden", activeTool === "idle");

    const feedTitle = document.getElementById("feed-title");
    if (feedTitle) {
        feedTitle.innerText = inputSource === "offline" ? "Offline video" : "Live camera";
    }
}

function setInputSource(src) {
    inputSource = src;
    const offlinePanel = document.getElementById("offline-panel");
    const cameraBtns = document.getElementById("camera-test-btns");
    if (offlinePanel) offlinePanel.classList.toggle("hidden", src !== "offline");
    if (cameraBtns) cameraBtns.classList.toggle("hidden", src === "offline");
    setText("input-source", src);
    if (src === "offline") {
        disconnectWebSocket();
        refreshOfflineVideoList();
    } else {
        stopOfflineSession();
        fetchStatus();
    }
}

async function refreshOfflineVideoList() {
    try {
        const res = await fetch("/offline/videos");
        const data = await res.json();
        const sel = document.getElementById("offline-video-select");
        if (!sel) return;
        const current = sel.value;
        sel.innerHTML = '<option value="">— upload or pick video —</option>';
        for (const v of data.videos || []) {
            const opt = document.createElement("option");
            opt.value = v.video_id;
            const exported = v.has_export ? " ✓ exported" : "";
            opt.textContent = `${v.video_id} (${v.duration_mmss}, ${v.frame_count} fr${exported})`;
            sel.appendChild(opt);
        }
        if (current) sel.value = current;
    } catch (e) {
        console.error(e);
    }
}

async function uploadOfflineVideo(file) {
    if (!file) return null;
    const form = new FormData();
    form.append("file", file);
    const res = await fetch("/offline/video/upload", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    await refreshOfflineVideoList();
    const sel = document.getElementById("offline-video-select");
    if (sel) sel.value = data.video_id;
    return data;
}

async function startOfflineSession() {
    const fileInput = document.getElementById("offline-file");
    const sel = document.getElementById("offline-video-select");
    const tool = document.getElementById("offline-tool")?.value || "detect_combined";
    const alt = Number(document.getElementById("offline-alt")?.value || 50);
    const stride = Number(document.getElementById("offline-stride")?.value || 1);
    if (stride > 1) {
        const ok = confirm(
            `Stride ${stride} records every ${stride}th frame only — plots will look sparse/flat. Use stride 1 for genuine per-frame graphs. Continue?`
        );
        if (!ok) return;
    }

    try {
        if (fileInput?.files?.length) {
            await uploadOfflineVideo(fileInput.files[0]);
            fileInput.value = "";
        }
        const videoId = sel?.value;
        if (!videoId) {
            alert("Upload or select an offline video first");
            return;
        }
        stopOfflineLoop();
        const res = await fetch("/offline/session/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                video_id: videoId,
                tool,
                altitude_m: alt,
                stride,
            }),
        });
        const data = await res.json();
        if (data.error) throw new Error(data.error);
        offlineRunning = true;
        setText("input-source", "offline_video");
        updateTaskUI(data.active_tool, data.active_tools || [tool]);
        if (data.output_dir) {
            setText("offline-export", `Writing to: ${data.output_dir}`);
        }
        offlineStepLoop();
    } catch (e) {
        console.error(e);
        alert(e.message || e);
    }
}

function stopOfflineLoop() {
    offlineRunning = false;
    offlineStepBusy = false;
}

async function stopOfflineSession() {
    stopOfflineLoop();
    try {
        const res = await fetch("/offline/session/stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ finalize: true }),
        });
        const data = await res.json();
        showExportResult(data);
        showIdleDashboard(data.message);
        setText("input-source", "idle");
        setText("offline-progress", data.export ? "Export saved." : "Stopped.");
        await refreshOfflineVideoList();
    } catch (e) {
        console.error(e);
    }
}

async function offlineStepLoop() {
    if (!offlineRunning || offlineStepBusy) return;
    offlineStepBusy = true;
    try {
        await offlineStep();
    } finally {
        offlineStepBusy = false;
        if (offlineRunning) {
            setTimeout(offlineStepLoop, 0);
        }
    }
}

function showExportResult(data) {
    const exp = data.export || {};
    const dir = exp.output_dir || data.output_dir || offDir(data);
    if (!dir) {
        setText("offline-export", "");
        return;
    }
    const lines = [
        `Saved to: ${dir}`,
        exp.overlay_video ? `Overlay: ${exp.overlay_video}` : "",
        exp.report_html ? `Report: ${exp.report_html}` : "",
    ].filter(Boolean);
    setText("offline-export", lines.join(" | "));
}

function offDir(data) {
    return data.offline?.output_dir || null;
}

async function offlineStep() {
    if (!offlineRunning) return;
    try {
        const res = await fetch("/offline/session/step", { method: "POST" });
        const data = await res.json();
        if (data.eof) {
            stopOfflineLoop();
            const off = data.offline || {};
            const target = off.frames_target || data.total_frames || "?";
            const err = data.error ? ` (error: ${data.error})` : "";
            setText(
                "offline-progress",
                `Finished — ${off.frames_processed ?? target} frames processed${err}`
            );
            showExportResult(data);
            if (data.export?.report_html) {
                const base = document.getElementById("offline-export")?.innerText || "";
                setText("offline-export", `${base} — open report.html in that folder`);
            }
            updateTaskUI("idle", []);
            setText("input-source", "idle");
            await refreshOfflineVideoList();
            return;
        }
        if (data.active_tool === "idle") {
            stopOfflineLoop();
            showIdleDashboard(data.message);
            return;
        }
        renderPayload(data);
        const off = data.offline || {};
        const target = off.frames_target || data.total_frames || "?";
        setText(
            "offline-progress",
            `frame ${off.frames_processed ?? "?"} / ${target} | ` +
            `tool=${data.active_tool} | wall=${off.wall_ms ?? "?"}ms | ` +
            `writing overlay…`
        );
    } catch (e) {
        console.error(e);
        stopOfflineLoop();
        setText("offline-progress", `Error: ${e.message || e} — finalizing partial export…`);
        try {
            const res = await fetch("/offline/session/stop", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ finalize: true }),
            });
            showExportResult(await res.json());
        } catch (stopErr) {
            console.error(stopErr);
        }
    }
}

function renderGridLocalization(grid) {
    if (!grid || grid.centroid == null) {
        setText("flood-lat", "—");
        setText("flood-lon", "—");
        setText("flood-cell", "—");
        setText("flood-cell-ratio", "—");
        setText("flood-ref-gps", "—");
        setText("flood-ref-alt", "—");
        setText("gps-text", "");
        return;
    }
    const lat = grid.latitude;
    const lon = grid.longitude;
    setText("flood-lat", lat != null ? Number(lat).toFixed(6) : "—");
    setText("flood-lon", lon != null ? Number(lon).toFixed(6) : "—");
    const cell = grid.best_cell;
    setText(
        "flood-cell",
        cell ? `row ${cell.row + 1}, col ${cell.col + 1}` : "—"
    );
    setText(
        "flood-cell-ratio",
        grid.max_cell_ratio != null ? Number(grid.max_cell_ratio).toFixed(3) : "—"
    );
    const refLat = grid.ref_latitude;
    const refLon = grid.ref_longitude;
    setText(
        "flood-ref-gps",
        refLat != null && refLon != null
            ? `${Number(refLat).toFixed(4)}, ${Number(refLon).toFixed(4)}`
            : "—"
    );
    setText("flood-ref-alt", grid.altitude_m != null ? Number(grid.altitude_m).toFixed(0) : "—");
    if (lat != null && lon != null) {
        setText(
            "gps-text",
            `Most flooded cell (simulated): ${Number(lat).toFixed(6)}, ${Number(lon).toFixed(6)}`
        );
    } else {
        setText("gps-text", grid.gps_text || "");
    }
}

function renderFlood(data) {
    const m = data.metrics || {};
    const p = data.power || {};
    const floodRatio = Number(
        data.segmentation?.flood_ratio ?? data.metrics?.flood_ratio ?? 0
    );
    const segActive =
        floodRatio >= 0.2 ||
        Boolean(data.segmentation_active ?? data.segmentation?.active);
    const classification = segActive ? "Flooded" : "Non-Flooded";

    setText("classification", classification);
    setText("raw-classification", data.classification?.raw_label || "—");
    setText("flood_ratio", Number(data.segmentation?.flood_ratio ?? 0).toFixed(3));
    setText("primary-model", data.active_models?.primary || data.primary_model || "—");
    setText("overlay-mode", segActive ? "4×4 grid (DeepLab)" : "none (ResNet)");
    renderGridLocalization(data.grid);

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

function renderPerfMetrics(data) {
    const m = data.metrics || {};
    const p = data.power || {};
    setText("memory-mb", m.memory_mb ?? "—");
    setText("cpu-pct", m.cpu_percent ?? "—");
    setText("idle-power", p.idle_power_w ?? m.idle_power_w ?? "—");
    setText("inference-power", p.inference_power_w ?? m.inference_power_w ?? "—");
    setText("extra-power", p.extra_power_w ?? m.extra_power_w ?? "—");
}

function renderHumanPanel(data) {
    const humans = data.humans || data.detections || [];
    setText("human-count", data.human_count ?? humans.length);
    const list = humans.map((h, i) => {
        const n = h.human_index ?? i + 1;
        return `#${n} conf=${h.confidence} bbox=[${h.bbox.join(",")}]`;
    }).join(" | ");
    setText("human-list", list || "No persons in frame");

    const gpsList = document.getElementById("human-gps-list");
    if (!gpsList) return;
    if (!humans.length) {
        gpsList.innerText = "—";
        return;
    }
    gpsList.innerHTML = humans.map((h, i) => {
        const n = h.human_index ?? i + 1;
        const lat = h.latitude;
        const lon = h.longitude;
        if (lat == null || lon == null) {
            return `<div>#${n}: GPS unavailable</div>`;
        }
        return (
            `<div><b>Person ${n}</b> — ` +
            `${Number(lat).toFixed(6)}, ${Number(lon).toFixed(6)} ` +
            `(conf ${h.confidence})</div>`
        );
    }).join("");
}

function renderHuman(data) {
    renderHumanPanel(data);
    const m = data.metrics || {};
    if (!isCombinedMode()) {
        setText("overlay-mode", "YOLOv8n bounding boxes");
    }
    setText("latency", m.total_latency_ms?.toFixed?.(1) ?? data.system?.latency_ms);
    setText("fps", m.instant_fps ?? data.system?.fps ?? "—");
    setText("det-ms", m.detection_ms ?? "—");
    renderPerfMetrics(data);
    setStatusBadge(document.getElementById("status"), data.system?.status);
}

function renderIdle(data) {
    setText("classification", "—");
    setText("raw-classification", "—");
    setText("flood_ratio", "—");
    setText("primary-model", "—");
    setText("human-count", "—");
    setText("human-list", "");
    setText("human-gps-list", "—");
    setText("latency", "—");
    setText("fps", "—");
    setText("clf-ms", "—");
    setText("seg-ms", "—");
    setText("det-ms", "—");
    setText("memory-mb", "—");
    setText("cpu-pct", "—");
    setText("idle-power", "—");
    setText("inference-power", "—");
    setText("extra-power", "—");
    setText("model-switch", "none");
    setText("overlay-mode", "none");
    setText("gps-text", "");
    renderGridLocalization(null);
    setText("camera-device", "—");
    setStatusBadge(document.getElementById("status"), "IDLE");
}

function showIdleDashboard(message) {
    updateTaskUI("idle", []);
    renderIdle({
        message:
            message ||
            "Model server ready — waiting for detect_flood / detect_human command",
    });
    const video = document.getElementById("video");
    if (video) video.removeAttribute("src");
}

function renderCombined(data) {
    renderFlood(data);
    renderHumanPanel(data.human_detection || data);
    setText("overlay-mode", data.overlay_mode || "flood grid + human boxes");
    setText(
        "det-ms",
        data.metrics?.human_detection_ms
            ?? data.human_detection?.metrics?.detection_ms
            ?? "—"
    );
    setText("latency", data.metrics?.combined_latency_ms ?? data.system?.latency_ms);
    setText("fps", data.system?.fps ?? data.metrics?.instant_fps);
    renderPerfMetrics(data);
    setStatusBadge(document.getElementById("status"), data.system?.status);
}

function renderPayload(data) {
    if (data.error) throw new Error(data.error);

    const tool = data.active_tool || data.task || activeTool;
    const tools = data.active_tools || activeTools;

    if (tool === "idle" || data.status === "idle") {
        showIdleDashboard(data.message);
        return;
    }

    updateTaskUI(tool, tools);

    if (tool === "detect_combined" || isCombinedMode()) renderCombined(data);
    else if (tool === "detect_flood") renderFlood(data);
    else if (tool === "detect_human") renderHuman(data);
    else showIdleDashboard(data.message);

    setText("input-source", data.input_source || (data.offline ? "offline_video" : "camera"));
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
        if (!data.inference_enabled || data.active_tool === "idle") {
            if (activeTool === "idle" && !offlineRunning) {
                disconnectWebSocket();
                showIdleDashboard();
            }
            return;
        }
        updateTaskUI(data.active_tool, data.active_tools);
        if (!wsConnected) {
            connectWebSocket();
        }
    } catch (e) {
        console.error(e);
    }
}

async function pollActiveTask() {
    if (activeTool === "idle" || wsConnected) return;
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
    if (inputSource === "offline") {
        alert("Switch to Live camera for camera test buttons, or use Run offline");
        return;
    }
    const body = tool === "detect_combined"
        ? { tools: ["detect_flood", "detect_human"] }
        : { tool };
    try {
        const response = await fetch("/tool", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await response.json();
        if (tool === "idle" || data.active_tool === "idle") {
            disconnectWebSocket();
            showIdleDashboard(data.message);
            return;
        }
        inputSource = "camera";
        setText("input-source", "camera");
        updateTaskUI(data.active_tool, data.active_tools);
        if (data.frame_base64 || data.frame) {
            renderPayload(data);
        }
        connectWebSocket();
    } catch (err) {
        console.error(err);
    }
}

function disconnectWebSocket() {
    wsConnected = false;
    if (!ws) return;
    ws.onclose = null;
    ws.close();
    ws = null;
}

function connectWebSocket() {
    if (activeTool === "idle") return;
    if (ws) {
        if (ws.readyState === WebSocket.OPEN) {
            wsConnected = true;
            return;
        }
        if (ws.readyState === WebSocket.CONNECTING) {
            return;
        }
        ws.onclose = null;
        ws.close();
        ws = null;
    }
    wsConnected = false;

    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws/live`);
    ws.onopen = () => {
        wsConnected = true;
        console.log("[WS] connected");
    };
    ws.onmessage = (ev) => {
        try {
            renderPayload(JSON.parse(ev.data));
        } catch (e) {
            console.error(e);
        }
    };
    ws.onclose = () => {
        wsConnected = false;
        ws = null;
        if (activeTool !== "idle") {
            setTimeout(connectWebSocket, 2000);
        }
    };
    ws.onerror = () => ws.close();
}

document.getElementById("offline-file")?.addEventListener("change", async (ev) => {
    const file = ev.target.files?.[0];
    if (file) {
        try {
            await uploadOfflineVideo(file);
            setText("offline-progress", `Uploaded ${file.name}`);
        } catch (e) {
            alert(e.message || e);
        }
    }
});

window.addEventListener("beforeunload", () => {
    if (offlineRunning) {
        fetch("/offline/session/stop", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ finalize: true }),
            keepalive: true,
        });
    }
});

showIdleDashboard();
setText("input-source", "idle");
fetchStatus();
setInterval(() => {
    if (inputSource === "camera" && !offlineRunning) fetchStatus();
}, 2000);
setInterval(pollActiveTask, 2500);
