let ws = null;
let activeTool = "idle";
let activeTools = [];
let wsConnected = false;
let inputSource = "camera";
let offlineRunning = false;
let offlineStepBusy = false;
let humanDetectorTier = "lightweight";
let floodSegmenterTier = "lightweight";

const LIVE_CHART_LEN = 50;
const latencyHistory = [];
const fpsHistory = [];

function resetLiveCharts() {
    latencyHistory.length = 0;
    fpsHistory.length = 0;
    setText("chart-latency-val", "—");
    setText("chart-fps-val", "—");
    drawLiveChart("chart-latency", [], "#00e5ff", "ms");
    drawLiveChart("chart-fps", [], "#00ffa6", "");
}

function extractLiveMetrics(data) {
    const m = data.metrics || {};
    const hm = data.human_detection?.metrics || {};
    let latency = m.total_latency_ms ?? m.combined_latency_ms ?? data.system?.latency_ms;
    if (latency == null && hm.detection_ms != null && isCombinedMode()) {
        latency = Number(m.classification_ms || 0) + Number(m.segmentation_ms || 0) + Number(hm.detection_ms || 0);
    }
    const fps = m.instant_fps ?? data.system?.fps ?? hm.instant_fps;
    return {
        latency: Number(latency),
        fps: Number(fps),
    };
}

function recordLiveMetrics(data) {
    const { latency, fps } = extractLiveMetrics(data);
    if (Number.isFinite(latency) && latency > 0) {
        latencyHistory.push(latency);
        if (latencyHistory.length > LIVE_CHART_LEN) latencyHistory.shift();
        setText("chart-latency-val", latency.toFixed(1) + " ms");
    }
    if (Number.isFinite(fps) && fps > 0) {
        fpsHistory.push(fps);
        if (fpsHistory.length > LIVE_CHART_LEN) fpsHistory.shift();
        setText("chart-fps-val", fps.toFixed(2));
    }
    drawLiveChart("chart-latency", latencyHistory, "#00e5ff", "ms");
    drawLiveChart("chart-fps", fpsHistory, "#00ffa6", "");
}

function drawLiveChart(canvasId, values, color, unit) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(200, Math.floor(rect.width * dpr));
    const h = Math.max(80, Math.floor(rect.height * dpr));
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#020617";
    ctx.fillRect(0, 0, w, h);

    const padL = 36 * dpr;
    const padR = 8 * dpr;
    const padT = 10 * dpr;
    const padB = 18 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;

    if (!values.length) {
        ctx.fillStyle = "#475569";
        ctx.font = `${12 * dpr}px monospace`;
        ctx.fillText("waiting for data…", padL, h / 2);
        return;
    }

    const min = Math.min(...values);
    const max = Math.max(...values);
    const span = Math.max(max - min, max * 0.15, 1);
    const yMin = Math.max(0, min - span * 0.1);
    const yMax = max + span * 0.1;

    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const y = padT + (plotH * i) / 4;
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(w - padR, y);
        ctx.stroke();
        const val = yMax - ((yMax - yMin) * i) / 4;
        ctx.fillStyle = "#64748b";
        ctx.font = `${10 * dpr}px monospace`;
        ctx.fillText(val.toFixed(unit === "ms" ? 0 : 1), 4 * dpr, y + 4 * dpr);
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * dpr;
    ctx.beginPath();
    values.forEach((v, i) => {
        const x = padL + (plotW * i) / Math.max(values.length - 1, 1);
        const y = padT + plotH * (1 - (v - yMin) / (yMax - yMin));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const last = values[values.length - 1];
    const lx = padL + plotW;
    const ly = padT + plotH * (1 - (last - yMin) / (yMax - yMin));
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(lx, ly, 4 * dpr, 0, Math.PI * 2);
    ctx.fill();
}

function humanModelsFromPayload(data) {
    if (data?.active_models?.tier) return data.active_models;
    if (data?.human_detection?.active_models) return data.human_detection.active_models;
    if (data?.human_detector?.tier) return data.human_detector;
    if (data?.human_detector) return data.human_detector;
    return null;
}

function floodSegFromPayload(data) {
    if (data?.flood_segmenter?.tier) return data.flood_segmenter;
    if (data?.active_models?.segmenter_tier) {
        return {
            tier: data.active_models.segmenter_tier,
            segmenter_label: data.active_models.segmenter,
            segmenter_key: data.active_models.segmenter_key,
            mode: data.flood_segmenter?.mode || "auto",
        };
    }
    return data?.flood_segmenter || null;
}

function renderFloodSegmenterTier(data) {
    const status = data?.flood_segmenter || {};
    const segInfo = floodSegFromPayload(data) || {};
    const models = data?.active_models || {};
    const mode = status.mode || segInfo.mode || "auto";
    const inferenceActive = Boolean(
        status.inference_active
        || data?.active_tool === "detect_flood"
        || data?.active_tool === "detect_combined"
        || data?.task === "detect_flood"
        || data?.task === "detect_combined"
    );

    const tier = inferenceActive
        ? (segInfo.tier || models.segmenter_tier || status.tier || floodSegmenterTier || "lightweight")
        : null;
    if (tier) floodSegmenterTier = tier;

    const label = tier === "robust"
        ? "Robust"
        : tier === "lightweight"
            ? "Lightweight"
            : "—";
    const backend = models.segmenter_backend ?? status.backend ?? "—";
    const m = data?.metrics || {};

    setText("flood-seg-tier-mode", mode === "forced" ? "forced" : "auto");
    setText("flood-seg-primary", tier ? label : "—");
    setText("flood-seg-backend", backend);
    setText(
        "flood-seg-skipped",
        m.segmentation_skipped == null ? "—" : (m.segmentation_skipped ? "yes" : "no")
    );
    setText("flood-seg-ms-inline", m.segmentation_ms != null ? Number(m.segmentation_ms).toFixed(1) + " ms" : "—");

    const idleNote = document.getElementById("adaptive-idle-note");
    if (idleNote) {
        if (!inferenceActive) {
            idleNote.style.display = "block";
            idleNote.innerText = status.status_note
                || "Flood segmentation and human detection tiers switch from mission context.";
        } else {
            idleNote.style.display = "none";
        }
    }

    let switchText = "none";
    const sw = status.last_switch
        || data?.model_switches?.seg_tier
        || segInfo.tier_switches?.tier;
    if (sw?.from && sw?.to) {
        const fromL = sw.from_label || sw.from;
        const toL = sw.to_label || sw.to;
        const reason = sw.reason ? ` (${sw.reason})` : "";
        switchText = `${fromL} → ${toL}${reason}`;
    }
    setText("flood-seg-tier-switch", switchText);

    const basis = segInfo.metadata?.decision_basis
        || status.last_selection?.metadata?.decision_basis
        || data?.context
        || {};
    setText("flood-ctx-alt", basis.altitude != null ? Number(basis.altitude).toFixed(1) + " m" : "—");
    setText("flood-ctx-priority", basis.priority != null ? Number(basis.priority).toFixed(2) : "—");
    setText("flood-ctx-visibility", basis.visibility != null ? Number(basis.visibility).toFixed(2) : "—");
    setText("flood-ctx-flood", basis.flood_ratio != null ? Number(basis.flood_ratio).toFixed(3) : "—");
    const clfLabel = basis.classification_label
        || data?.classification?.raw_label
        || (basis.clf_class_index === 0 ? "Flood" : basis.clf_class_index === 1 ? "Non-Flood" : null);
    setText("flood-ctx-clf", clfLabel ?? "—");
    const batt = basis.battery != null ? Number(basis.battery).toFixed(0) + "%" : "—";
    const cpu = basis.cpu_usage != null ? Number(basis.cpu_usage).toFixed(0) + "%" : "—";
    setText("flood-ctx-power", `${batt} / ${cpu}`);

    const chipLight = document.getElementById("chip-seg-light");
    const chipRobust = document.getElementById("chip-seg-robust");
    const btnAuto = document.getElementById("btn-flood-seg-auto");
    const btnForceLight = document.getElementById("btn-flood-seg-light");
    const btnForceRobust = document.getElementById("btn-flood-seg-robust");
    if (chipLight) chipLight.classList.toggle("primary", tier === "lightweight");
    if (chipRobust) chipRobust.classList.toggle("primary", tier === "robust");
    if (btnAuto) btnAuto.classList.toggle("active", mode !== "forced");
    if (btnForceLight) btnForceLight.classList.toggle("active", mode === "forced" && status.force_tier === "lightweight");
    if (btnForceRobust) btnForceRobust.classList.toggle("active", mode === "forced" && status.force_tier === "robust");
}

async function setFloodSegmenterMode(mode) {
    const body = mode === "auto" ? { mode: "auto" } : { tier: mode, force: true };
    try {
        const res = await fetch("/flood/segmenter-tier", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        renderFloodSegmenterTier(data);
        const logBox = document.getElementById("logs");
        if (logBox) {
            const msg = mode === "auto"
                ? "[FLOOD SEG TIER] mode → auto (context-aware)"
                : `[FLOOD SEG TIER] forced → ${mode}`;
            logBox.innerText += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
            logBox.scrollTop = logBox.scrollHeight;
        }
    } catch (e) {
        console.error(e);
        alert(e.message || e);
    }
}

function renderHumanDetectorTier(data) {
    const status = data?.human_detector || {};
    const models = humanModelsFromPayload(data) || {};
    const mode = status.mode || models.mode || "auto";
    const inferenceActive = Boolean(
        status.inference_active
        || data?.active_tool === "detect_human"
        || data?.active_tool === "detect_combined"
        || data?.task === "detect_human"
        || data?.task === "detect_combined"
    );

    const tier = inferenceActive
        ? (models.tier || status.tier || humanDetectorTier || "lightweight")
        : null;
    if (tier) humanDetectorTier = tier;

    const label = tier === "robust"
        ? "YOLO11s VisDrone"
        : tier === "lightweight"
            ? "YOLOv8n"
            : "—";
    const backend = models.backend ?? status.backend ?? "—";

    setText("human-tier-mode", mode === "forced" ? "forced" : "auto");
    setText("human-primary-model", tier ? label : "—");
    setText("human-backend", backend);

    const idleNote = document.getElementById("adaptive-idle-note");
    if (idleNote && inferenceActive && !document.getElementById("flood-seg-tier-mode")) {
        idleNote.style.display = "none";
    }

    let switchText = "none";
    const sw = status.last_switch
        || data?.model_switches?.tier
        || data?.human_detector?.tier_switches?.tier
        || data?.switch;
    if (sw?.from && sw?.to) {
        const reason = sw.reason ? ` (${sw.reason})` : "";
        switchText = `${sw.from_label || sw.from} → ${sw.to_label || sw.to}${reason}`;
    }
    setText("human-tier-switch", switchText);

    const basis = models.selection?.decision_basis
        || status.last_selection?.metadata?.decision_basis
        || status.metadata?.decision_basis
        || {};
    setText("human-ctx-alt", basis.altitude != null ? Number(basis.altitude).toFixed(1) + " m" : "—");
    setText("human-ctx-priority", basis.priority != null ? Number(basis.priority).toFixed(2) : "—");
    setText("human-ctx-flood", basis.flood_ratio != null ? Number(basis.flood_ratio).toFixed(3) : "—");
    setText("human-ctx-battery", basis.battery != null ? Number(basis.battery).toFixed(0) + "%" : "—");

    const chipLight = document.getElementById("chip-human-light");
    const chipRobust = document.getElementById("chip-human-robust");
    const btnAuto = document.getElementById("btn-tier-auto");
    const btnForceLight = document.getElementById("btn-tier-force-light");
    const btnForceRobust = document.getElementById("btn-tier-force-robust");
    if (chipLight) chipLight.classList.toggle("primary", tier === "lightweight");
    if (chipRobust) chipRobust.classList.toggle("primary", tier === "robust");
    if (btnAuto) btnAuto.classList.toggle("active", mode !== "forced");
    if (btnForceLight) btnForceLight.classList.toggle("active", mode === "forced" && status.force_tier === "lightweight");
    if (btnForceRobust) btnForceRobust.classList.toggle("active", mode === "forced" && status.force_tier === "robust");
}

async function setHumanDetectorMode(mode) {
    const body = mode === "auto" ? { mode: "auto" } : { tier: mode, force: true };
    try {
        const res = await fetch("/human/detector-tier", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await res.json();
        if (data.error) {
            alert(data.error);
            return;
        }
        renderHumanDetectorTier(data);
        const logBox = document.getElementById("logs");
        if (logBox) {
            const msg = mode === "auto"
                ? "[HUMAN TIER] mode → auto (context-aware)"
                : `[HUMAN TIER] forced → ${mode}`;
            logBox.innerText += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
            logBox.scrollTop = logBox.scrollHeight;
        }
    } catch (e) {
        console.error(e);
        alert(e.message || e);
    }
}

function isCombinedMode() {
    return activeTool === "detect_combined"
        || (activeTools.includes("detect_flood") && activeTools.includes("detect_human"));
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value ?? "—";
}

function setStatusBadge(el, systemStatus) {
    if (!el) return;
    const s = systemStatus || "NORMAL";
    el.innerText = s;
    if (s === "IDLE") {
        el.className = "status ok";
        el.style.background = "#1e3a5f";
        el.style.color = "#94a3b8";
        return;
    }
    el.style.background = "";
    el.style.color = "";
    if (s === "CRITICAL" || s === "ALERT") {
        el.className = "status alert";
    } else if (s === "WARNING") {
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
    if (el) {
        el.innerText = label;
        el.className =
            isCombinedMode() ? "mode-combined"
            : activeTool === "detect_flood" ? "mode-flood"
            : activeTool === "detect_human" ? "mode-human"
            : "mode-idle";
    }

    const showFlood = activeTool === "detect_flood" || isCombinedMode();
    const showHuman = activeTool === "detect_human" || isCombinedMode();
    const isActive = activeTool !== "idle";

    const appContainer = document.getElementById("app-container");
    const pageLayout = document.getElementById("page-layout");
    const rightColumn = document.getElementById("right-column");
    document.body.classList.toggle("inference-active", isActive);
    if (appContainer) appContainer.classList.toggle("inference-active", isActive);
    if (pageLayout) pageLayout.classList.toggle("inference-active", isActive);
    if (rightColumn) rightColumn.classList.toggle("hidden", !isActive);

    const activeDash = document.getElementById("active-dashboard");
    if (activeDash) activeDash.classList.toggle("hidden", !isActive);

    const liveCharts = document.getElementById("live-charts-panel");
    if (liveCharts) liveCharts.classList.toggle("hidden", !isActive);
    if (!isActive) resetLiveCharts();

    const toggleBlock = (id, show) => {
        const el = document.getElementById(id);
        if (el) el.classList.toggle("hidden", !show);
    };

    const floodModels = document.getElementById("block-flood-models");
    const humanModels = document.getElementById("block-human-models");
    if (floodModels) floodModels.classList.toggle("inactive", isActive && !showFlood);
    if (humanModels) humanModels.classList.toggle("inactive", isActive && !showHuman);

    toggleBlock("block-flood-results", showFlood);
    toggleBlock("block-human-results", showHuman);
    toggleBlock("block-flood-loc", showFlood);
    toggleBlock("block-human-loc", showHuman);
    toggleBlock("flood-context-basis", showFlood);
    toggleBlock("human-context-basis", showHuman);
    toggleBlock("flood-metric-clf", showFlood);
    toggleBlock("flood-metric-seg", showFlood);
    toggleBlock("human-metric-det", showHuman);

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
    setText("overlay-mode", segActive ? `4×4 grid (${data.active_models?.segmenter || "DeepLab"})` : "none (ResNet)");
    renderGridLocalization(data.grid);

    const resnet = document.getElementById("chip-resnet");
    if (resnet) {
        resnet.classList.toggle("primary", !segActive);
    }

    renderFloodSegmenterTier(data);

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
        return `#${n} ${h.label || "human"} conf=${h.confidence} bbox=[${h.bbox.join(",")}]`;
    }).join(" | ");
    setText("human-list", list || "No humans in frame");

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
            `<div><b>Human ${n}</b> — ` +
            `${Number(lat).toFixed(6)}, ${Number(lon).toFixed(6)} ` +
            `(conf ${h.confidence})</div>`
        );
    }).join("");
}

function renderHuman(data) {
    renderHumanPanel(data);
    renderHumanDetectorTier(data);
    const m = data.metrics || {};
    const models = data.active_models || {};
    if (!isCombinedMode()) {
        const tier = models.tier || "lightweight";
        const name = tier === "robust" ? "YOLO11s VisDrone" : "YOLOv8n";
        setText("overlay-mode", `${name} human boxes`);
    }
    setText("latency", m.total_latency_ms?.toFixed?.(1) ?? data.system?.latency_ms);
    setText("fps", m.instant_fps ?? data.system?.fps ?? "—");
    setText("det-ms", m.detection_ms ?? "—");
    renderPerfMetrics(data);
    setStatusBadge(document.getElementById("status"), data.system?.status);
}

function renderIdle(data) {
    renderHumanDetectorTier({ human_detector: { mode: "auto", inference_active: false } });
    renderFloodSegmenterTier({ flood_segmenter: { mode: "auto", inference_active: false } });
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
    const resnet = document.getElementById("chip-resnet");
    const chipSegLight = document.getElementById("chip-seg-light");
    const chipSegRobust = document.getElementById("chip-seg-robust");
    const chipLight = document.getElementById("chip-human-light");
    const chipRobust = document.getElementById("chip-human-robust");
    if (resnet) resnet.classList.remove("primary");
    if (chipSegLight) chipSegLight.classList.remove("primary");
    if (chipSegRobust) chipSegRobust.classList.remove("primary");
    if (chipLight) chipLight.classList.remove("primary");
    if (chipRobust) chipRobust.classList.remove("primary");
    const idleNote = document.getElementById("adaptive-idle-note");
    if (idleNote) idleNote.style.display = "block";
    setStatusBadge(document.getElementById("status"), "IDLE");
}

function showIdleDashboard(message) {
    updateTaskUI("idle", []);
    renderGatewayResult("Waiting for a command…");
    renderIdle({
        message:
            message ||
            "Model server ready — waiting for detect_flood / detect_human command",
    });
    const video = document.getElementById("video");
    if (video) video.removeAttribute("src");
    const rightColumn = document.getElementById("right-column");
    if (rightColumn) rightColumn.classList.add("hidden");
    const appContainer = document.getElementById("app-container");
    const pageLayout = document.getElementById("page-layout");
    document.body.classList.remove("inference-active");
    if (appContainer) appContainer.classList.remove("inference-active");
    if (pageLayout) pageLayout.classList.remove("inference-active");
    const liveCharts = document.getElementById("live-charts-panel");
    if (liveCharts) liveCharts.classList.add("hidden");
    resetLiveCharts();
}

function renderCombined(data) {
    renderFlood(data);
    renderHumanPanel(data.human_detection || data);
    renderHumanDetectorTier(data);
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

    if (data.human_detector) {
        renderHumanDetectorTier(data);
    }
    if (data.flood_segmenter || data.active_models?.segmenter_tier || data.metrics?.segmenter_tier) {
        renderFloodSegmenterTier(data);
    }

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

    recordLiveMetrics(data);

    setText("input-source", data.input_source || (data.offline ? "offline_video" : "camera"));
    setText("camera-device", data.camera?.device || "—");

    const frameB64 = data.frame_base64 || data.frame;
    const videoEl = document.getElementById("video");
    if (frameB64 && videoEl) {
        videoEl.src = "data:image/jpeg;base64," + frameB64;
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
        renderHumanDetectorTier(data);
        renderFloodSegmenterTier(data);
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

function appendLog(line) {
    const logBox = document.getElementById("logs");
    if (!logBox) return;
    logBox.innerText += `[${new Date().toLocaleTimeString()}] ${line}\n`;
    logBox.scrollTop = logBox.scrollHeight;
    const lines = logBox.innerText.split("\n");
    if (lines.length > 50) logBox.innerText = lines.slice(-50).join("\n");
}

const FLOOD_GATEWAY_TOOLS = new Set(["flood_seg", "flood_class"]);
const HUMAN_GATEWAY_TOOLS = new Set(["human_detect"]);

function gatewayModelToolsFromResponse(gw) {
    const names = [];
    const tools = gw?.tools;
    if (Array.isArray(tools)) {
        for (const step of tools) {
            if (step?.category === "model" && step.name) names.push(step.name);
        }
    }
    if (!names.length && gw?.category === "model" && gw?.tool_name) {
        names.push(gw.tool_name);
    }
    return names;
}

function mapGatewayToModelTool(modelNames) {
    if (!modelNames?.length) return null;
    const set = new Set(modelNames.map((n) => String(n).toLowerCase()));
    const hasHuman = [...set].some((n) => HUMAN_GATEWAY_TOOLS.has(n));
    const hasFlood = [...set].some((n) => FLOOD_GATEWAY_TOOLS.has(n));
    if (hasHuman && hasFlood) return "detect_combined";
    if (hasHuman) return "detect_human";
    if (hasFlood) return "detect_flood";
    return null;
}

function renderGatewayResult(html) {
    const el = document.getElementById("gateway-result");
    if (el) el.innerHTML = html;
}

function setGatewayUiBusy(busy) {
    const input = document.getElementById("gateway-prompt");
    const btn = document.getElementById("btn-gateway-send");
    if (input) input.disabled = busy;
    if (btn) btn.disabled = busy;
}

function updateGatewayStatusBar(data) {
    const dot = document.getElementById("gateway-dot");
    const cmd = document.getElementById("gateway-active-cmd");
    const llmText = document.getElementById("llm-status-text");
    const reachable = data?.reachable !== false && !data?.error;
    if (dot) {
        dot.className = "gateway-dot " + (reachable ? "ok" : "err");
    }
    if (cmd) {
        const active = data?.active_command;
        cmd.textContent = active && active !== "none"
            ? `Gateway: ${active}`
            : reachable ? "Gateway ready" : "Gateway unreachable";
    }
    if (llmText) {
        if (data?.llm_reachable === true) {
            llmText.textContent = " · online";
            llmText.style.color = "#22c55e";
        } else if (data?.llm_reachable === false) {
            llmText.textContent = " · offline (keyword fallback when LLM fails)";
            llmText.style.color = "#fbbf24";
        } else {
            llmText.textContent = "";
        }
    }
}

async function pollGatewayStatus() {
    try {
        const res = await fetch("/gateway/status");
        const data = await res.json();
        updateGatewayStatusBar(data);
    } catch (e) {
        updateGatewayStatusBar({ reachable: false, error: String(e) });
    }
}

async function submitGatewayPrompt() {
    const input = document.getElementById("gateway-prompt");
    const prompt = (input?.value || "").trim();
    if (!prompt) return;
    if (inputSource === "offline") {
        alert("Switch to Live camera first, or use Run offline for video playback.");
        return;
    }

    setGatewayUiBusy(true);
    renderGatewayResult("Sending to gateway LLM…");
    appendLog(`[GATEWAY] ${prompt}`);

    try {
        const res = await fetch("/gateway/infer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ prompt, activate: true }),
        });
        const data = await res.json();
        const gw = data.gateway || {};
        const plan = data.plan || {};
        const mapped = data.model_server_tool || plan.model_server_tool;
        const gatewayDown = data.gateway_reachable === false;
        const llmMs = gw.llm_latency_ms != null ? `${gw.llm_latency_ms} ms` : "—";

        let html = `LLM ${llmMs}`;
        if (gw.tool_name || plan.gateway_model_tools?.length) {
            const gwTools = plan.gateway_model_tools?.length
                ? plan.gateway_model_tools.join(", ")
                : `${gw.category}:${gw.tool_name}`;
            html += ` · Gateway: <code>${gwTools}</code>`;
        }
        if (plan.drone_steps?.length) {
            html += ` · Drone: ${plan.drone_steps.join(" → ")}`;
        }
        if (mapped) {
            html += ` · Model server: <span class="mapped-tool">${mapped}</span>`;
        }
        if (gw.drone_error) {
            html += `<br><span class="err">Drone: ${gw.drone_error}</span>`;
        }
        if (data.error || gw.error) {
            html += `<br><span class="err">${data.error || gw.error}</span>`;
        }
        if (data.fallback_used && mapped) {
            const via = gatewayDown
                ? "Gateway offline — keyword fallback"
                : "LLM offline — keyword fallback";
            html += `<br>${via} → <span class="mapped-tool">${mapped}</span>`;
        } else if (gw.action_taken && String(gw.action_taken).includes("llm_http_failed")) {
            html += `<br><span class="err">LLM server offline at ${data.llm_url || "port 8080"} — start llama-server / OpenAI-compatible API.</span>`;
            if (!mapped) {
                html += "<br>No keyword match — use Quick presets or start the LLM.";
            }
        } else if (gatewayDown && !mapped) {
            html += "<br>Gateway offline and no keyword match — use Quick presets or start the gateway.";
        } else if (gw.category === "none" || plan.is_none) {
            html += `<br>No model tool — ${gw.tool_name || gw.action_taken || "no action"}`;
        } else if (!mapped && plan.drone_steps?.length) {
            html += "<br>Drone step(s) only — no vision tool activated.";
        } else if (!mapped) {
            html += `<br>${gw.action_taken || "No model tool mapped."}`;
        }
        renderGatewayResult(html);

        if (gw.llm_tool_json) {
            appendLog(`[GATEWAY] plan ${gw.llm_tool_json}`);
        }
        if (plan.drone_steps?.length) {
            appendLog(`[GATEWAY] drone steps: ${plan.drone_steps.join(" → ")}`);
        }

        const toolResult = data.tool_result;
        if (data.activated && toolResult && !toolResult.error) {
            appendLog(`[MODEL] activated ${mapped}`);
            inputSource = "camera";
            setText("input-source", "camera");
            const tool = toolResult.active_tool || mapped;
            const tools = toolResult.active_tools || [];
            try {
                updateTaskUI(tool, tools);
                renderPayload(toolResult);
            } catch (uiErr) {
                console.error(uiErr);
                appendLog(`[UI] render error: ${uiErr.message || uiErr}`);
            }
            connectWebSocket();
            if (input) input.value = "";
        } else if (toolResult?.error) {
            appendLog(`[MODEL] error: ${toolResult.error}`);
        }

        pollGatewayStatus();
    } catch (e) {
        console.error(e);
        renderGatewayResult(`<span class="err">Gateway request failed: ${e.message || e}</span>`);
        updateGatewayStatusBar({ reachable: false });
    } finally {
        setGatewayUiBusy(false);
    }
}

function setLlmPromptsEnabled(enabled) {
    document.querySelectorAll(".llm-prompt").forEach((btn) => {
        btn.disabled = !enabled;
    });
}

async function activateFromLlmPrompt(tool, phrase) {
    appendLog(`[PRESET] ${phrase}`);
    await activateTool(tool);
}

async function activateTool(tool) {
    if (inputSource === "offline") {
        alert("Switch to Live camera first, or use Run offline for video playback.");
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
resetLiveCharts();
fetchStatus();
pollGatewayStatus();
setInterval(() => {
    if (inputSource === "camera" && !offlineRunning) fetchStatus();
}, 2000);
setInterval(pollActiveTask, 2500);
setInterval(() => {
    if (activeTool === "idle") pollGatewayStatus();
}, 5000);

window.addEventListener("resize", () => {
    if (activeTool !== "idle") {
        drawLiveChart("chart-latency", latencyHistory, "#00e5ff", "ms");
        drawLiveChart("chart-fps", fpsHistory, "#00ffa6", "");
    }
});
