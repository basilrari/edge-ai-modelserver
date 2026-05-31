let ws = null;

function setStatusBadge(el, systemStatus) {
    el.innerText = systemStatus;
    if (systemStatus === "CRITICAL" || systemStatus === "ALERT") {
        el.className = "status alert";
        el.style.background = systemStatus === "CRITICAL" ? "#7f1d1d" : "#b91c1c";
    } else if (systemStatus === "WARNING") {
        el.className = "status alert";
        el.style.background = "#78350f";
    } else {
        el.className = "status ok";
        el.style.background = "#064e3b";
    }
}

function updateModelChips(primaryKey, switches, segmentationActive) {
    const resnet = document.getElementById("chip-resnet");
    const deeplab = document.getElementById("chip-deeplab");
    resnet.classList.remove("active", "primary", "switching");
    deeplab.classList.remove("active", "primary", "switching");
    resnet.classList.add("active");
    deeplab.classList.add("active");

    if (segmentationActive || primaryKey === "deeplabv3plus") {
        deeplab.classList.add("primary");
    } else {
        resnet.classList.add("primary");
    }

    if (switches && switches.primary) {
        const chip = switches.primary.to === "resnet18" ? resnet : deeplab;
        chip.classList.add("switching");
        setTimeout(() => chip.classList.remove("switching"), 1600);
        return `${switches.primary.from} → ${switches.primary.to}`;
    }
    return "";
}

function setText(id, value) {
    const el = document.getElementById(id);
    if (el) el.innerText = value ?? "—";
}

function renderPayload(data) {
    if (data.error) throw new Error(data.error);

    const m = data.metrics || {};
    const p = data.power || {};
    const segActive = Boolean(
        data.segmentation_active ?? data.segmentation?.active
    );
    const classification = segActive
        ? "Flooded"
        : (data.classification?.label || "Non-Flooded");
    const rawClass = data.classification?.raw_label || "—";
    const floodRatio = data.segmentation?.flood_ratio ?? 0;
    const systemStatus = data.system?.status || "UNKNOWN";
    const primaryKey = data.primary_model || "resnet18";
    const active = data.active_models || {};

    setText("classification", classification);
    setText("raw-classification", rawClass);
    setText("flood_ratio", Number(floodRatio).toFixed(3));
    setText("primary-model", active.primary || primaryKey);
    setText("camera-device", data.camera?.device || "unknown");
    setText("overlay-mode", data.segmentation_active ? "4×4 grid (DeepLab)" : "none (ResNet)");

    const gps = data.grid?.gps_text;
    setText("gps-text", gps || "");

    setText("latency", (m.total_latency_ms ?? data.system?.latency_ms)?.toFixed?.(1) ?? m.total_latency_ms);
    setText("fps", m.fps ?? "—");
    setText("instant-fps", m.instant_fps ?? data.system?.fps ?? "—");
    setText("clf-ms", m.classification_ms);
    setText("seg-ms", m.segmentation_ms);
    setText("switch-ms", m.model_switch_latency_ms);
    setText("memory-mb", m.memory_mb);
    setText("peak-memory", m.peak_memory_mb);
    setText("cpu-pct", m.cpu_percent);
    setText("peak-cpu", m.peak_cpu_percent);
    setText("clf-load", m.clf_load_ms);
    setText("seg-load", m.seg_load_ms);

    setText("idle-power", p.idle_power_w ?? m.idle_power_w);
    setText("inference-power", p.inference_power_w ?? m.inference_power_w);
    setText("extra-power", p.extra_power_w ?? m.extra_power_w);
    setText("peak-power", p.peak_inference_power_w ?? m.peak_inference_power_w);

    setStatusBadge(document.getElementById("status"), systemStatus);

    const switchText = updateModelChips(
        primaryKey,
        data.model_switches,
        data.segmentation_active
    );
    setText("model-switch", switchText || "none (stable)");

    const frameB64 = data.frame_base64 || data.frame;
    if (frameB64) {
        document.getElementById("video").src = "data:image/jpeg;base64," + frameB64;
    }

    const logBox = document.getElementById("logs");
    logBox.innerText += `[${new Date().toLocaleTimeString()}] ${data.log || ""}\n`;
    logBox.scrollTop = logBox.scrollHeight;
    const lines = logBox.innerText.split("\n");
    if (lines.length > 50) logBox.innerText = lines.slice(-50).join("\n");
}

async function pollDetectFlood() {
    try {
        const response = await fetch("/detect_flood", { method: "POST" });
        renderPayload(await response.json());
    } catch (err) {
        console.error(err);
        document.getElementById("status").innerText = "ERROR";
        document.getElementById("status").className = "status alert";
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

connectWebSocket();
setInterval(pollDetectFlood, 3000);
