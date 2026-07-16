"""Generate the Model Server progress-report PPTX (python-pptx, no Node needed)."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

DOCS = Path(__file__).resolve().parent
PLOTS = DOCS.parent / "benchmarks/results/dashboard/chittur_river_rescue"
OUT = DOCS / "Model_Server_Progress_Report.pptx"

BG = RGBColor(0x0A, 0x0F, 0x1C)
CARD = RGBColor(0x14, 0x1E, 0x33)
WHITE = RGBColor(0xF1, 0xF5, 0xF9)
MUTE = RGBColor(0x94, 0xA3, 0xB8)
CYAN = RGBColor(0x38, 0xBD, 0xF8)
GREEN = RGBColor(0x4A, 0xDE, 0x80)
AMBER = RGBColor(0xFB, 0xBF, 0x24)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def slide(bg=BG):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    return s


def box(s, l, t, w, h, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    return tf


def para(tf, text, size=18, color=WHITE, bold=False, bullet=False,
         align=PP_ALIGN.LEFT, space=6, first=False, italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space)
    r = p.add_run()
    r.text = ("•  " + text) if bullet else text
    f = r.font
    f.size = Pt(size)
    f.color.rgb = color
    f.bold = bold
    f.italic = italic
    f.name = "Calibri"
    return p


def accent_bar(s):
    bar = s.shapes.add_shape(1, Inches(0.6), Inches(1.35), Inches(2.2), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = CYAN
    bar.line.fill.background()


def header(s, title, kicker=None):
    tf = box(s, 0.6, 0.45, 12.1, 0.9)
    para(tf, title, size=32, color=WHITE, bold=True, first=True)
    if kicker:
        ktf = box(s, 0.62, 1.45, 12.0, 0.5)
        para(ktf, kicker, size=15, color=CYAN, first=True)
    accent_bar(s)


def bullets(s, items, left=0.7, top=1.7, width=12.0, size=18, gap=8):
    tf = box(s, left, top, width, 5.4)
    for i, it in enumerate(items):
        if isinstance(it, tuple):
            txt, lvl = it
        else:
            txt, lvl = it, 0
        color = WHITE if lvl == 0 else MUTE
        sz = size if lvl == 0 else size - 3
        p = para(tf, txt, size=sz, color=color, bullet=(lvl == 0),
                 space=gap, first=(i == 0))
        if lvl == 1:
            p.level = 1


def table(s, rows, left, top, width, col_w=None, head=True, fs=14):
    nr, nc = len(rows), len(rows[0])
    gt = s.shapes.add_table(nr, nc, Inches(left), Inches(top),
                            Inches(width), Inches(0.4 * nr)).table
    if col_w:
        for j, w in enumerate(col_w):
            gt.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = gt.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = CARD if (i == 0 and head) else BG
            c.margin_left = Inches(0.1)
            c.margin_top = Inches(0.03)
            c.margin_bottom = Inches(0.03)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(fs)
            r.font.bold = (i == 0 and head)
            r.font.color.rgb = CYAN if (i == 0 and head) else WHITE
            r.font.name = "Calibri"
    return gt


def image(s, path, left, top, width):
    if Path(path).exists():
        s.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))
    else:
        tf = box(s, left, top, width, 1.0)
        para(tf, "[plot missing]", size=12, color=MUTE, first=True)


def mono(s, text, left=0.7, top=1.7, width=12.0, height=5.2, size=14):
    card = s.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD
    card.line.fill.background()
    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Inches(0.25)
    tf.margin_top = Inches(0.2)
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.name = "Consolas"
        r.font.color.rgb = WHITE if "►" not in line else CYAN


# ---------------------------------------------------------------- 1 Title
s = slide()
tf = box(s, 0.8, 2.2, 11.7, 2.0, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Model Server — Progress Report", size=44, color=WHITE, bold=True, first=True)
para(tf, "Context-Aware Adaptive Perception for Flood-Rescue UAVs", size=22, color=CYAN)
tf2 = box(s, 0.8, 4.5, 11.7, 2.0)
para(tf2, "Platform:  NVIDIA Jetson Orin · TensorRT · FastAPI", size=17, color=MUTE, first=True)
para(tf2, "Scope:  Onboard vision inference + adaptive model orchestration", size=17, color=MUTE)
para(tf2, "Status:  Working prototype — flood + human + combined rescue mode", size=17, color=MUTE)

# ---------------------------------------------------------------- 2 Role
s = slide()
header(s, "My Role in the Bigger Picture", "I own the Model Server node")
mono(s, (
    "User prompt --> Gateway --> LLM  (decides intent)\n"
    "                   |\n"
    "                   |---->  ► MODEL SERVER   (my scope: vision + model switch)\n"
    "                   |\n"
    "                   |---->  Rule Gate --> Drone Server  (flight control)\n"
    "                   v\n"
    "                 Reply"
), top=1.8, height=2.6, size=15)
bullets(s, [
    "Upstream (not mine): Gateway, LLM, user interface",
    "Downstream (not mine): Rule Gate, Drone Server, flight control",
    "Mine: deterministic, real-time perception + adaptive compute on the edge",
], top=4.7, gap=8)

# ---------------------------------------------------------------- 3 What it does
s = slide()
header(s, "What the Model Server Does", "Starts idle; runs inference only when a tool is activated")
table(s, [
    ["Capability", "Detail"],
    ["Flood classification", "ResNet18 — Flooded / Non-Flooded"],
    ["Flood segmentation", "DeepLabv3+ (MobileNetV3) — mask + flood ratio"],
    ["Human detection", "YOLOv8n — person bounding boxes"],
    ["Combined rescue mode", "Flood grid + humans on the same frame"],
    ["Localization", "Simulated GPS for flooded cell + each human"],
    ["Acceleration", "TensorRT engines, CUDA streams, FP16"],
    ["Adaptation", "Context-aware selection + smart segmentation"],
    ["Interfaces", "REST /tool, live WS /ws/live, dashboard"],
], left=0.7, top=1.75, width=12.0, col_w=[3.4, 8.6], fs=15)

# ---------------------------------------------------------------- 4 Architecture
s = slide()
header(s, "System Architecture — Adaptive Pipeline")
mono(s, (
    "  Camera frame\n"
    "      |\n"
    "  ► Context Evaluator   CPU/GPU/mem (real) + UAV/mission (sim)\n"
    "      |\n"
    "  ► Model Selector      which model leads + run/skip segmenter\n"
    "      |\n"
    "  ► Model Manager       lazy load + cache + TensorRT engines\n"
    "      |\n"
    "  ► Inference Engine    preprocess -> infer -> postprocess\n"
    "      |\n"
    "  ► Performance Monitor FPS · latency · power · memory\n"
    "      |\n"
    "      v  outputs --> dashboard / WebSocket / (future) ROS topics"
), top=1.7, height=5.3, size=15)

# ---------------------------------------------------------------- 5 Nodes 1
s = slide()
header(s, "Node Responsibilities (1/2)")
bullets(s, [
    "Task Session  ·  core/task_session.py",
    ("Active tool state: idle / detect_flood / detect_human / detect_combined", 1),
    ("Entry point for Gateway/LLM via POST /tool; triggers warmup + inference", 1),
    "Context Evaluator  ·  core/context_evaluator.py",
    ("System context: CPU, memory, GPU load (REAL)", 1),
    ("UAV / environment / mission context currently SIMULATED (no FC yet)", 1),
    "Model Selector  ·  core/model_selector.py",
    ("Chooses primary model from flood ratio (ResNet18 <-> DeepLabv3+)", 1),
    ("Gates the expensive segmenter on battery / CPU constraints", 1),
], top=1.7, gap=7)

# ---------------------------------------------------------------- 6 Nodes 2
s = slide()
header(s, "Node Responsibilities (2/2)")
bullets(s, [
    "Model Manager  ·  core/model_manager.py",
    ("Lazy load, cache, unload/switch; picks TensorRT engine when available", 1),
    ("Falls back to PyTorch if an engine is missing", 1),
    "Inference Engine  ·  core/inference_engine.py",
    ("Pinned-memory preprocessing, FP16 autocast, CUDA-stream execution", 1),
    ("Stable output contract (label / mask+ratio / boxes) across backends", 1),
    "Performance Monitor  ·  core/power_monitor.py + payload metrics",
    ("FPS, latency, memory, CPU, Jetson power (tegrastats VDD_IN)", 1),
    "Supporting: model_warmup, inference_gate, flood_grid (GPS), trt_runner, gpu_runtime",
], top=1.7, gap=7)

# ---------------------------------------------------------------- 7 Dashboard
s = slide()
header(s, "Live Dashboard", "Operate and observe the server in real time")
bullets(s, [
    "Input source: live camera OR offline video upload",
    "Task buttons: Test flood · Test human · Test both · Stop",
    "Live feed: annotated frames over WebSocket (flood grid, human boxes)",
    "Flood panel: status, ratio, primary model, most-flooded-cell GPS",
    "Human panel: count + per-person GPS (bbox centroid)",
    "Performance panel: latency, FPS, memory, CPU",
    "Power panel: idle / inference / extra watts",
    "Offline panel: stride, altitude, export path -> auto report + plots",
], top=1.7, gap=9)

# ---------------------------------------------------------------- 8 Methodology
s = slide()
header(s, "Offline Benchmark — Methodology", "Reproducible per-frame evaluation on real footage")
bullets(s, [
    "Upload video -> run a tool -> export per-video folder",
    "Outputs: overlay MP4, metrics.csv (one row/frame), plots, HTML/MD report",
    "Stride = 1  ->  genuine time series (no interpolation)",
    "12 clips processed (archive, YouTube, drone rescue, synthetic)",
], top=1.8, gap=12)
tf = box(s, 0.7, 4.6, 12.0, 1.0)
para(tf, "Chosen for deep analysis:  chittur_river_rescue  (combined mode)",
     size=20, color=GREEN, bold=True, first=True)

# ---------------------------------------------------------------- 9 Why suitable
s = slide()
header(s, "Why chittur_river_rescue Is the Best Clip", "Only clip exercising the full rescue pipeline at once")
table(s, [
    ["Property", "Value", "Why it matters"],
    ["Mode", "detect_combined", "Flood AND human paths both active"],
    ["Mean flood ratio", "0.57 (0.46-0.70)", "Above 0.20 -> segmenter stays active"],
    ["Mean humans/frame", "1.68 (1-3)", "Multiple people -> per-human GPS"],
    ["Mean FPS", "34.5", "Highest among combined+human clips"],
    ["Steady p95 latency", "30.1 ms", "Tight, stable distribution"],
], left=0.7, top=1.75, width=12.0, col_w=[2.9, 3.3, 5.8], fs=15)
tf = box(s, 0.7, 5.7, 12.0, 1.2)
para(tf, "Represents the actual mission: people stranded in flood water — "
         "flood mapping + human localization simultaneously.",
     size=15, color=AMBER, italic=True, first=True)

# ---------------------------------------------------------------- 10 Overview plot
s = slide()
header(s, "Results — Per-Frame Overview (chittur)",
       "120 frames · combined mode · 50 m simulated altitude · stride 1")
image(s, PLOTS / "overview_per_frame.png", left=1.4, top=1.8, width=10.5)

# ---------------------------------------------------------------- 11 Latency breakdown
s = slide()
header(s, "Results — Latency Breakdown (Steady State)")
table(s, [
    ["Stage", "Mean (ms)", "Backend"],
    ["Classification (ResNet18)", "5.6", "TensorRT"],
    ["Segmentation (DeepLabv3+)", "11.0", "TensorRT"],
    ["Human + compose (YOLOv8n)", "~14.5", "TensorRT"],
    ["Total inference", "31.2", "—"],
    ["Total latency", "32.1", "end-to-end"],
], left=0.7, top=1.75, width=8.2, col_w=[4.4, 2.0, 1.8], fs=15)
bullets(s, [
    "Steady-state: mean 28.7 · p50 28.6 · p95 30.1 ms",
    "Sustained ~34.5 FPS in full combined mode",
    "Segmentation = largest single cost -> the target of adaptation",
], left=9.2, top=1.9, width=3.7, size=14, gap=10)

# ---------------------------------------------------------------- 12 Tradeoff cold start
s = slide()
header(s, "Tradeoff 1 — Cold Start vs Steady State")
image(s, PLOTS / "series_total_latency_ms.png", left=7.4, top=1.9, width=5.4)
bullets(s, [
    "First frame: 436 ms   vs   steady ~29 ms",
    "Cause: one-time model load + GPU warmup",
    ("clf 134 ms + seg 53 ms + human 3.4 ms = 191 ms", 1),
    "Single outlier inflates mean (32.1 ± 37.2 ms)",
    "Steady-state p95 is only 30 ms — true operating point",
    "Takeaway: warm up before flight; report steady-state",
], left=0.7, top=2.0, width=6.4, size=16, gap=12)

# ---------------------------------------------------------------- 13 Tradeoff switching
s = slide()
header(s, "Tradeoff 2 — Logical Switch vs Physical Swap", "The key design decision")
table(s, [
    ["Approach", "Latency", "Note"],
    ["Logical primary switch", "0.016 ms avg", "Both models stay loaded; change leader"],
    ["Physical model swap", "255 ms", "Unload clf + load seg (worst case)"],
], left=0.7, top=1.8, width=12.0, col_w=[3.4, 2.6, 6.0], fs=15)
bullets(s, [
    "36 switches over 50 steps cost ~0.016 ms each — effectively free",
    "Physical swap is ~16,000x slower",
    "Decision: keep both flood models resident, switch logically -> no stutter",
    "Cost: higher VRAM (acceptable on Orin for 2 small models)",
], top=3.4, gap=10)

# ---------------------------------------------------------------- 14 Tradeoff smart seg
s = slide()
header(s, "Tradeoff 3 — Smart Segmentation (Accuracy vs Speed)",
       "DeepLab is the most expensive stage — run it only when needed")
bullets(s, [
    "Policy: skip segmentation when classifier says 'dry' + periodic refresh + hysteresis",
    "Dry scenes: classifier-only (~6 ms) instead of clf+seg (~17 ms)",
    "Flood scenes (like chittur): segmenter forced active -> full accuracy",
    "Tradeoff: small risk of late flood onset, bounded by SEG_INTERVAL refresh",
    "Net effect: spend GPU budget where flood actually exists",
], top=1.9, gap=14)

# ---------------------------------------------------------------- 15 Tradeoff altitude
s = slide()
header(s, "Tradeoff 4 — Altitude vs Resolution")
image(s, PLOTS / "tradeoff_altitude_latency.png", left=7.6, top=1.9, width=5.2)
bullets(s, [
    "Higher altitude -> wider ground coverage per frame",
    "But -> coarser pixels (GSD) -> small humans / flood edges shrink",
    "Lower altitude -> sharper detail, less area, more passes",
    "Benchmark simulates altitude via resize (pre-flight study)",
    "Implication: choose altitude per mission (search wide vs inspect close)",
], left=0.7, top=2.0, width=6.6, size=16, gap=12)

# ---------------------------------------------------------------- 16 Resources
s = slide()
header(s, "Resource Usage (chittur, combined mode)")
table(s, [
    ["Resource", "Value"],
    ["Memory (RSS)", "~1.22 GB stable (no leak across 120 frames)"],
    ["CPU", "~26% mean (12-45%)"],
    ["Flood ratio", "0.46-0.70 (consistently flooded)"],
    ["Humans", "1-3 per frame"],
    ["Power", "not captured in this offline run (tegrastats live-only)"],
], left=0.7, top=1.8, width=12.0, col_w=[3.0, 9.0], fs=15)
bullets(s, [
    "Memory flat -> clean lazy-load + cache, no per-frame growth",
    "CPU headroom remains for the rest of the onboard stack",
], top=4.6, gap=10)

# ---------------------------------------------------------------- 17 Cross video
s = slide()
header(s, "Cross-Video Sanity Check", "Stable behavior across very different footage")
table(s, [
    ["Video", "Frames", "FPS", "p95 ms", "Flood", "Humans"],
    ["chittur_river_rescue", "120", "34.5", "30.1", "0.57", "1.68"],
    ["kherson_drone_rescue", "1033", "37.4", "28.8", "0.31", "0.0"],
    ["archive_flood_airfield", "2177", "21.8", "55.5", "0.34", "0.02"],
    ["archive_flood_1955", "485", "26.9", "50.0", "0.73", "0.0"],
    ["youtube_cumbria_rescue", "377", "23.0", "59.0", "0.15", "2.29"],
], left=0.7, top=1.8, width=12.0, col_w=[4.0, 1.8, 1.6, 1.8, 1.4, 1.4], fs=14)
tf = box(s, 0.7, 5.4, 12.0, 1.3)
para(tf, "Higher-res / busier clips cost more per frame, but the pipeline stays "
         "stable (no crashes, no memory growth) over 2000+ frame runs.",
     size=15, color=AMBER, italic=True, first=True)

# ---------------------------------------------------------------- 18 Takeaways
s = slide()
header(s, "Key Takeaways")
bullets(s, [
    "Real-time on the edge: ~34 FPS, p95 30 ms in full combined rescue mode",
    "Adaptation works: logical switching is free; smart segmentation saves the most expensive stage",
    "Stable: flat memory, tight latency over long runs",
    "Honest scope: UAV state + GPS are SIMULATED; vision + compute adaptation are REAL",
    "Clean boundaries: drop-in perception node for the Gateway/LLM stack",
], top=1.9, gap=16)

# ---------------------------------------------------------------- 19 Limitations
s = slide()
header(s, "Limitations & Next Steps")
bullets(s, [
    "Limitations",
    ("Lightweight test weights -> results validate the pipeline, not operational accuracy", 1),
    ("UAV context simulated; power not logged offline", 1),
    ("One model per task slot", 1),
    "Planned",
    ("Real context schema + mission-linked policy + decision logging", 1),
    ("Closed loop: Performance Monitor -> Model Selector (auto-degrade under load)", 1),
    ("Model registry + 2nd-tier models (MobileNetV2, UNet-Light, YOLOv8s)", 1),
    ("Flight integration (Pixhawk/ROS 2) for real GPS — interface-first", 1),
], top=1.7, gap=8)

# ---------------------------------------------------------------- 20 Thank you
s = slide()
tf = box(s, 0.8, 2.4, 11.7, 2.5, anchor=MSO_ANCHOR.MIDDLE)
para(tf, "Thank You", size=46, color=WHITE, bold=True, first=True)
para(tf, "Model Server — adaptive, real-time, edge perception for flood-rescue UAVs",
     size=18, color=CYAN)
tf2 = box(s, 0.8, 4.8, 11.7, 1.5)
para(tf2, "Repo:  github.com/aykumar21/Drone_LLM", size=15, color=MUTE, first=True)
para(tf2, "Stack:  Python · FastAPI · PyTorch · TensorRT · CUDA · YOLOv8 · Jetson Orin",
     size=15, color=MUTE)

prs.save(str(OUT))
print(f"Saved {OUT} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
