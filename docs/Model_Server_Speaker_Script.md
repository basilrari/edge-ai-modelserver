# Model Server — Speaker Script (Progress Presentation)

**Total time:** ~12–15 minutes (adjust sections as needed)  
**Tip:** Show live dashboard or `chittur_river_rescue` overlay video during slides 8–10.

---

## Opening (~1 min)

> “Good [morning/afternoon]. I’m presenting the **Model Server** — the onboard vision system for our flood-rescue drone project.
>
> In simple terms: it’s the **eyes and analyst** on the Jetson computer. It looks at camera video, detects **flood water** and **people**, and sends structured results back to the rest of the system.
>
> **Important:** it does **not** fly the drone. My scope is perception and adaptive AI only.”

---

## Slide: My role in the bigger picture (~1.5 min)

> “The full stack has several parts. The **user** speaks in natural language. The **Gateway** and **LLM** turn that into a command like ‘start flood and human detection.’
>
> That command comes to **my Model Server**. I run the camera and AI models and return detections, metrics, and annotated images.
>
> Separately, the **Rule Gate** and **Drone Server** handle **flight** — arm, move, return-to-launch. They get safety-checked commands; they don’t run computer vision.
>
> So: **Gateway decides what to look for; Model Server decides how to run the AI; Drone Server decides where to fly.**”

---

## Slide: What the Model Server does (~1.5 min)

> “Technically it’s a **FastAPI service** on Jetson Orin. It starts **idle** — no heavy GPU work until it receives a tool command.
>
> We support three detection modes:
> - **detect_flood** — classification plus segmentation and a flood grid  
> - **detect_human** — YOLO person detection  
> - **detect_combined** — both on the **same frame** — our rescue scenario  
>
> Models used today: **ResNet18**, **DeepLabv3+**, and **YOLOv8n**, accelerated with **TensorRT** where possible.
>
> Outputs go out as **JSON plus a JPEG overlay**, either over HTTP or a **WebSocket live stream** for the dashboard or Gateway.”

---

## Slide: Architecture / pipeline (~2 min)

> “Let me walk through **one frame** in plain language.
>
> **First**, a command arrives — for example POST `/tool` with detect_combined. **Task Session** records the active mode and triggers **model warmup** — loading ResNet, DeepLab, and YOLO into GPU memory.
>
> **Second**, we grab a **camera frame** — one snapshot from USB video or from an uploaded MP4 in offline mode.
>
> **Third**, the **Context Evaluator** collects a situation snapshot: real CPU and memory from the Jetson; simulated battery and altitude for now until we integrate the flight controller.
>
> **Fourth**, the **Model Selector** decides strategy: which model ‘leads’ the flood decision, and whether to run expensive **segmentation**. If the scene looks dry, we **skip DeepLab** most frames — that saves about three seconds per frame on Jetson.
>
> **Fifth**, **Model Manager** supplies cached models, preferring **TensorRT engines** built on-device.
>
> **Sixth**, the **Inference Engine** preprocesses the image, runs FP16 inference on **CUDA streams** — flood and human can run in **parallel** in combined mode.
>
> **Seventh**, **flood_grid** adds the 4×4 grid, red dot on the most flooded cell, and **GPS estimates**. For humans, we attach **per-person GPS** from each bounding box centroid. GPS uses **simulated drone reference** today — clearly labeled on the dashboard.
>
> **Finally**, the **Performance Monitor** records latency, FPS, memory, and Jetson power, and we send the full payload back.”

---

## Slide: Node responsibilities (~1 min — optional, shorten if short on time)

> “Supporting pieces worth naming:
> - **inference_gate** — only one GPU inference at a time, prevents pile-up from WebSocket  
> - **model_warmup** — loads only models needed for the active tool  
> - **trt_runner** and **gpu_runtime** — TensorRT and parallel CUDA streams  
> - **segment_policy** — smart rules for when to run DeepLab  
>
> These are modular so we can add more models later without rewriting the pipeline.”

---

## Slide: Dashboard (~1 min)

> “The **dashboard** is our operator view — same APIs the Gateway would use.
>
> You can switch **live camera** or **offline video**, press Test flood / human / both, and see annotated video, flood ratio, model switches, per-person GPS, latency, and power.
>
> Offline mode exports a full folder: overlay MP4, per-frame CSV, plots, and an HTML report — useful for thesis evaluation without flying.”

---

## Slide: Why chittur_river_rescue (~1.5 min)

> “For offline evaluation we processed twelve flood clips. The best showcase is **chittur_river_rescue**.
>
> **Why this one?** It’s the only clip that exercises the **full rescue pipeline** together: high flood ratio — around **57%** on average — and **one to three people** per frame.
>
> It matches the real mission: **people stranded in flood water**, not just water alone or empty aerial footage.
>
> Performance on Jetson: steady **about 34 frames per second**, **p95 latency about 30 milliseconds** in combined mode with TensorRT. Memory stayed flat around **1.2 gigabytes** over 120 frames — no leak.”

---

## Slide: Tradeoffs (~2 min)

> “Four tradeoffs we measured and designed for:
>
> **One — cold start vs steady state.** The first frame after activation can take **400-plus milliseconds** because models load into GPU. Steady operation is **about 29 milliseconds**. Lesson: **warm up before flight**; report steady-state numbers in evaluation.
>
> **Two — logical vs physical model switch.** Keeping both flood models loaded and switching which one ‘leads’ costs **0.016 milliseconds**. Physically unloading and loading models costs **255 milliseconds**. We keep models resident and switch logically — no stutter.
>
> **Three — smart segmentation.** DeepLab is the most expensive stage. We skip it when the classifier says dry, with periodic refresh. Tradeoff: tiny risk of missing early flood onset, bounded by refresh interval. Net: GPU budget spent where water actually appears.
>
> **Four — altitude vs resolution.** Higher altitude sees more ground but coarser detail. We simulate altitude in benchmarks to study this before flight. Search wide vs inspect close is a mission choice.”

---

## Slide: Limitations & next steps (~1.5 min)

> “Being transparent about limitations:
>
> - Models are **lightweight test weights** — results validate the **pipeline and adaptation**, not operational flood accuracy yet.  
> - **UAV state and GPS are simulated** on the dashboard; vision and compute metrics are real.  
> - **Performance feedback loop** — auto-downgrade models under load — is planned, not fully closed yet.  
>
> **Next steps:** real context from Pixhawk when integrated; model registry with second-tier models like MobileNet and YOLO-small; decision logging for thesis; ROS bridge for Gateway when the team defines topics.”

---

## Closing (~30 sec)

> “In summary: we have a working **edge Model Server** — idle until commanded, adaptive flood cascade, parallel rescue mode, live dashboard, and reproducible offline benchmarks.
>
> It’s designed as a **drop-in perception node** for the Gateway and LLM stack, with clear boundaries from flight control.
>
> Thank you — happy to take questions.”

---

## Q&A — short answers (cheat sheet)

**Q: Does it fly the drone?**  
> No. Perception only. Drone Server + Rule Gate handle flight.

**Q: Is GPS real?**  
> Simulated reference today. Real Pixhawk integration is planned; math and dashboard fields are ready.

**Q: Why not always run the best model?**  
> DeepLab is slow on Jetson. Adaptive policy skips it when safe, keeping real-time FPS.

**Q: What input does it need besides camera?**  
> A JSON tool command (from Gateway or dashboard). Optionally env toggles for TensorRT and smart segment.

**Q: GoPro only — enough?**  
> For live AI, GoPro must stream to Jetson. Record-only GoPro works for offline MP4 testing.

**Q: How does Gateway connect?**  
> Same as dashboard: POST `/tool`, WebSocket `/ws/live`, GET `/status`. ROS adapter can wrap these later.

---

## Timing cheat sheet

| Section | Minutes |
|---------|---------|
| Opening + role | 2.5 |
| What it does + architecture | 3.5 |
| Dashboard + chittur results | 2.5 |
| Tradeoffs | 2 |
| Limitations + close | 2 |
| **Total** | **~12–13** |

Add 2–3 minutes for live demo or one plot slide if allowed.
