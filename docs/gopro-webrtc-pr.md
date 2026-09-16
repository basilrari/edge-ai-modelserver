# GoPro WebRTC PR copy

Use this on GitHub: https://github.com/aykumar21/Drone_LLM/compare/main...basilrari:edge-ai-modelserver:pr/gopro-webrtc-live?expand=1

## Title

Add GoPro WebRTC live view for Jetson

## Body

Adds WebRTC live camera streaming to Drone_LLM, with optional GoPro USB capture on Jetson via the existing drone-competition/perception preview pipeline.

This lets a browser watch the live GoPro feed from a Jetson while the same camera stream feeds detection tools. No separate video pipeline. Useful for field demos and remote operator view.

New endpoints:
- POST /camera/webrtc/offer
- GET /camera/status
- GET /camera/snapshot

Also includes CAMERA_BACKEND=gopro, coturn helpers under deploy/coturn/, and run.sh plus .env.example.

Based on current main, should merge cleanly. Tested on Jetson Orin with GoPro USB preview.

Out of scope: frontend/gateway proxy, removing model weights from git, SAR stack wiring.

Please review when you have a moment. Happy to adjust naming or split coturn into a follow-up.
