#!/bin/bash
# Jetson Orin: ONNX deps for YOLO → TensorRT export (no onnxruntime-gpu on aarch64).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m pip install --upgrade "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71" "onnxruntime>=1.16.0"
echo "OK — run: python3 tools/export_tensorrt.py"
