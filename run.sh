#!/usr/bin/env bash
# Start Drone_LLM model server (inference API + WebRTC live view for SAR frontend).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${ROOT}/.env"
  set +a
fi

export GATEWAY_URL="${GATEWAY_URL:-http://127.0.0.1:3000}"
HOST="${MODEL_SERVER_HOST:-0.0.0.0}"
PORT="${MODEL_SERVER_PORT:-8000}"

PY="${DRONE_LLM_PYTHON:-python3}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
fi

exec "$PY" -m uvicorn main:app --host "$HOST" --port "$PORT"
