#!/usr/bin/env bash
# Start GoPro USB preview UDP (for Drone_LLM WebRTC). Uses perception/.env (GOPRO_SERIAL).
set -euo pipefail
DRONE_LLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERCEPTION="${DRONE_PERCEPTION_PATH:-${DRONE_LLM_ROOT}/../drone-competition/perception}"
cd "$PERCEPTION"
if [[ -f .env ]]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi
export GOPRO_UDP_URL="${GOPRO_UDP_URL:-udp://@0.0.0.0:8554?overrun_nonfatal=1&fifo_size=4000&buffer_size=131072}"
exec python3 gopro_enable.py
