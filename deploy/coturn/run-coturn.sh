#!/usr/bin/env bash
# Run coturn for remote internet WebRTC viewing (after configure-coturn.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${ROOT}/turnserver.generated.conf"

if [[ ! -f "$CONF" ]]; then
  "${ROOT}/configure-coturn.sh" > /tmp/coturn-env.txt
  # shellcheck source=/dev/null
  source /tmp/coturn-env.txt 2>/dev/null || true
fi

if ! command -v turnserver >/dev/null 2>&1; then
  echo "install coturn: sudo apt install -y coturn" >&2
  exit 1
fi

exec turnserver -c "$CONF" -o
