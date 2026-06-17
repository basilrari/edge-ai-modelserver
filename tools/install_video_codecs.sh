#!/bin/bash
# Ubuntu 22.04 / Jetson: H.264 + AAC decoders for GNOME Videos / Totem.
# Fixes: "MPEG-4 AAC decoder, H.264 (Constrained Baseline Profile) decoder are required"
set -euo pipefail

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run with sudo:"
  echo "  sudo bash tools/install_video_codecs.sh"
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y \
  gstreamer1.0-libav \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-ugly \
  libavcodec-extra

echo ""
echo "Verifying GStreamer decoders..."
for elem in avdec_h264 faad; do
  if gst-inspect-1.0 "$elem" >/dev/null 2>&1; then
    echo "  OK  $elem"
  else
    echo "  MISSING  $elem"
    ok=0
  fi
done

echo ""
echo "Done. Re-open the MP4 in Videos (Totem) or log out/in if still failing."
