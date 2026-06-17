#!/usr/bin/env python3
"""Download open-source drone/flood sample videos for offline benchmarks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "benchmarks/drone_video_catalog.json"
OUT_DIR = ROOT / "benchmarks/videos"

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _urls_for_entry(entry: dict) -> list[str]:
    urls: list[str] = []
    if entry.get("archive_identifier") and entry.get("archive_file"):
        ident = entry["archive_identifier"]
        fname = entry["archive_file"]
        urls.append(
            "https://archive.org/download/"
            + ident
            + "/"
            + urllib.parse.quote(fname)
        )
    if entry.get("direct_download"):
        urls.append(entry["direct_download"])
    urls.extend(entry.get("fallback_urls") or [])
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def download_url(url: str, dest: Path, referer: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    hdrs = {**DEFAULT_HEADERS}
    if referer:
        hdrs["Referer"] = referer
    elif "pexels.com" in url:
        hdrs["Referer"] = "https://www.pexels.com/"

    print(f"[DOWNLOAD] {url}")
    print(f"           → {dest}")

    req = urllib.request.Request(url, headers=hdrs)
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
        if len(data) < 50_000:
            raise RuntimeError(f"Response too small ({len(data)} bytes)")
        dest.write_bytes(data)

    print(f"[OK] {dest.stat().st_size / 1e6:.1f} MB")


def download_curl(url: str, dest: Path, referer: str | None = None) -> None:
    if not shutil.which("curl"):
        raise RuntimeError("curl not found")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fL",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-A",
        DEFAULT_HEADERS["User-Agent"],
        "-o",
        str(dest),
        url,
    ]
    if referer:
        cmd.extend(["-e", referer])
    print(f"[CURL] {url}")
    subprocess.run(cmd, check=True)
    if dest.stat().st_size < 50_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError("curl download too small")
    print(f"[OK] {dest.stat().st_size / 1e6:.1f} MB")


def download_entry(entry: dict) -> Path | None:
    if entry.get("manual_only") or (
        not _urls_for_entry(entry) and not entry.get("synthetic")
    ):
        return None

    dest = OUT_DIR / f"{entry['id']}.mp4"
    if dest.exists() and dest.stat().st_size > 50_000:
        print(f"[SKIP] exists: {dest}")
        return dest

    if entry.get("synthetic"):
        return create_synthetic_video(dest, entry)

    page = entry.get("page")
    referer = page if page and "pexels" in page else None
    last_err: Exception | None = None
    for url in _urls_for_entry(entry):
        try:
            download_url(url, dest, referer=referer)
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as exc:
            last_err = exc
            print(f"[FAIL] {url}: {exc}")
            dest.unlink(missing_ok=True)
            try:
                download_curl(url, dest, referer=referer)
                return dest
            except (subprocess.CalledProcessError, RuntimeError) as exc2:
                print(f"[FAIL] curl: {exc2}")
                dest.unlink(missing_ok=True)

    print(f"[FAIL] {entry['id']}: all URLs failed ({last_err})")
    return None


def create_synthetic_video(dest: Path, entry: dict) -> Path:
    """Generate a short MP4 with water-like regions for pipeline smoke tests."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv required for --synthetic") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 480
    fps = 15
    frames = int(entry.get("synthetic_frames", 90))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed (codec mp4v)")

    print(f"[SYNTHETIC] {frames} frames @ {w}x{h} → {dest}")
    for i in range(frames):
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:, :, 0] = 90
        base[:, :, 1] = 120
        base[:, :, 2] = 70
        # Brown flood band + blue water patch (moves slightly)
        y0 = int(h * 0.35 + 20 * np.sin(i / 10))
        y1 = min(h, y0 + int(h * 0.35))
        base[y0:y1, :, 0] = 40
        base[y0:y1, :, 1] = 80
        base[y0:y1, :, 2] = 140
        x0 = int(w * 0.1 + i * 2) % (w // 2)
        cv2.rectangle(base, (x0, y0), (x0 + w // 3, y1), (180, 100, 40), -1)
        cv2.putText(
            base,
            "synthetic flood bench",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(base)
    writer.release()
    print(f"[OK] synthetic {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def download_yt_dlp(url: str, dest_id: str) -> Path | None:
    if not shutil.which("yt-dlp"):
        print("[SKIP] yt-dlp not installed (pip install yt-dlp)")
        return None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_tpl = str(OUT_DIR / f"{dest_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/best[height<=720]",
        "--merge-output-format",
        "mp4",
        "-o",
        out_tpl,
        url,
    ]
    print(f"[YT-DLP] {url}")
    subprocess.run(cmd, check=True)
    mp4 = [p for p in OUT_DIR.glob(f"{dest_id}.*") if p.suffix == ".mp4"]
    if mp4:
        print(f"[OK] {mp4[0]}")
        return mp4[0]
    return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--youtube", type=str, help="YouTube URL (CC-licensed flood/drone)")
    parser.add_argument("--id", type=str, default="youtube_custom")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic flood-style test MP4 (no network)",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Skip Pexels/Coverr; download Internet Archive entries only",
    )
    parser.add_argument(
        "--rescue-only",
        action="store_true",
        help="Download clips tagged for combined flood+human rescue testing",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.youtube:
        path = download_yt_dlp(args.youtube, args.id)
        sys.exit(0 if path else 1)

    if args.synthetic:
        path = create_synthetic_video(
            OUT_DIR / "synthetic_flood_bench.mp4",
            {"id": "synthetic_flood_bench", "synthetic_frames": 120},
        )
        print(f"\n[DONE] {path}")
        return

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    downloaded: list[Path] = []

    for entry in catalog["videos"]:
        if args.rescue_only and not entry.get("combined_rescue"):
            continue
        if args.archive_only and not entry.get("archive_identifier"):
            continue
        if entry.get("pexels_only") and args.archive_only:
            continue
        if entry.get("youtube_url") and not args.youtube:
            continue
        path = download_entry(entry)
        if path:
            downloaded.append(path)

    if not downloaded:
        print("\n" + "=" * 60)
        print("No videos downloaded.")
        print("=" * 60)
        print("\nReliable on Jetson (no Pexels):")
        print("  python3 tools/download_drone_videos.py --archive-only")
        print("  python3 tools/download_drone_videos.py --synthetic")
        print("\nPexels (often 403 from servers) — use browser:")
        for entry in catalog["videos"]:
            if entry.get("page") and "pexels" in (entry.get("page") or ""):
                print(f"  {entry['page']}")
                print(f"    → {OUT_DIR}/{entry['id']}.mp4")
        print("\nYouTube CC:")
        print("  pip install yt-dlp")
        print("  python3 tools/download_drone_videos.py --youtube 'URL'")
        sys.exit(1)

    print(f"\n[DONE] {len(downloaded)} video(s) in {OUT_DIR}")
    for p in downloaded:
        print(f"  {p}")


if __name__ == "__main__":
    main()
