# Drone video benchmarks (pre-flight)

Offline evaluation before physical drone flights: **non-realtime video**, simulated altitudes, and full `live_pipeline.py` metrics.

## Quick start

```bash
cd ~/python-worker/model_server
pip install pandas matplotlib

## Video playback on Jetson (GNOME Videos error)

If Videos shows *"H.264 / AAC decoder are required but are not installed"*:

```bash
cd ~/python-worker/model_server
sudo bash tools/install_video_codecs.sh
```

Benchmark inference uses OpenCV/ffmpeg and does **not** need this for `run_drone_video_benchmark.py`.

# 1) Download benchmark videos
python3 tools/download_drone_videos.py --archive-only

# Combined rescue — COLOR (recommended for your RGB-trained models):
pip install yt-dlp
python3 tools/download_drone_videos.py --youtube 'https://www.youtube.com/watch?v=TxniKN7jL8U' --id youtube_cumbria_flood_rescue
# → youtube_cumbria_flood_rescue.mp4 (~7:38, color, 10+ humans + flood)

# Short color clip (~1:21):
python3 tools/download_drone_videos.py --youtube 'https://www.youtube.com/watch?v=rMuIEvaeQuU' --id youtube_cal_oes_flood_prep

# Kherson drone rescue (r/ukraine — color flood, ~41s):
# benchmarks/videos/kherson_drone_rescue.mp4
python3 tools/run_drone_video_benchmark.py \
  --video benchmarks/videos/kherson_drone_rescue.mp4 \
  --render-only --render-mode combined --render-altitude 50
# Smoke test without network:
# python3 tools/download_drone_videos.py --synthetic
# Pexels: browser → benchmarks/videos/pexels_flood_*.mp4
# YouTube CC: pip install yt-dlp && python3 tools/download_drone_videos.py --youtube 'URL'

# 2) Run benchmark (GPU required)
python3 tools/run_drone_video_benchmark.py \
  --video benchmarks/videos/pexels_flood_aerial_2680630.mp4 \
  --altitudes 25,50,75,100 \
  --max-frames 60 \
  --modes flood,combined

# 3) Plots + report
python3 tools/plot_drone_benchmark_report.py
```

Open `benchmarks/results/report/report.html` in a browser.

## Open-source video sources

See `drone_video_catalog.json` for URLs, licenses, and notes.

| Source | Use |
|--------|-----|
| [Pexels flood aerial](https://www.pexels.com/video/aerial-view-of-flooded-area-2680630/) | Primary auto-download |
| [Pexels urban flood](https://www.pexels.com/video/drone-footage-of-a-flooded-city-3188992/) | Urban rescue scenario |
| YouTube (Creative Commons filter) | `yt-dlp` manual download |
| [FloodNet](https://github.com/BinaLab/FloodNet) | Still images / research |

## Altitude simulation

Not real barometric altitude — **proxy** for trade-off studies:

- Reference: **50 m AGL** ≈ native 640×480 camera view
- Higher altitude → frame downscaled → **more ground coverage**, **less detail per pixel**
- Reports include `coverage_factor = altitude / 50`

## Outputs

| File | Content |
|------|---------|
| `results/drone_video_benchmark.csv` | Per-frame metrics |
| `results/videos/*_flood_50m.mp4` | Annotated review video (grid + metrics overlay) |

### Review video (matches input duration — e.g. 1:28 → 1:28)

The script probes your MP4 and sets frame count + FPS automatically.

```bash
# Full annotated video only (2177 frames @ 25fps → 1:28 for archive_flood_airfield)
python3 tools/run_drone_video_benchmark.py \
  --video benchmarks/videos/archive_flood_airfield.mp4 \
  --render-only --render-mode combined --render-altitude 50

# CSV metrics (auto stride) + full review video
python3 tools/run_drone_video_benchmark.py \
  --video benchmarks/videos/archive_flood_airfield.mp4 \
  --render-video --render-mode combined
```

Keep `--render-stride 1` and `--render-fps 0` (defaults) so output length matches input.
Long videos (e.g. `archive_flood_1955.mp4` ~16 min): use `--render-stride 2` knowing output will be shorter.

Output: `benchmarks/results/videos/<video>_<mode>_<alt>m.mp4` with:
- 4×4 flood grid + cell flood ratios
- GPS estimate on flooded cell (when geographiclib installed)
- Human bounding boxes (combined mode)
- Per-frame metrics panel (latency, FPS, memory, CPU, model state)
| `results/system_benchmark.json` | Load / unload / switch times |
| `results/report/*.png` | Graphs |
| `results/report/REPORT.md` | Summary tables |
