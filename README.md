# Pylox Vision — Multi-Camera AI Security Platform

Production AI security system for commercial sites. Deployed at a pilot installation (9 cameras, 74,000+ events processed in the live database).

**This is a sanitized public release of the backend engine.** Client-specific data, credentials, and integrations have been stripped — placeholders are used where real values lived.

---

## Architecture

```
Cameras (RTSP) → go2rtc → Frigate NVR (TRT YOLOv8 @ 5fps)
                                    │
                                    ▼ MQTT events
                            pylox-v2 FastAPI engine :3450
                                    ├─ live_pose       (TRT YOLOv8n-pose FP16, 6ms inference)
                                    ├─ people/reid     (OSNet 512-dim embeddings, cosine sim)
                                    ├─ sessions        (visit bucketing)
                                    ├─ timeline        (events, zigbee, doors)
                                    ├─ analytics       (SQL aggregations)
                                    ├─ gemini          (summaries, vision auto-map, NLP search)
                                    ├─ gemini_budget   ($5/day spend cap, SQLite-backed)
                                    ├─ anomaly         (behavior scoring)
                                    ├─ pose            (skeleton extraction pipeline)
                                    ├─ spatial         (3D camera mapping)
                                    ├─ deterrence      (Lorex speaker control)
                                    ├─ alerts          (SMS + push + webhook fan-out)
                                    ├─ sensors         (Zigbee contact sensors)
                                    ├─ training        (on-device ReID learning loop)
                                    ├─ watchdog        (health monitor)
                                    └─ police          (incident report API)
                                    │
                                    ▼ uvicorn :3450 (FastAPI + WebSocket)
                                    │
                          pylox-tunnel (Node.js WS relay) → Vultr relay
                                    │
                                    ▼
                            iOS native app (SwiftUI)
                              • 9-stream HLS live grid
                              • Real-time skeleton overlay via WebSocket
                              • Live Activities + Dynamic Island
                              • Per-person intelligence reports
```

## Hard problems solved

- **WebSocket instead of SSE for live pose stream** — the reverse WS tunnel buffered `text/event-stream` bodies before forwarding. Swapped to true WebSocket which proxies frame-by-frame.
- **TRT-compiled YOLOv8n-pose** — host PyTorch can't use CUDA due to JetPack/CUDA version mismatch. Exported ONNX on CPU, compiled FP16 TRT engine with `trtexec --fp16`. 25× speedup (150 ms → 6 ms).
- **OSNet ReID without torchreid** — torchreid's scipy/numpy/gdown dep cascade breaks on Jetson. Vendored just `osnet_model.py` as a standalone file, loaded state_dict directly.
- **Tiered ReID thresholds** — 0.65 for named persons, 0.78 for unnamed. Single threshold fragmented known people into 5+ clusters across angles.
- **Skeleton aspect-fill math** — iOS `AVPlayerLayer.resizeAspectFill` crops 4:3 source in 16:9 tile. Recomputed skeleton projection to match AVPlayer's scale/offset + clipped keypoints outside visible rect.
- **go2rtc direct RTSP read** — Frigate's `/latest.jpg` caps at 5 fps. Spawned per-camera readers via `http://localhost:1984/api/stream.mp4` for real 25+ fps frames only when a person event is active.
- **Gemini budget gate** — central `gemini_budget.py` wraps all 6 Gemini call sites with `can_spend() + record_usage()`. $5/day hard cap, SQLite-persisted across restarts.

## Tech stack

**Models / AI:**
- YOLOv8n-pose FP16 TRT engine (Jetson GPU, 17-kpt COCO)
- OSNet x1_0 (torchreid weights, 512-dim L2-normalized embeddings)
- Frigate 0.17 (YOLO @ 5fps, TRT)
- Gemini 2.5 Pro (summaries, fleet briefings, vision auto-map)

**Backend:**
- Python 3.10, FastAPI + uvicorn
- paho-mqtt, httpx, websockets, numpy, Pillow, opencv-python 4.13
- tensorrt 10.3, cuda-python 12.9.6, torch 2.11
- SQLite WAL, 23 tables

**iOS:**
- SwiftUI + AVKit + AVFoundation (9 AVPlayers, PiP support)
- ActivityKit (Live Activities with snapshot rendering)
- WidgetKit (home + lock screen)
- URLSessionWebSocketTask (pose stream)

**Infra:**
- PM2 process manager
- Docker (Frigate, zigbee2mqtt)
- Vultr relay (Node WS server)
- Tailscale mesh
- Sonoff ZBDongle-E (Zigbee 3.0 coordinator)

## Codebase

~14,500 LOC Python backend across 15+ modules. Plus the iOS app (8,486 LOC Swift) and tunnel layer (not included in this demo repo).

## Production evidence

Live deployment running continuously:
- 74,670 events in `events` table
- 373,301 rows in `tracks`
- 425,702 rows in `behavior_stats`
- 631,867 rows in `anomaly_scores`
- 8 identified persons with 52 embeddings across angles
- 10 cameras with spatial metadata + adjacency graph

## What's in this repo

This is the `engine/` portion of the full system — the Python FastAPI backend that processes Frigate events and serves the iOS app. The iOS app, tunnel layer, and production database are not included.

## What's NOT in this repo

- Real credentials (API keys, passwords, phone numbers — replaced with placeholders)
- Production database contents
- Client-specific configuration (replaced with `acme-corp` examples)
- iOS app source
- Vultr tunnel relay (Node)
- Model weights (too large)

## License

PolyForm-Noncommercial-1.0.0. See LICENSE.

## Contact

Portfolio: [pyloxvision.com](https://pyloxvision.com)
Questions: open an issue or email [your@email].
