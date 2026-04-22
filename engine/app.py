"""Pylox V2 Engine — FastAPI application.

The central intelligence service that processes Frigate detections through
multiple AI layers: behavior analysis, anomaly detection, event classification,
pose estimation, and 3D spatial mapping.

Port: 3450
"""

import os
import json
import time
import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from engine.config import API_HOST, API_PORT, FRIGATE_API, CAMERAS, DATA_DIR
from engine.database import init_db, get_events, get_active_tracks, get_track, get_incidents, get_incident
from engine.behavior.tracker import BehaviorAnalyzer
from engine.events.engine import EventEngine
from engine.mqtt_listener import MQTTListener
from engine.gpu_queue import GPUQueue
from engine.anomaly.detector import AnomalyDetector
from engine.anomaly.frame_scorer import FrameScorer
from engine.pose.pipeline import PosePipeline
from engine.spatial.camera_mapping import CameraMapping
from engine.reports.generator import generate_incident_report, generate_daily_summary
from engine.gemini.connector import GeminiConnector
from engine.gemini.detection_handler import DetectionHandler
from engine.gemini.signal_collector import SignalCollector
from engine.gemini.watch_manager import WatchManager
from engine.deterrence.lorex_controller import LorexController
from engine.alerts.notifier import AlertNotifier
from engine.alerts.sms_sender import SMSSender
from engine.alerts.morning_summary import generate_overnight_summary
from engine.training.manager import TrainingManager
from engine.watchdog.health_monitor import HealthMonitor
from engine.sensors.zigbee_listener import ZigbeeListener
from engine.police import officer_api as police_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pylox-v2")

# --- Global state ---
behavior = BehaviorAnalyzer()
event_engine = EventEngine()
gpu_queue = GPUQueue(max_concurrent=2)
anomaly_detector = AnomalyDetector()
frame_scorer: FrameScorer = None
pose_pipeline: PosePipeline = None
camera_mapping = CameraMapping()
gemini = GeminiConnector()
signal_collector: SignalCollector = None
detection_handler: DetectionHandler = None
watch_manager: WatchManager = None
deterrent = LorexController()
training = TrainingManager(site_id="default")
sms_sender = SMSSender()
notifier = AlertNotifier(sms_sender=sms_sender if sms_sender.available else None)
health_monitor: HealthMonitor = None
zigbee_listener: ZigbeeListener = None
ws_clients: set[WebSocket] = set()
_ws_lock = threading.Lock()
mqtt_listener: MQTTListener = None


_loop: asyncio.AbstractEventLoop = None


def broadcast_event(event: dict):
    """Send event to all connected WebSocket clients (thread-safe)."""
    if not _loop:
        return
    message = json.dumps(event)
    disconnected = []
    with _ws_lock:
        clients = list(ws_clients)
    for ws in clients:
        try:
            asyncio.run_coroutine_threadsafe(ws.send_text(message), _loop)
        except Exception:
            disconnected.append(ws)
    if disconnected:
        with _ws_lock:
            for ws in disconnected:
                ws_clients.discard(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    global mqtt_listener, frame_scorer, pose_pipeline, detection_handler, health_monitor, zigbee_listener, watch_manager, _loop

    # Capture event loop for thread-safe WebSocket broadcasting
    _loop = asyncio.get_event_loop()

    # Init database
    init_db()
    logger.info("Database initialized")

    # Start GPU queue
    await gpu_queue.start(num_workers=2)
    logger.info("GPU queue started")

    # Load anomaly models (if any trained)
    anomaly_detector.load_models()

    # Start frame scorer (auto-scores cameras with trained models)
    frame_scorer = FrameScorer(anomaly_detector, event_engine, gpu_queue, interval=5.0)
    await frame_scorer.start()

    # Start pose pipeline (estimates body pose for active person tracks)
    pose_pipeline = PosePipeline(gpu_queue, on_pose_update=broadcast_event)
    await pose_pipeline.start()

    # Initialize default camera mappings (simplified — proper calibration done via API)
    default_positions = {
        "cam1": ([-4, 3, -4], [0, 0, 0]),
        "cam2": ([4, 3, -4], [0, 0, 0]),
        "cam3": ([0, 3, 4], [0, 0, 0]),
        "cam4": ([-4, 3, 4], [0, 0, 0]),
        "cam5": ([4, 3, 4], [0, 0, 0]),
        "cam6": ([-4, 4, 0], [0, 0, 0]),
        "cam7": ([4, 4, 0], [0, 0, 0]),
        "cam8": ([0, 5, 0], [0, 0, 0]),
        "cam9": ([0, 3, -5], [0, 0, 0]),
    }
    for cam_id, (pos, look) in default_positions.items():
        try:
            camera_mapping.add_simple_calibration(cam_id, pos, look)
        except Exception as e:
            logger.warning(f"Camera mapping init failed for {cam_id}: {e}")
    logger.info(f"Camera mappings initialized for {len(camera_mapping.calibrations)} cameras")

    # Wire training manager to Gemini
    gemini.set_training_manager(training)

    # Initialize signal collector with all subsystems
    global signal_collector
    signal_collector = SignalCollector(
        anomaly_detector=anomaly_detector,
        behavior_analyzer=behavior,
        zigbee_listener=None,  # Set after zigbee starts
        training_manager=training,
    )
    logger.info("Signal collector initialized")

    # Initialize Gemini detection handler with alert notifier
    def handle_alert(alert_data):
        notifier.send_alert(alert_data)
        broadcast_event({"source": "alert", "alert": alert_data})

    detection_handler = DetectionHandler(
        gemini=gemini,
        signal_collector=signal_collector,
        on_alert=handle_alert,
        on_event=broadcast_event,
    )

    # Initialize WatchManager — the AI security guard
    def on_incident_start(event):
        broadcast_event({"source": "incident_start", **event})

    def on_incident_narration(event):
        broadcast_event({"source": "incident_narration", **event})

    def on_incident_action(event):
        broadcast_event({"source": "incident_action", **event})
        if event.get("action") == "notification":
            notifier.send_alert({
                "camera": event.get("camera"),
                "camera_name": event.get("camera"),
                "threat": event.get("threat", 5),
                "type": "incident",
                "description": event.get("description", "Active incident"),
                "action": "alert_owner",
                "timestamp": event.get("timestamp", time.time()),
            })

    def on_incident_end(event):
        broadcast_event({"source": "incident_end", **event})

    watch_manager = WatchManager(
        gemini=gemini,
        signal_collector=signal_collector,
        deterrent=deterrent,
        on_start=on_incident_start,
        on_narration=on_incident_narration,
        on_action=on_incident_action,
        on_end=on_incident_end,
    )
    logger.info("WatchManager initialized — AI guard ready")
    logger.info(f"Gemini detection handler initialized (available: {gemini.available})")

    # Configure police API with shared Gemini + site + device info
    police_api.configure(
        gemini_connector=gemini,
        site_config=training.get_data().get("site", {}) if training else {},
        device_info={
            "serial": os.getenv("PYLOX_DEVICE_SERIAL", "PYLOX-DEV-001"),
            "version": "2.0.0",
        },
    )
    logger.info("Police API configured (/api/v2/police + /responder)")

    # Wire event engine to WebSocket broadcast
    event_engine.subscribe(broadcast_event)

    # Start MQTT listener
    mqtt_listener = MQTTListener(behavior=behavior, event_engine=event_engine,
                                    detection_handler=detection_handler,
                                    watch_manager=watch_manager,
                                    on_event=broadcast_event,
                                    loop=_loop)
    mqtt_listener.start()
    logger.info("MQTT listener started")

    # Start Zigbee sensor listener
    def handle_door_event_with_strobe(event):
        # Feed door state to detection handler for Gemini context
        if detection_handler:
            detection_handler.update_door_state(event["sensor"], event["state"])
        # Broadcast to WebSocket
        broadcast_event({"source": "door_sensor", **event})

        # INSTANT STROBE if door opens after hours (no Gemini gating)
        if event.get("state") == "open" and watch_manager and not watch_manager.is_business_hours():
            camera = event.get("camera")
            if camera and deterrent.can_strobe(camera):
                logger.warning(f"DOOR OPEN AFTER HOURS: {event.get('sensor')} → INSTANT STROBE on {camera}")
                if _loop:
                    # Trigger strobe immediately
                    asyncio.run_coroutine_threadsafe(
                        deterrent.strobe_on(camera, duration_sec=30),
                        _loop
                    )
                    # Then start watch session for narration/escalation
                    asyncio.run_coroutine_threadsafe(
                        watch_manager.trigger(
                            camera=camera,
                            trigger_source=f"door_sensor:{event.get('sensor')}",
                            force=True,
                        ),
                        _loop
                    )

    zigbee_listener = ZigbeeListener(on_door_event=handle_door_event_with_strobe)
    zigbee_listener.start()
    if signal_collector:
        signal_collector.zigbee = zigbee_listener
    logger.info("Zigbee sensor listener started")

    # Start health monitor (cellular watchdog)
    health_monitor = HealthMonitor(
        sms_sender=sms_sender if sms_sender.available else None,
        notifier=notifier,
        check_interval=30,
    )
    await health_monitor.start()

    logger.info(f"Pylox V2 Engine running on port {API_PORT}")

    yield

    # Shutdown
    if watch_manager:
        await watch_manager.stop_all()
    if zigbee_listener:
        zigbee_listener.stop()
    if health_monitor:
        await health_monitor.stop()
    if pose_pipeline:
        await pose_pipeline.stop()
    if frame_scorer:
        await frame_scorer.stop()
    if mqtt_listener:
        mqtt_listener.stop()
    await gpu_queue.stop()
    logger.info("Pylox V2 Engine stopped")


app = FastAPI(
    title="Pylox V2 Intelligence Engine",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Police / Law Enforcement routers ---
# /api/v2/police/*   — detective API (narrative, tactical brief, cases, holds, BOLO)
# /responder/{token} — mobile live view for responding officers
app.include_router(police_api.router)
app.include_router(police_api.responder_router)

# Serve spatial viewer
SPATIAL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "spatial")


@app.get("/spatial")
async def spatial_page():
    """Serve the 3D Spatial Twin viewer."""
    return FileResponse(os.path.join(SPATIAL_DIR, "index.html"))


# --- WebSocket endpoint for real-time events ---

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    with _ws_lock:
        ws_clients.add(ws)
    logger.info(f"WebSocket client connected ({len(ws_clients)} total)")
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        with _ws_lock:
            ws_clients.discard(ws)
        logger.info(f"WebSocket client disconnected ({len(ws_clients)} total)")


# --- REST API ---

@app.get("/api/v2/status")
async def get_status():
    """Engine status and stats."""
    return {
        "status": "running",
        "version": "2.0.0",
        "uptime": time.time(),
        "mqtt": mqtt_listener.get_stats() if mqtt_listener else {},
        "gpu": gpu_queue.get_stats(),
        "behavior": {
            "active_tracks": behavior.get_active_count(),
        },
        "anomaly": anomaly_detector.get_stats(),
        "frame_scorer": frame_scorer.get_stats() if frame_scorer else {},
        "pose": pose_pipeline.get_stats() if pose_pipeline else {},
        "events": event_engine.get_summary(),
        "ws_clients": len(ws_clients),
    }


@app.get("/api/v2/tracks")
async def api_get_tracks(camera: str = None):
    """Get all active tracks, optionally filtered by camera."""
    tracks = get_active_tracks(camera)
    # Parse JSON fields
    for t in tracks:
        t["positions"] = json.loads(t["positions"]) if isinstance(t["positions"], str) else t["positions"]
        t["zones"] = json.loads(t["zones"]) if isinstance(t["zones"], str) else t["zones"]
    return {"tracks": tracks, "count": len(tracks)}


@app.get("/api/v2/tracks/{track_id}")
async def api_get_track(track_id: str):
    """Get a specific track with full position history."""
    track = get_track(track_id)
    if not track:
        return JSONResponse(status_code=404, content={"error": "Track not found"})
    track["positions"] = json.loads(track["positions"]) if isinstance(track["positions"], str) else track["positions"]
    track["zones"] = json.loads(track["zones"]) if isinstance(track["zones"], str) else track["zones"]

    # Add live behavior state if active
    live_state = behavior.get_track_state(track_id)
    if live_state:
        track["live"] = live_state

    return track


@app.get("/api/v2/tracks/{track_id}/trajectory")
async def api_get_trajectory(track_id: str):
    """Get simplified trajectory for 3D twin rendering."""
    track = get_track(track_id)
    if not track:
        return JSONResponse(status_code=404, content={"error": "Track not found"})

    positions = json.loads(track["positions"]) if isinstance(track["positions"], str) else track["positions"]

    # Simplify to just x, y, t for rendering
    trajectory = [{"x": p["x"], "y": p["y"], "t": p["t"]} for p in positions]

    return {
        "track_id": track_id,
        "camera": track["camera"],
        "label": track["label"],
        "trajectory": trajectory,
    }


@app.get("/api/v2/events")
async def api_get_events(
    camera: str = None,
    event_type: str = None,
    since: float = None,
    limit: int = Query(default=50, le=500),
):
    """Get events with optional filters."""
    events = get_events(camera=camera, event_type=event_type, since=since, limit=limit)
    for e in events:
        e["data"] = json.loads(e["data"]) if isinstance(e["data"], str) else e["data"]
    return {"events": events, "count": len(events)}


@app.get("/api/v2/behavior/active")
async def api_behavior_active():
    """Get all active behavior states."""
    states = {}
    for track_id in list(behavior.tracks.keys()):
        state = behavior.get_track_state(track_id)
        if state:
            states[track_id] = state
    return {"tracks": states, "count": len(states)}


@app.get("/api/v2/cameras/activity")
async def api_camera_activity():
    """Get per-camera activity summary."""
    counts = behavior.get_active_count()
    camera_stats = {}
    for cam in ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8", "cam9"]:
        recent_events = get_events(camera=cam, limit=5)
        camera_stats[cam] = {
            "active_persons": counts.get(cam, 0),
            "recent_events": len(recent_events),
            "last_event": recent_events[0] if recent_events else None,
        }
    return camera_stats


@app.get("/api/v2/gpu/stats")
async def api_gpu_stats():
    """Get GPU queue statistics."""
    return gpu_queue.get_stats()


# --- Anomaly Detection API ---

@app.get("/api/v2/anomaly/status")
async def api_anomaly_status():
    """Get anomaly detection status."""
    return {
        "detector": anomaly_detector.get_stats(),
        "scorer": frame_scorer.get_stats() if frame_scorer else {},
    }


@app.post("/api/v2/anomaly/collect/{camera}")
async def api_anomaly_collect(camera: str, count: int = 300, interval: int = 10):
    """Start collecting training frames for a camera (background task)."""
    if camera not in CAMERAS:
        return JSONResponse(status_code=400, content={"error": f"Unknown camera: {camera}"})

    # Run in background
    import threading
    thread = threading.Thread(
        target=AnomalyDetector.collect_training_frames,
        args=(camera, FRIGATE_API, count, interval),
        daemon=True,
    )
    thread.start()

    return {
        "status": "collecting",
        "camera": camera,
        "count": count,
        "interval_sec": interval,
        "estimated_time_min": round(count * interval / 60, 1),
    }


@app.post("/api/v2/anomaly/train/{camera}")
async def api_anomaly_train(camera: str):
    """Train anomaly model for a camera (long-running, GPU-intensive)."""
    if camera not in CAMERAS:
        return JSONResponse(status_code=400, content={"error": f"Unknown camera: {camera}"})

    # Run training in background via GPU queue
    async def _train():
        return AnomalyDetector.train_model(camera)

    import threading
    thread = threading.Thread(
        target=AnomalyDetector.train_model,
        args=(camera,),
        daemon=True,
    )
    thread.start()

    return {
        "status": "training_started",
        "camera": camera,
        "note": "Training takes 5-10 minutes. Check /api/v2/anomaly/status for progress.",
    }


@app.post("/api/v2/anomaly/reload")
async def api_anomaly_reload():
    """Reload anomaly models from disk (after training)."""
    anomaly_detector.load_models()
    if frame_scorer:
        await frame_scorer.stop()
        await frame_scorer.start()
    return {
        "status": "reloaded",
        "models": anomaly_detector.get_stats(),
    }


# --- Pose API ---

@app.get("/api/v2/pose/all")
async def api_pose_all():
    """Get all active body meshes for 3D twin rendering."""
    if not pose_pipeline:
        return {"poses": {}, "count": 0}
    poses = pose_pipeline.get_all_poses()
    return {"poses": poses, "count": len(poses)}


@app.get("/api/v2/pose/{track_id}")
async def api_pose_track(track_id: str):
    """Get body mesh for a specific track."""
    if not pose_pipeline:
        return JSONResponse(status_code=503, content={"error": "Pose pipeline not running"})
    mesh = pose_pipeline.get_cached_pose(track_id)
    if not mesh:
        return JSONResponse(status_code=404, content={"error": "No pose data"})
    return mesh


@app.get("/api/v2/pose/stats")
async def api_pose_stats():
    """Get pose pipeline statistics."""
    if not pose_pipeline:
        return {}
    return pose_pipeline.get_stats()


# --- Spatial/Camera Mapping API ---

@app.get("/api/v2/spatial/cameras")
async def api_spatial_cameras():
    """Get all camera calibrations for 3D twin."""
    return camera_mapping.get_all_calibrations()


@app.post("/api/v2/spatial/calibrate/{camera}")
async def api_spatial_calibrate(camera: str, data: dict):
    """Calibrate a camera with reference points or simple position."""
    if "reference_points" in data:
        camera_mapping.add_calibration(
            camera, data["reference_points"],
            data.get("position", [0, 3, 0]),
            data.get("fov", 90),
        )
    elif "position" in data:
        camera_mapping.add_simple_calibration(
            camera, data["position"],
            data.get("look_at", [0, 0, 0]),
        )
    return {"status": "calibrated", "camera": camera}


@app.get("/api/v2/spatial/scene")
async def api_spatial_scene():
    """Get the full scene state for initial 3D twin load."""
    # Active tracks with world positions
    active = get_active_tracks()
    persons = []
    for track in active:
        positions = json.loads(track["positions"]) if isinstance(track["positions"], str) else track["positions"]
        if not positions:
            continue

        last = positions[-1]
        world_pos = camera_mapping.bbox_to_world(
            track["camera"], last.get("x", 0), last.get("y", 0),
            last.get("w", 50), last.get("h", 120),
        )

        # Get pose if available
        pose_mesh = None
        if pose_pipeline:
            pose_mesh = pose_pipeline.get_cached_pose(track["id"])

        persons.append({
            "track_id": track["id"],
            "camera": track["camera"],
            "label": track["label"],
            "world_position": world_pos,
            "pose_mesh": pose_mesh,
            "zones": json.loads(track["zones"]) if isinstance(track["zones"], str) else track["zones"],
        })

    return {
        "cameras": camera_mapping.get_all_calibrations(),
        "persons": persons,
        "person_count": len(persons),
    }


# --- Rules API ---

@app.get("/api/v2/rules")
async def api_get_rules():
    """Get all configured rules."""
    return {"rules": event_engine.rules.get_rules()}


@app.post("/api/v2/rules")
async def api_add_rule(rule_data: dict):
    """Add or update a rule."""
    from engine.events.rules import Rule
    try:
        rule = Rule(**rule_data)
        event_engine.rules.add_rule(rule)
        return {"status": "ok", "rule": rule.to_dict()}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.delete("/api/v2/rules/{rule_id}")
async def api_delete_rule(rule_id: str):
    """Delete a rule."""
    event_engine.rules.remove_rule(rule_id)
    return {"status": "deleted", "rule_id": rule_id}


@app.patch("/api/v2/rules/{rule_id}/toggle")
async def api_toggle_rule(rule_id: str):
    """Enable/disable a rule."""
    rule = event_engine.rules.rules.get(rule_id)
    if not rule:
        return JSONResponse(status_code=404, content={"error": "Rule not found"})
    rule.enabled = not rule.enabled
    return {"rule_id": rule_id, "enabled": rule.enabled}


# --- Signals API ---

@app.get("/api/v2/signals/stats")
async def api_signals_stats():
    """Get signal collector statistics."""
    return signal_collector.get_stats() if signal_collector else {}


@app.get("/api/v2/signals/test/{camera}")
async def api_signals_test(camera: str):
    """Test: collect all signals for a camera (without calling Gemini)."""
    if not signal_collector:
        return {"error": "not initialized"}
    collected = signal_collector.collect(
        camera=camera, track_id="test", label="person",
        score=0.9, zones=[], duration=10.0,
    )
    prompt_text = signal_collector.build_prompt_context(collected)
    return {"signals": collected, "prompt_context": prompt_text}


# --- Sensors API ---

@app.get("/api/v2/sensors")
async def api_sensors():
    """Get all Zigbee sensor states."""
    if not zigbee_listener:
        return {"sensors": {}, "stats": {}}
    return {
        "sensors": zigbee_listener.get_all_sensors(),
        "doors": zigbee_listener.get_door_states(),
        "stats": zigbee_listener.get_stats(),
    }


# --- Incidents API (AI Guard sessions) ---

@app.get("/api/v2/incidents")
async def api_get_incidents(camera: str = None, since: float = None,
                              limit: int = Query(default=50, le=200)):
    """Get list of incidents."""
    incidents = get_incidents(camera=camera, since=since, limit=limit)
    return {"incidents": incidents, "count": len(incidents)}


@app.get("/api/v2/incidents/{incident_id}")
async def api_get_incident(incident_id: int):
    """Get full incident with all narrations."""
    incident = get_incident(incident_id)
    if not incident:
        return JSONResponse(status_code=404, content={"error": "Incident not found"})
    return incident


@app.get("/api/v2/incidents/active/all")
async def api_active_incidents():
    """Get currently active watch sessions."""
    if not watch_manager:
        return {"sessions": {}}
    return {"sessions": watch_manager.get_active_sessions()}


@app.get("/api/v2/watch/stats")
async def api_watch_stats():
    """Watch manager statistics."""
    if not watch_manager:
        return {}
    return watch_manager.get_stats()


@app.post("/api/v2/incidents/{incident_id}/dismiss")
async def api_dismiss_incident(incident_id: int):
    """Manually dismiss an active incident."""
    if not watch_manager:
        return JSONResponse(status_code=503, content={"error": "Watch manager not running"})

    # Find and stop the session for this incident
    for camera, session in list(watch_manager._active_sessions.items()):
        if session.incident_id == incident_id:
            await session.stop("manual_dismiss")
            return {"status": "dismissed"}

    return JSONResponse(status_code=404, content={"error": "No active session for this incident"})


# --- Deterrence API ---

@app.post("/api/v2/deterrent/strobe/{camera}")
async def api_strobe_on(camera: str, duration: int = 15):
    """Manually trigger strobe on a camera (for testing)."""
    success = await deterrent.strobe_on(camera, duration_sec=duration)
    return {"status": "strobing" if success else "failed", "camera": camera, "duration": duration}


@app.post("/api/v2/deterrent/strobe-off/{camera}")
async def api_strobe_off(camera: str):
    """Manually turn off strobe."""
    success = await deterrent.strobe_off(camera)
    return {"status": "off" if success else "failed", "camera": camera}


@app.get("/api/v2/deterrent/stats")
async def api_deterrent_stats():
    """Deterrent controller stats."""
    return deterrent.get_stats()


# --- Watchdog API ---

@app.get("/api/v2/health")
async def api_health():
    """Get full system health status."""
    return health_monitor.get_stats() if health_monitor else {"error": "not running"}


# --- Alerts API ---

@app.post("/api/v2/alerts/register-push")
async def api_register_push(data: dict):
    """Register an Expo push token for notifications."""
    token = data.get("token")
    if not token:
        return JSONResponse(status_code=400, content={"error": "token required"})
    notifier.register_push_token(token)
    return {"status": "registered"}


@app.post("/api/v2/alerts/register-sms")
async def api_register_sms(data: dict):
    """Register a phone number for SMS alerts."""
    number = data.get("number")
    if not number:
        return JSONResponse(status_code=400, content={"error": "number required"})
    notifier.register_sms_number(number)
    return {"status": "registered", "number": number}


@app.delete("/api/v2/alerts/sms/{number}")
async def api_remove_sms(number: str):
    """Remove an SMS number."""
    notifier.remove_sms_number(number)
    return {"status": "removed"}


@app.get("/api/v2/alerts/status")
async def api_alerts_status():
    """Get alert system status."""
    return notifier.get_stats()


@app.get("/api/v2/alerts/morning-summary")
async def api_morning_summary():
    """Get the overnight summary."""
    return generate_overnight_summary()


@app.post("/api/v2/alerts/send-summary")
async def api_send_summary():
    """Manually trigger morning summary push notification."""
    summary = generate_overnight_summary()
    notifier.send_morning_summary(summary["text"])
    return {"status": "sent", "summary": summary["text"]}


# --- Gemini API ---

@app.get("/api/v2/gemini/status")
async def api_gemini_status():
    """Get Gemini connector and detection handler status."""
    return detection_handler.get_stats() if detection_handler else {"error": "not initialized"}


@app.post("/api/v2/gemini/false-positive")
async def api_gemini_false_positive(data: dict):
    """Mark a detection as false positive (client feedback)."""
    camera = data.get("camera")
    description = data.get("description", "Unmarked false positive")
    training.add_false_positive(camera, description)
    return {"status": "learned", "camera": camera, "description": description}


@app.post("/api/v2/gemini/learn")
async def api_gemini_learn(data: dict):
    """Add a learned pattern (e.g., 'FedEx delivers Mon/Wed/Fri at 2pm')."""
    pattern = data.get("pattern")
    if not pattern:
        return JSONResponse(status_code=400, content={"error": "pattern required"})
    training.add_pattern(pattern)
    return {"status": "learned", "pattern": pattern}


# --- Training / Feedback API ---

@app.post("/api/v2/feedback")
async def api_feedback(data: dict):
    """Process client feedback on a detection.

    Body: { "event_id": 123, "feedback": "false_alarm"|"correct"|"known_person",
            "camera": "cam4", "description": "optional override" }
    """
    event_id = data.get("event_id")
    feedback = data.get("feedback")
    camera = data.get("camera")

    if not feedback or feedback not in ("false_alarm", "correct", "known_person"):
        return JSONResponse(status_code=400, content={"error": "feedback must be false_alarm, correct, or known_person"})

    # Get Gemini's description from the event if not provided
    description = data.get("description")
    if not description and event_id:
        from engine.database import get_db
        conn = get_db()
        row = conn.execute("SELECT data, camera FROM events WHERE id = ?", (event_id,)).fetchone()
        conn.close()
        if row:
            event_data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
            description = event_data.get("description", "")
            if not camera:
                camera = row["camera"]

    training.process_feedback(event_id, feedback, camera, description)
    return {"status": "learned", "feedback": feedback, "camera": camera}


@app.get("/api/v2/training")
async def api_get_training():
    """Get the full training data for this site."""
    return training.get_data()


@app.get("/api/v2/training/stats")
async def api_training_stats():
    """Get training statistics."""
    return training.get_stats()


@app.post("/api/v2/training/site")
async def api_set_site(data: dict):
    """Set site information (name, type, hours, address)."""
    training.set_site_info(
        name=data.get("name"),
        site_type=data.get("type"),
        hours=data.get("hours"),
        address=data.get("address"),
    )
    return {"status": "updated"}


@app.post("/api/v2/training/known-person")
async def api_add_known_person(data: dict):
    """Add a known person for a camera."""
    camera = data.get("camera")
    description = data.get("description")
    if not camera or not description:
        return JSONResponse(status_code=400, content={"error": "camera and description required"})
    training.add_known_person(camera, description)
    return {"status": "added"}


# --- Reports API ---

@app.post("/api/v2/reports/incident")
async def api_generate_incident_report(data: dict = {}):
    """Generate an incident report."""
    try:
        path = generate_incident_report(
            event_id=data.get("event_id"),
            camera=data.get("camera"),
            since=data.get("since"),
            until=data.get("until"),
            title=data.get("title"),
        )
        return {"status": "generated", "path": path}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/api/v2/reports/daily")
async def api_generate_daily_report(data: dict = {}):
    """Generate a daily summary report."""
    path = generate_daily_summary(date=data.get("date"))
    return {"status": "generated", "path": path}


@app.get("/api/v2/reports/{report_name}")
async def api_get_report(report_name: str):
    """Serve a generated report."""
    report_path = os.path.join(DATA_DIR, "reports", f"{report_name}.html")
    if not os.path.exists(report_path):
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    return FileResponse(report_path, media_type="text/html")


@app.get("/api/v2/reports")
async def api_list_reports():
    """List all generated reports."""
    reports_dir = os.path.join(DATA_DIR, "reports")
    if not os.path.exists(reports_dir):
        return {"reports": []}
    files = sorted(os.listdir(reports_dir), reverse=True)
    return {"reports": [f.replace(".html", "") for f in files if f.endswith(".html")]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
