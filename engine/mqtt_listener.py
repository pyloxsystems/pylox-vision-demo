"""MQTT Listener — consumes Frigate events and feeds them to the V2 engine.

Subscribes to:
  - frigate/events          (tracked object lifecycle: new, update, end)
  - frigate/+/person        (per-camera person count)
  - frigate/+/person/snapshot (person snapshots for anomaly detection)

Produces:
  - Track updates to BehaviorAnalyzer
  - Events to the event broadcast system
  - Frames to the anomaly detection queue (when enabled)
"""

import json
import time
import logging
import threading
from typing import Callable
import paho.mqtt.client as mqtt

from engine.config import MQTT_HOST, MQTT_PORT, CAMERAS
from engine.behavior.tracker import BehaviorAnalyzer
from engine.events.engine import EventEngine
from engine import database as db

logger = logging.getLogger("pylox-v2.mqtt")


class MQTTListener:
    def __init__(self, behavior: BehaviorAnalyzer, event_engine: EventEngine = None,
                 detection_handler=None, watch_manager=None, on_event: Callable = None,
                 loop=None):
        self.behavior = behavior
        self.event_engine = event_engine
        self.detection_handler = detection_handler
        self.watch_manager = watch_manager
        self.on_event = on_event
        self.loop = loop  # asyncio event loop reference (for cross-thread calls)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pylox-v2-engine")
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self._running = False
        self._camera_resolutions = {}  # camera -> (width, height)
        self._stats = {
            "messages_received": 0,
            "events_processed": 0,
            "tracks_active": 0,
            "last_message_time": 0,
        }

    def _load_camera_resolutions(self):
        """Fetch actual camera resolutions from Frigate API."""
        from engine.config import FRIGATE_API
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"{FRIGATE_API}/api/config", timeout=5)
            config = json.loads(resp.read().decode())
            for cam_name, cam_cfg in config.get("cameras", {}).items():
                detect = cam_cfg.get("detect", {})
                w = detect.get("width", 1920)
                h = detect.get("height", 1080)
                self._camera_resolutions[cam_name] = (w, h)
            logger.info(f"Loaded resolutions for {len(self._camera_resolutions)} cameras")
        except Exception as e:
            logger.warning(f"Failed to load camera resolutions, using 1920x1080 default: {e}")

    def start(self):
        """Connect to MQTT broker and start listening with retry."""
        self._load_camera_resolutions()
        logger.info(f"Connecting to MQTT broker at {MQTT_HOST}:{MQTT_PORT}")
        for attempt in range(5):
            try:
                self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
                self._running = True
                self.client.loop_start()
                return
            except Exception as e:
                logger.error(f"MQTT connect attempt {attempt+1}/5 failed: {e}")
                time.sleep(2 ** attempt)
        logger.error("MQTT connection failed after 5 attempts")

    def stop(self):
        """Disconnect from MQTT broker."""
        self._running = False
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("MQTT listener stopped")

    def get_stats(self) -> dict:
        return {**self._stats, "tracks_active": len(self.behavior.tracks)}

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT broker (rc={rc})")
        # Subscribe to Frigate event topics
        client.subscribe("frigate/events")
        client.subscribe("frigate/+/person")
        # Also subscribe to all object types for future use
        client.subscribe("frigate/+/car")
        client.subscribe("frigate/+/dog")
        client.subscribe("frigate/+/cat")
        # Audio events
        client.subscribe("frigate/+/audio/+")
        logger.info("Subscribed to Frigate topics (including audio)")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        if self._running:
            logger.warning(f"Disconnected from MQTT (rc={rc}), will reconnect...")

    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT messages from Frigate."""
        self._stats["messages_received"] += 1
        self._stats["last_message_time"] = time.time()

        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            if topic == "frigate/events":
                self._handle_event(payload)
            elif "/audio/" in topic:
                # Audio event: frigate/{camera}/audio/{event_type}
                parts = topic.split("/")
                if len(parts) >= 4:
                    camera = parts[1]
                    audio_type = parts[3]
                    if payload == "ON":
                        logger.info(f"Audio event: {audio_type} on {camera}")
                        if self.detection_handler and hasattr(self.detection_handler, 'signals'):
                            if self.detection_handler.signals:
                                self.detection_handler.signals.add_audio_event(camera, audio_type)
            elif "/person" in topic or "/car" in topic:
                pass

        except json.JSONDecodeError:
            # Some messages are just counts (plain integers)
            pass
        except Exception as e:
            import traceback
            logger.error(f"Error processing MQTT message on {msg.topic}: {e}\n{traceback.format_exc()}")

    def _handle_event(self, payload):
        """Handle a Frigate tracked object event.

        Frigate event payload structure:
        {
            "type": "new" | "update" | "end",
            "before": { ... previous state ... },
            "after": {
                "id": "1234-5678",
                "camera": "cam1",
                "label": "person",
                "current_zones": ["yard"],
                "box": [x1, y1, x2, y2],  # normalized 0-1
                "area": 12345,
                "region": [x1, y1, x2, y2],
                "score": 0.87,
                "has_snapshot": true,
                "has_clip": false,
                ...
            }
        }
        """
        if not isinstance(payload, dict):
            return

        event_type = payload.get("type")
        after = payload.get("after", {})

        if not isinstance(after, dict):
            return

        track_id = after.get("id")
        camera = after.get("camera")
        label = after.get("label", "unknown")

        if not track_id or not camera:
            return

        self._stats["events_processed"] += 1

        if event_type in ("new", "update"):
            # Extract bounding box (Frigate uses normalized coords 0-1)
            box = after.get("box", [0, 0, 0, 0])
            if len(box) == 4:
                x1, y1, x2, y2 = box
                # Convert normalized coords to pixels using actual camera resolution
                res_w, res_h = self._camera_resolutions.get(camera, (1920, 1080))
                px = x1 * res_w
                py = y1 * res_h
                pw = (x2 - x1) * res_w
                ph = (y2 - y1) * res_h
            else:
                return

            zones = after.get("current_zones", [])
            score = after.get("score", 0)

            # Skip low-confidence detections
            if score < 0.5:
                return

            # Update database track
            db.upsert_track(
                track_id=track_id,
                camera=camera,
                label=label,
                x=px, y=py, w=pw, h=ph,
                zones=zones,
            )

            # Run behavior analysis
            behavior_events = self.behavior.update(
                track_id=track_id,
                camera=camera,
                x=px, y=py, w=pw, h=ph,
                zones=zones,
            )

            # Process behavior events
            for evt in behavior_events:
                event_id = db.insert_event(
                    camera=camera,
                    event_type=evt["type"],
                    severity=evt["severity"],
                    track_id=track_id,
                    data=evt["data"],
                )
                evt["id"] = event_id
                logger.info(f"Behavior event: {evt['type']} on {camera} "
                           f"(track {track_id[:8]})")

                # Broadcast via WebSocket
                if self.on_event:
                    self.on_event({
                        "source": "behavior",
                        "event": evt,
                        "camera": camera,
                        "track_id": track_id,
                    })

            # Trigger watch session via WatchManager (Gemini-based AI guard)
            if self.watch_manager and self.loop and label == "person" and score > 0.6:
                import asyncio
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.watch_manager.trigger(
                            camera=camera,
                            trigger_source="frigate",
                            track_id=track_id,
                        ),
                        self.loop
                    )
                except Exception as e:
                    logger.error(f"Watch trigger failed: {e}")

            # V1 rules engine DISABLED — Gemini WatchManager is the sole decision maker
            # (event_engine kept for context but doesn't fire alerts)

            # Broadcast position update for 3D twin
            if self.on_event and label == "person":
                track_state = self.behavior.get_track_state(track_id)
                if track_state:
                    self.on_event({
                        "source": "position",
                        "camera": camera,
                        "track_id": track_id,
                        "label": label,
                        "position": track_state.get("current_pos"),
                        "speed": track_state.get("speed"),
                        "heading": track_state.get("heading"),
                        "zones": zones,
                        "score": score,
                    })

        elif event_type == "end":
            # Track ended — compute summary
            summary = self.behavior.end_track(track_id)
            db.deactivate_track(track_id)

            if summary and self.on_event:
                self.on_event({
                    "source": "track_end",
                    "camera": camera,
                    "track_id": track_id,
                    "summary": summary,
                })

            if summary:
                logger.info(
                    f"Track ended: {track_id[:8]} on {camera} — "
                    f"{summary.get('duration', 0):.0f}s, "
                    f"{summary.get('total_distance', 0):.0f}px traveled"
                )
