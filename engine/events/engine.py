"""Event Engine — combines signals from all V2 layers into classified events.

Aggregates:
  - Frigate detections (person, car, etc.)
  - Behavior analysis (loitering, running, erratic)
  - Anomaly scores (from Anomalib)
  - Zone violations
  - Time-based rules (after-hours presence)
  - Cross-camera tracking

Produces classified events with severity levels.
"""

import time
import logging
from collections import defaultdict
from engine import database as db
from engine.config import EVENT_COOLDOWN_SECONDS
from engine.events.rules import RulesEngine

logger = logging.getLogger("pylox-v2.events")


class EventEngine:
    def __init__(self):
        # Cooldown tracker: (camera, event_type) -> last_fire_time
        self._cooldowns = {}
        # Active person count per camera
        self._person_counts = defaultdict(int)
        # Subscribers for real-time event push
        self._subscribers = []
        # Rules engine
        self.rules = RulesEngine()

    def subscribe(self, callback):
        """Register a callback for real-time events."""
        self._subscribers.append(callback)

    def _broadcast(self, event: dict):
        """Push event to all subscribers."""
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception as e:
                logger.error(f"Event subscriber error: {e}")

    def process_detection(self, camera: str, track_id: str, label: str,
                          score: float, zones: list, speed: float = 0,
                          duration: float = 0):
        """Process a raw detection through the rules engine."""
        # Evaluate all rules
        fired = self.rules.evaluate_detection(
            camera=camera, track_id=track_id, label=label,
            score=score, zones=zones, speed=speed, duration=duration,
        )

        # Broadcast fired events
        for event in fired:
            event["timestamp"] = time.time()
            self._broadcast({"source": "rules", "event": event})

        return fired

    def process_behavior_event(self, camera: str, event: dict):
        """Process an event from the behavior analyzer."""
        event_type = event.get("type")
        severity = event.get("severity", "info")
        track_id = event.get("track_id")

        # Check cooldown
        cooldown_key = (camera, event_type)
        now = time.time()
        if cooldown_key in self._cooldowns:
            if now - self._cooldowns[cooldown_key] < EVENT_COOLDOWN_SECONDS:
                return None

        self._cooldowns[cooldown_key] = now

        # Classify and enrich
        classified = self._classify_event(camera, event_type, severity, event.get("data", {}))

        # Store
        event_id = db.insert_event(
            camera=camera,
            event_type=classified["type"],
            severity=classified["severity"],
            track_id=track_id,
            data=classified["data"],
        )

        classified["id"] = event_id
        classified["camera"] = camera
        classified["track_id"] = track_id
        classified["timestamp"] = now

        # Broadcast
        self._broadcast(classified)

        return classified

    def process_anomaly(self, camera: str, score: float, frame_path: str = None):
        """Process an anomaly score from Anomalib."""
        if score < 0.5:
            return None

        severity = "critical" if score > 0.8 else "warning" if score > 0.6 else "info"

        cooldown_key = (camera, "anomaly")
        now = time.time()
        if cooldown_key in self._cooldowns:
            if now - self._cooldowns[cooldown_key] < EVENT_COOLDOWN_SECONDS:
                return None
        self._cooldowns[cooldown_key] = now

        # Also evaluate anomaly rules
        self.rules.evaluate_anomaly(camera, score)

        event_data = {
            "anomaly_score": round(score, 3),
            "frame_path": frame_path,
        }

        event_id = db.insert_event(
            camera=camera,
            event_type="anomaly",
            severity=severity,
            data=event_data,
        )

        event = {
            "id": event_id,
            "type": "anomaly",
            "severity": severity,
            "camera": camera,
            "data": event_data,
            "timestamp": now,
        }

        self._broadcast(event)
        return event

    def update_person_count(self, camera: str, count: int):
        """Update active person count for a camera."""
        prev = self._person_counts[camera]
        self._person_counts[camera] = count
        self.rules.update_person_count(camera, count)

        if count >= 5 and prev < 5:
            self.process_behavior_event(camera, {
                "type": "crowding",
                "severity": "warning",
                "data": {"person_count": count},
            })

    def _classify_event(self, camera: str, event_type: str, severity: str,
                        data: dict) -> dict:
        """Enrich and possibly escalate event classification."""
        if event_type == "loitering" and data.get("duration", 0) > 120:
            severity = "critical"

        if event_type == "zone_dwell" and data.get("level") == "alert":
            severity = "critical"

        return {
            "type": event_type,
            "severity": severity,
            "data": data,
        }

    def get_summary(self) -> dict:
        """Get current event engine state summary."""
        return {
            "person_counts": dict(self._person_counts),
            "active_cooldowns": len(self._cooldowns),
            "rules_count": len(self.rules.rules),
            "rules_enabled": sum(1 for r in self.rules.rules.values() if r.enabled),
        }
