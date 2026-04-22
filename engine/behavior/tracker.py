"""Behavior Analyzer — trajectory tracking, loitering, speed, direction analysis.

Consumes position updates from MQTT detections and computes behavioral signals
in real-time. No GPU required — pure math on bounding box positions.
"""

import math
import time
import json
import threading
from collections import defaultdict
from engine.config import (
    LOITER_TIME_SECONDS, LOITER_RADIUS_PIXELS,
    SPEED_FAST_THRESHOLD, SPEED_SLOW_THRESHOLD,
    ZONE_DWELL_WARNING, ZONE_DWELL_ALERT,
)
from engine import database as db


class BehaviorAnalyzer:
    """Analyzes movement patterns from tracked object positions."""

    def __init__(self):
        self._lock = threading.Lock()
        # In-memory state for active tracks
        # track_id -> { positions: [{x,y,t}], zones: {zone: enter_time}, alerts_fired: set }
        self.tracks = defaultdict(lambda: {
            "positions": [],
            "zones": {},
            "alerts_fired": set(),
            "camera": None,
        })

    def update(self, track_id: str, camera: str, x: float, y: float,
               w: float, h: float, zones: list = None) -> list:
        """Process a new position for a track. Returns list of behavior events detected."""
        with self._lock:
            return self._update_locked(track_id, camera, x, y, w, h, zones)

    def _update_locked(self, track_id: str, camera: str, x: float, y: float,
                       w: float, h: float, zones: list = None) -> list:
        now = time.time()
        cx = x + w / 2  # center x
        cy = y + h / 2  # center y

        track = self.tracks[track_id]
        track["camera"] = camera
        track["positions"].append({"x": cx, "y": cy, "t": now, "w": w, "h": h})

        # Keep last 300 positions (enough for ~5 min at 1fps)
        if len(track["positions"]) > 300:
            track["positions"] = track["positions"][-300:]

        events = []

        # Run all analyzers
        events.extend(self._check_loitering(track_id, track))
        events.extend(self._check_speed(track_id, track))
        events.extend(self._check_direction_change(track_id, track))
        events.extend(self._check_zone_dwell(track_id, track, zones or []))

        return events

    def end_track(self, track_id: str) -> dict:
        """Finalize a track and compute summary statistics."""
        with self._lock:
            return self._end_track_locked(track_id)

    def _end_track_locked(self, track_id: str) -> dict:
        track = self.tracks.get(track_id)
        if not track or not track["positions"]:
            self.tracks.pop(track_id, None)
            return {}

        positions = track["positions"]
        duration = positions[-1]["t"] - positions[0]["t"]

        # Total distance traveled
        total_dist = 0
        for i in range(1, len(positions)):
            dx = positions[i]["x"] - positions[i-1]["x"]
            dy = positions[i]["y"] - positions[i-1]["y"]
            total_dist += math.sqrt(dx*dx + dy*dy)

        # Average speed
        avg_speed = total_dist / duration if duration > 0 else 0

        # Displacement (start to end straight line)
        displacement = math.sqrt(
            (positions[-1]["x"] - positions[0]["x"])**2 +
            (positions[-1]["y"] - positions[0]["y"])**2
        )

        # Linearity ratio (1.0 = straight line, 0.0 = went nowhere)
        linearity = displacement / total_dist if total_dist > 0 else 0

        summary = {
            "track_id": track_id,
            "camera": track["camera"],
            "duration": round(duration, 1),
            "total_distance": round(total_dist, 1),
            "displacement": round(displacement, 1),
            "avg_speed": round(avg_speed, 1),
            "linearity": round(linearity, 3),
            "position_count": len(positions),
        }

        # Persist summary
        db.insert_behavior(
            camera=track["camera"],
            track_id=track_id,
            behavior_type="track_summary",
            value=duration,
            data=summary,
        )

        self.tracks.pop(track_id, None)
        return summary

    def _check_loitering(self, track_id: str, track: dict) -> list:
        """Detect if a person stays in the same area too long."""
        positions = track["positions"]
        if len(positions) < 10:
            return []

        alert_key = "loiter"
        if alert_key in track["alerts_fired"]:
            return []

        # Check if the last N positions are within LOITER_RADIUS of their centroid
        recent = positions[-30:] if len(positions) >= 30 else positions
        duration = recent[-1]["t"] - recent[0]["t"]

        if duration < LOITER_TIME_SECONDS:
            return []

        # Compute centroid
        cx = sum(p["x"] for p in recent) / len(recent)
        cy = sum(p["y"] for p in recent) / len(recent)

        # Check all positions within radius
        all_within = all(
            math.sqrt((p["x"] - cx)**2 + (p["y"] - cy)**2) < LOITER_RADIUS_PIXELS
            for p in recent
        )

        if all_within:
            track["alerts_fired"].add(alert_key)
            event_data = {
                "behavior": "loitering",
                "duration": round(duration, 1),
                "center_x": round(cx, 1),
                "center_y": round(cy, 1),
                "radius": LOITER_RADIUS_PIXELS,
            }
            db.insert_behavior(
                camera=track["camera"], track_id=track_id,
                behavior_type="loitering", value=duration, data=event_data,
            )
            return [{"type": "loitering", "severity": "warning",
                      "track_id": track_id, "data": event_data}]

        return []

    def _check_speed(self, track_id: str, track: dict) -> list:
        """Detect running (fast movement) or suspicious stationary behavior."""
        positions = track["positions"]
        if len(positions) < 5:
            return []

        events = []

        # Calculate speed over last 5 positions
        recent = positions[-5:]
        dt = recent[-1]["t"] - recent[0]["t"]
        if dt < 0.5:
            return []

        dx = recent[-1]["x"] - recent[0]["x"]
        dy = recent[-1]["y"] - recent[0]["y"]
        dist = math.sqrt(dx*dx + dy*dy)
        speed = dist / dt  # pixels per second

        # Running detection
        if speed > SPEED_FAST_THRESHOLD and "running" not in track["alerts_fired"]:
            track["alerts_fired"].add("running")
            event_data = {"behavior": "running", "speed_px_sec": round(speed, 1)}
            db.insert_behavior(
                camera=track["camera"], track_id=track_id,
                behavior_type="running", value=speed, data=event_data,
            )
            events.append({"type": "running", "severity": "warning",
                          "track_id": track_id, "data": event_data})

        return events

    def _check_direction_change(self, track_id: str, track: dict) -> list:
        """Detect sudden direction changes (pacing, erratic movement)."""
        positions = track["positions"]
        if len(positions) < 15:
            return []

        alert_key = "erratic"
        if alert_key in track["alerts_fired"]:
            return []

        # Count direction reversals in last 15 positions
        recent = positions[-15:]
        reversals = 0
        for i in range(2, len(recent)):
            dx1 = recent[i-1]["x"] - recent[i-2]["x"]
            dy1 = recent[i-1]["y"] - recent[i-2]["y"]
            dx2 = recent[i]["x"] - recent[i-1]["x"]
            dy2 = recent[i]["y"] - recent[i-1]["y"]

            # Dot product — negative means direction reversal
            dot = dx1 * dx2 + dy1 * dy2
            if dot < -10:  # threshold to ignore tiny jitter
                reversals += 1

        if reversals >= 5:  # 5+ reversals in 15 frames = pacing/erratic
            track["alerts_fired"].add(alert_key)
            event_data = {
                "behavior": "erratic_movement",
                "reversals": reversals,
                "window": 15,
            }
            db.insert_behavior(
                camera=track["camera"], track_id=track_id,
                behavior_type="erratic", value=reversals, data=event_data,
            )
            return [{"type": "erratic_movement", "severity": "warning",
                      "track_id": track_id, "data": event_data}]

        return []

    def _check_zone_dwell(self, track_id: str, track: dict, zones: list) -> list:
        """Track how long a person dwells in each zone."""
        now = time.time()
        events = []

        # Update zone enter times
        for zone in zones:
            if zone not in track["zones"]:
                track["zones"][zone] = now

        # Remove zones we've left
        current_zones = set(zones)
        for zone in list(track["zones"].keys()):
            if zone not in current_zones:
                del track["zones"][zone]

        # Check dwell times
        for zone, enter_time in track["zones"].items():
            dwell = now - enter_time
            alert_key = f"zone_dwell_{zone}"

            if dwell > ZONE_DWELL_ALERT and f"{alert_key}_alert" not in track["alerts_fired"]:
                track["alerts_fired"].add(f"{alert_key}_alert")
                event_data = {
                    "behavior": "zone_dwell",
                    "zone": zone,
                    "dwell_seconds": round(dwell, 1),
                    "level": "alert",
                }
                db.insert_behavior(
                    camera=track["camera"], track_id=track_id,
                    behavior_type="zone_dwell", value=dwell, data=event_data,
                )
                events.append({"type": "zone_dwell", "severity": "alert",
                              "track_id": track_id, "data": event_data})

            elif dwell > ZONE_DWELL_WARNING and f"{alert_key}_warning" not in track["alerts_fired"]:
                track["alerts_fired"].add(f"{alert_key}_warning")
                event_data = {
                    "behavior": "zone_dwell",
                    "zone": zone,
                    "dwell_seconds": round(dwell, 1),
                    "level": "warning",
                }
                events.append({"type": "zone_dwell", "severity": "warning",
                              "track_id": track_id, "data": event_data})

        return events

    def get_active_count(self) -> dict:
        """Get count of active tracks per camera."""
        counts = defaultdict(int)
        for track in self.tracks.values():
            if track["camera"]:
                counts[track["camera"]] += 1
        return dict(counts)

    def get_track_state(self, track_id: str) -> dict:
        """Get current in-memory state of a track."""
        with self._lock:
            track = self.tracks.get(track_id)
            if not track:
                return {}
            # Copy data under lock to avoid race
            positions = list(track["positions"])
            camera = track["camera"]
            zones = list(track["zones"].keys())

        if not positions:
            return {"track_id": track_id, "camera": camera}

        # Current speed
        speed = 0
        heading = 0
        if len(positions) >= 2:
            dx = positions[-1]["x"] - positions[-2]["x"]
            dy = positions[-1]["y"] - positions[-2]["y"]
            dt = positions[-1]["t"] - positions[-2]["t"]
            if dt > 0:
                speed = math.sqrt(dx*dx + dy*dy) / dt
                heading = math.degrees(math.atan2(dy, dx))

        return {
            "track_id": track_id,
            "camera": camera,
            "current_pos": {"x": positions[-1]["x"], "y": positions[-1]["y"]},
            "speed": round(speed, 1),
            "heading": round(heading, 1),
            "duration": round(positions[-1]["t"] - positions[0]["t"], 1),
            "position_count": len(positions),
            "zones": zones,
        }
