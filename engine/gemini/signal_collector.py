"""Signal Collector — gathers all available context signals for Gemini.

Collects from every subsystem and builds the richest possible context
for Gemini's analysis. More signals = higher accuracy.

Signals:
  1. Anomaly score (Anomalib) — how different the scene looks
  2. Re-ID match (OSNet) — known person or stranger
  3. Behavior data (tracker) — speed, duration, loitering, direction changes
  4. Door sensor states (Zigbee) — which doors open/closed
  5. Multi-camera correlation (Re-ID) — same person across cameras
  6. Audio events (Frigate) — glass break, shout, alarm
  7. Weather (OpenWeatherMap) — rain, fog, night conditions
  8. Detection confidence (Frigate YOLO) — how sure the detection is
  9. Camera history (training file) — known false positives
  10. Time context — day of week, business hours, holidays
"""

import io
import time
import json
import logging
import base64
import urllib.request
from datetime import datetime
from typing import Optional

logger = logging.getLogger("pylox-v2.signals")

REID_URL = "http://localhost:3101"
WEATHER_CACHE_TTL = 3600  # 1 hour


class SignalCollector:
    """Collects all context signals for a detection."""

    def __init__(self, anomaly_detector=None, behavior_analyzer=None,
                 zigbee_listener=None, training_manager=None):
        self.anomaly = anomaly_detector
        self.behavior = behavior_analyzer
        self.zigbee = zigbee_listener
        self.training = training_manager

        # Re-ID embedding history: track_id -> {embedding, camera, timestamp}
        self._reid_history = {}
        self._reid_known = {}  # label -> [embeddings] (known people)

        # Weather cache
        self._weather = None
        self._weather_time = 0

        # Audio events: camera -> [{event, timestamp}]
        self._audio_events = {}

        self._stats = {
            "signals_collected": 0,
            "reid_matches": 0,
            "reid_unknowns": 0,
            "audio_events": 0,
        }

    def collect(self, camera: str, track_id: str, label: str,
                score: float, zones: list, duration: float,
                frame_bytes: bytes = None) -> dict:
        """Collect all signals for a detection.

        Returns a dict of all available context for the Gemini prompt.
        """
        self._stats["signals_collected"] += 1
        signals = {}

        # 1. Anomaly score
        signals["anomaly"] = self._get_anomaly_score(camera)

        # 2. Re-ID
        signals["reid"] = self._get_reid_match(track_id, camera, frame_bytes)

        # 3. Behavior
        signals["behavior"] = self._get_behavior(track_id)

        # 4. Door sensors
        signals["doors"] = self._get_door_states()

        # 5. Multi-camera correlation
        signals["multi_camera"] = self._get_multi_camera(track_id, camera)

        # 6. Audio
        signals["audio"] = self._get_recent_audio(camera)

        # 7. Weather — removed (unnecessary API call)

        # 8. Detection confidence
        signals["detection_confidence"] = round(score * 100)

        # 9. Camera history (from training file)
        signals["camera_history"] = self._get_camera_history(camera)

        # 10. Time context
        signals["time_context"] = self._get_time_context()

        return signals

    def build_prompt_context(self, signals: dict) -> str:
        """Convert collected signals into text for the Gemini prompt."""
        lines = []

        # Detection confidence
        lines.append(f"Detection confidence: {signals.get('detection_confidence', '?')}%")

        # Anomaly
        anomaly = signals.get("anomaly")
        if anomaly is not None:
            if anomaly < 0.2:
                lines.append(f"Anomaly score: {anomaly:.2f}/1.0 — scene looks normal")
            elif anomaly < 0.5:
                lines.append(f"Anomaly score: {anomaly:.2f}/1.0 — slight deviation from normal")
            else:
                lines.append(f"Anomaly score: {anomaly:.2f}/1.0 — scene looks significantly different from normal")

        # Re-ID
        reid = signals.get("reid", {})
        if reid.get("status") == "known":
            lines.append(f"Re-ID: KNOWN PERSON — matched '{reid['label']}' (seen {reid.get('times_seen', '?')} times)")
        elif reid.get("status") == "seen_before":
            lines.append(f"Re-ID: Seen before {reid.get('times_seen', '?')} times (not identified by name)")
        elif reid.get("status") == "unknown":
            lines.append(f"Re-ID: NEVER SEEN BEFORE — no match in history")

        # Behavior
        beh = signals.get("behavior", {})
        if beh:
            parts = []
            if beh.get("duration"):
                parts.append(f"visible for {beh['duration']:.0f}s")
            if beh.get("speed"):
                parts.append(f"speed {beh['speed']:.1f}px/s")
            if beh.get("direction_changes"):
                parts.append(f"{beh['direction_changes']} direction changes")
            if beh.get("loitering"):
                parts.append("LOITERING detected")
            if parts:
                lines.append(f"Behavior: {', '.join(parts)}")

        # Door sensors
        doors = signals.get("doors", {})
        if doors:
            open_doors = [k for k, v in doors.items() if v == "open"]
            closed_doors = [k for k, v in doors.items() if v == "closed"]
            if open_doors:
                lines.append(f"Door sensors OPEN: {', '.join(open_doors)}")
            if closed_doors:
                lines.append(f"Door sensors closed: {', '.join(closed_doors)}")

        # Multi-camera
        multi = signals.get("multi_camera")
        if multi:
            lines.append(f"Multi-camera: Same person seen on {multi['from_camera']} → {multi['to_camera']} in {multi['elapsed']:.0f} seconds")

        # Audio
        audio = signals.get("audio", [])
        if audio:
            audio_str = ", ".join([a["event"] for a in audio])
            lines.append(f"Audio detected: {audio_str}")


        # Time
        time_ctx = signals.get("time_context", {})
        if time_ctx:
            lines.append(f"Time: {time_ctx.get('time', '')} — {time_ctx.get('day', '')} — {time_ctx.get('status', '')}")

        # Camera history
        history = signals.get("camera_history", "")
        if history:
            lines.append(history)

        return "\n".join(lines)

    # ── Signal getters ──

    def _get_anomaly_score(self, camera: str) -> Optional[float]:
        if not self.anomaly:
            return None
        try:
            from engine.database import get_db
            conn = get_db()
            row = conn.execute(
                "SELECT score FROM anomaly_scores WHERE camera = ? ORDER BY timestamp DESC LIMIT 1",
                (camera,)
            ).fetchone()
            conn.close()
            if row:
                return round(row["score"], 3)
        except Exception:
            pass
        return None

    def _get_reid_match(self, track_id: str, camera: str,
                         frame_bytes: bytes = None) -> dict:
        if not frame_bytes:
            return {}

        try:
            img_b64 = base64.b64encode(frame_bytes).decode()
            req = urllib.request.Request(
                f"{REID_URL}/embed",
                data=json.dumps({"image": img_b64}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urllib.request.urlopen(req, timeout=5)
            data = json.loads(resp.read().decode())
            embedding = data.get("embedding")

            if not embedding:
                return {}

            # Compare against history
            best_match = None
            best_sim = 0

            for hist_id, hist in self._reid_history.items():
                sim = self._cosine_similarity(embedding, hist["embedding"])
                if sim > best_sim:
                    best_sim = sim
                    best_match = hist

            # Store this embedding
            self._reid_history[track_id] = {
                "embedding": embedding,
                "camera": camera,
                "timestamp": time.time(),
            }

            # Cleanup old entries (keep last 100)
            if len(self._reid_history) > 100:
                oldest = sorted(self._reid_history.items(),
                              key=lambda x: x[1]["timestamp"])[:50]
                for k, _ in oldest:
                    del self._reid_history[k]

            # Check against known people
            for label, embeddings in self._reid_known.items():
                for known_emb in embeddings:
                    sim = self._cosine_similarity(embedding, known_emb)
                    if sim > 0.7:
                        self._stats["reid_matches"] += 1
                        return {
                            "status": "known",
                            "label": label,
                            "similarity": round(sim, 3),
                            "times_seen": len(embeddings),
                        }

            if best_match and best_sim > 0.7:
                self._stats["reid_matches"] += 1
                times = sum(1 for h in self._reid_history.values()
                           if self._cosine_similarity(embedding, h["embedding"]) > 0.7)
                return {
                    "status": "seen_before",
                    "similarity": round(best_sim, 3),
                    "times_seen": times,
                    "last_camera": best_match["camera"],
                }

            self._stats["reid_unknowns"] += 1
            return {"status": "unknown"}

        except Exception as e:
            logger.debug(f"Re-ID failed: {e}")
            return {}

    def _get_behavior(self, track_id: str) -> dict:
        if not self.behavior:
            return {}

        state = self.behavior.get_track_state(track_id)
        if not state:
            return {}

        return {
            "duration": state.get("duration", 0),
            "speed": state.get("speed", 0),
            "heading": state.get("heading", 0),
            "position_count": state.get("position_count", 0),
            "zones": state.get("zones", []),
            "loitering": state.get("duration", 0) > 30 and state.get("speed", 0) < 5,
            "direction_changes": 0,  # Would need to compute from positions
        }

    def _get_door_states(self) -> dict:
        if not self.zigbee:
            return {}
        return self.zigbee.get_door_states()

    def _get_multi_camera(self, track_id: str, current_camera: str) -> Optional[dict]:
        """Check if this person was seen on another camera recently."""
        if track_id not in self._reid_history:
            return None

        current = self._reid_history[track_id]
        now = time.time()

        for hist_id, hist in self._reid_history.items():
            if hist_id == track_id:
                continue
            if hist["camera"] == current_camera:
                continue
            if now - hist["timestamp"] > 120:  # within 2 minutes
                continue

            sim = self._cosine_similarity(
                current.get("embedding", []),
                hist.get("embedding", [])
            )
            if sim > 0.7:
                return {
                    "from_camera": hist["camera"],
                    "to_camera": current_camera,
                    "elapsed": now - hist["timestamp"],
                    "similarity": round(sim, 3),
                }

        return None

    def _get_recent_audio(self, camera: str) -> list:
        """Get recent audio events for a camera."""
        events = self._audio_events.get(camera, [])
        now = time.time()
        # Only events in last 30 seconds
        recent = [e for e in events if now - e["timestamp"] < 30]
        self._audio_events[camera] = recent
        return recent

    def add_audio_event(self, camera: str, event: str):
        """Add an audio event (called from MQTT listener)."""
        self._stats["audio_events"] += 1
        events = self._audio_events.setdefault(camera, [])
        events.append({"event": event, "timestamp": time.time()})
        # Keep last 20
        if len(events) > 20:
            self._audio_events[camera] = events[-20:]

    def _get_weather(self) -> Optional[str]:
        now = time.time()
        if self._weather and now - self._weather_time < WEATHER_CACHE_TTL:
            return self._weather

        try:
            # Free API — no key needed for basic data
            url = "https://wttr.in/Fort+Lauderdale?format=%C+%t+%w"
            req = urllib.request.urlopen(url, timeout=3)
            self._weather = req.read().decode().strip()
            self._weather_time = now
            return self._weather
        except Exception:
            return self._weather  # Return cached even if expired

    def _get_camera_history(self, camera: str) -> str:
        if not self.training:
            return ""
        return self.training.get_prompt_context(camera)

    def _get_time_context(self) -> dict:
        now = datetime.now()
        return {
            "time": now.strftime("%I:%M %p"),
            "day": now.strftime("%A, %B %d"),
            "status": "AFTER HOURS" if now.hour < 6 or now.hour >= 20 else "BUSINESS HOURS",
            "weekend": now.weekday() >= 5,
        }

    def register_known_person(self, label: str, embedding: list):
        """Register a known person's embedding for Re-ID matching."""
        embs = self._reid_known.setdefault(label, [])
        embs.append(embedding)
        # Keep max 5 embeddings per person
        if len(embs) > 5:
            embs.pop(0)
        logger.info(f"Registered known person: {label} ({len(embs)} embeddings)")

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "reid_history_size": len(self._reid_history),
            "known_people": list(self._reid_known.keys()),
            "weather_cached": self._weather is not None,
        }
