"""Detection Handler — decides when to call Gemini and what to do with the result.

The core decision logic:
  1. Frigate detects person → MQTT event arrives
  2. Is it business hours? → log silently, feed 3D twin
  3. After hours? → grab frames → send to Gemini
  4. Gemini says false positive → suppress, learn
  5. Gemini says real, threat 7+ → WhatsApp alert
  6. Gemini says real, threat 4-6 → flag for morning review
  7. Gemini unavailable → fall back to V1 rules engine
"""

import io
import time
import logging
import urllib.request
from typing import Optional
from datetime import datetime
from PIL import Image

from engine.gemini.connector import GeminiConnector, GeminiAssessment
from engine.gemini.signal_collector import SignalCollector
from engine.config import FRIGATE_API
from engine import database as db

logger = logging.getLogger("pylox-v2.gemini.handler")

# Camera descriptions for Gemini context
CAMERA_DESCRIPTIONS = {
    "cam1": "Front Entrance — faces the main entry door and front windows",
    "cam2": "Parking Lot — covers the parking area and company vans",
    "cam3": "Back Area — rear exterior of the building",
    "cam4": "Back Door — the rear entry/exit door",
    "cam5": "Office Lobby — interior office with couch, hallway to shop",
    "cam6": "Warehouse — main warehouse floor, materials and workbenches",
    "cam7": "Office Entrance — glass front doors seen from inside",
    "cam8": "Shop Floor — high angle view of full warehouse floor",
    "cam9": "Driveway — outdoor view between company vans",
}


class DetectionHandler:
    """Handles detection events and routes them through Gemini or V1 fallback."""

    def __init__(self, gemini: GeminiConnector, signal_collector: SignalCollector = None,
                 site_config: dict = None, on_alert=None, on_event=None):
        self.gemini = gemini
        self.signals = signal_collector
        self.site_config = site_config or {
            "id": "default",
            "name": "ACME-CORP Warehouse",
            "type": "warehouse / office",
            "hours": {"open": 6, "close": 20},
        }
        self.on_alert = on_alert  # callback for WhatsApp alerts
        self.on_event = on_event  # callback for WebSocket broadcast
        self._cooldowns = {}  # (camera, track_id) → last analysis time
        self._cooldown_sec = 30  # don't re-analyze same track within 30s
        self._door_states = {}  # door_name → "open"/"closed"
        self._stats = {
            "detections_received": 0,
            "business_hours_skipped": 0,
            "gemini_calls": 0,
            "v1_fallbacks": 0,
            "alerts_sent": 0,
            "false_positives_caught": 0,
        }

    def is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        hour = datetime.now().hour
        open_h = self.site_config.get("hours", {}).get("open", 6)
        close_h = self.site_config.get("hours", {}).get("close", 20)
        return open_h <= hour < close_h

    def update_door_state(self, door_name: str, state: str):
        """Update door sensor state (from Zigbee)."""
        self._door_states[door_name] = state
        logger.info(f"Door sensor: {door_name} → {state}")

        # After hours door open → immediate Gemini check on nearest camera
        if state == "open" and not self.is_business_hours():
            logger.warning(f"AFTER HOURS door open: {door_name}")
            # Find the camera for this door
            door_camera = self._door_to_camera(door_name)
            if door_camera:
                self._analyze_camera_now(door_camera, trigger=f"door_sensor:{door_name}")

    def handle_detection(self, camera: str, track_id: str, label: str,
                          score: float, zones: list, duration: float,
                          anomaly_score: float = None) -> Optional[dict]:
        """Handle a detection event from MQTT.

        Returns event dict if an alert/event was generated, None otherwise.
        """
        self._stats["detections_received"] += 1

        # Cooldown check — don't re-analyze same person on same camera
        cooldown_key = (camera, track_id)
        now = time.time()
        if cooldown_key in self._cooldowns:
            if now - self._cooldowns[cooldown_key] < self._cooldown_sec:
                return None

        # Business hours → log only, no Gemini call
        if self.is_business_hours():
            self._stats["business_hours_skipped"] += 1
            return None

        # After hours → Gemini analysis with full signal context
        self._cooldowns[cooldown_key] = now

        # Grab frames from Frigate
        frames = self._grab_frames(camera, track_id)
        if not frames:
            logger.warning(f"No frames captured for {camera}/{track_id}")
            return None

        self._stats["gemini_calls"] += 1

        # Get first frame as bytes for Re-ID
        frame_bytes = None
        if frames:
            buf = io.BytesIO()
            frames[0].save(buf, format="JPEG", quality=80)
            frame_bytes = buf.getvalue()

        # Collect ALL signals
        signal_context = ""
        if self.signals:
            collected = self.signals.collect(
                camera=camera, track_id=track_id, label=label,
                score=score, zones=zones, duration=duration,
                frame_bytes=frame_bytes,
            )
            signal_context = self.signals.build_prompt_context(collected)

        camera_name = CAMERA_DESCRIPTIONS.get(camera, camera)

        # Call Gemini with full context
        assessment = self.gemini.analyze_detection(
            frames=frames,
            camera_id=camera,
            camera_name=camera_name,
            label=label,
            score=score,
            zones=zones,
            duration=duration,
            anomaly_score=anomaly_score,
            door_state=self._door_states if self._door_states else None,
            site_config=self.site_config,
            signal_context=signal_context,
        )

        # If Gemini unavailable, fall back to V1
        if assessment is None:
            self._stats["v1_fallbacks"] += 1
            return self._v1_fallback(camera, label, score, zones, duration)

        # Process the assessment
        return self._process_assessment(assessment, camera, track_id, label)

    def _process_assessment(self, assessment: GeminiAssessment,
                             camera: str, track_id: str, label: str) -> dict:
        """Process Gemini's assessment and take action."""
        event = {
            "source": "gemini",
            "camera": camera,
            "track_id": track_id,
            "label": label,
            "assessment": assessment.to_dict(),
            "timestamp": time.time(),
        }

        # Store in database
        db.insert_event(
            camera=camera,
            event_type=f"gemini_{assessment.type}",
            severity=assessment.severity,
            track_id=track_id,
            data=assessment.to_dict(),
        )

        if not assessment.real:
            # False positive — learn from it
            self._stats["false_positives_caught"] += 1
            self.gemini.add_false_positive(
                self.site_config.get("id", "default"),
                camera,
                assessment.description,
            )
            logger.info(f"False positive caught on {camera}: {assessment.description}")

        elif assessment.should_alert:
            # THREAT 7-10 → immediate alert
            self._stats["alerts_sent"] += 1
            event["alert"] = True
            if self.on_alert:
                self.on_alert({
                    "camera": camera,
                    "camera_name": CAMERA_DESCRIPTIONS.get(camera, camera),
                    "track_id": track_id,
                    "threat": assessment.threat,
                    "type": assessment.type,
                    "description": assessment.description,
                    "action": assessment.action,
                    "timestamp": time.time(),
                })

        elif assessment.should_flag:
            # THREAT 4-6 → flag for morning review
            event["flagged"] = True
            logger.info(f"Flagged for review on {camera}: {assessment.description}")

        # Broadcast via WebSocket
        if self.on_event:
            self.on_event(event)

        return event

    def _grab_frames(self, camera: str, track_id: str) -> list:
        """Grab frames from Frigate for analysis."""
        frames = []

        # Try event snapshot first
        try:
            url = f"{FRIGATE_API}/api/events/{track_id}/snapshot.jpg"
            req = urllib.request.urlopen(url, timeout=3)
            data = req.read()
            if len(data) > 1000:
                frames.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            pass

        # Get latest camera snapshot (different angle/moment)
        try:
            url = f"{FRIGATE_API}/api/{camera}/latest.jpg?h=720"
            req = urllib.request.urlopen(url, timeout=3)
            data = req.read()
            if len(data) > 1000:
                frames.append(Image.open(io.BytesIO(data)).convert("RGB"))
        except Exception:
            pass

        # Try to get a few more frames with slight delays
        for _ in range(2):
            try:
                time.sleep(0.5)
                url = f"{FRIGATE_API}/api/{camera}/latest.jpg?h=720"
                req = urllib.request.urlopen(url, timeout=3)
                data = req.read()
                if len(data) > 1000:
                    frames.append(Image.open(io.BytesIO(data)).convert("RGB"))
            except Exception:
                break

        return frames

    def _analyze_camera_now(self, camera: str, trigger: str = ""):
        """Force-analyze a camera immediately (e.g., door sensor triggered)."""
        frames = self._grab_frames(camera, f"door-trigger-{int(time.time())}")
        if not frames:
            return

        camera_name = CAMERA_DESCRIPTIONS.get(camera, camera)
        assessment = self.gemini.analyze_detection(
            frames=frames,
            camera_id=camera,
            camera_name=camera_name + f" [TRIGGERED BY: {trigger}]",
            label="person",
            score=0.0,
            zones=[],
            duration=0,
            door_state=self._door_states,
            site_config=self.site_config,
        )

        if assessment and assessment.real:
            self._process_assessment(assessment, camera, f"door-{int(time.time())}", "person")

    def _door_to_camera(self, door_name: str) -> Optional[str]:
        """Map a door sensor name to the nearest camera."""
        mapping = {
            "front_door": "cam1",
            "front_entrance": "cam1",
            "back_door": "cam4",
            "backdoor": "cam4",
            "warehouse_door": "cam6",
            "shop_door": "cam5",
            "office_door": "cam7",
        }
        return mapping.get(door_name)

    def _v1_fallback(self, camera: str, label: str, score: float,
                      zones: list, duration: float) -> Optional[dict]:
        """V1 rules engine fallback when Gemini is unavailable."""
        logger.warning(f"V1 fallback active for {camera}")

        # Simple rules — after hours person = alert
        if label == "person" and score > 0.6:
            event = {
                "source": "v1_fallback",
                "camera": camera,
                "severity": "warning",
                "description": f"Person detected on {camera} (V1 fallback — Gemini unavailable)",
                "timestamp": time.time(),
            }
            db.insert_event(
                camera=camera,
                event_type="v1_after_hours",
                severity="warning",
                data={"label": label, "score": score, "zones": zones},
            )
            if self.on_alert:
                self.on_alert({
                    "camera": camera,
                    "camera_name": CAMERA_DESCRIPTIONS.get(camera, camera),
                    "threat": 6,
                    "type": "unknown",
                    "description": f"Person detected (AI verification unavailable)",
                    "action": "alert_owner",
                    "timestamp": time.time(),
                })
            return event

        return None

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "gemini": self.gemini.get_stats(),
            "business_hours": self.is_business_hours(),
            "door_states": dict(self._door_states),
        }
