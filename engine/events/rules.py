"""Rules Engine — configurable detection rules that combine multiple signals.

Rules are defined per-camera or globally. Each rule specifies:
  - Conditions (what triggers it)
  - Severity level
  - Cooldown period
  - Actions (what happens when triggered)

Built-in rule types:
  - time_restricted: Alert for presence during off-hours
  - zone_intrusion: Alert when entering a restricted zone
  - person_count: Alert when person count exceeds threshold
  - speed_threshold: Alert for running/fast movement
  - loiter_duration: Alert for extended loitering
  - anomaly_score: Alert when anomaly score exceeds threshold
  - multi_camera: Alert when same person appears on multiple cameras rapidly
"""

import time
import json
import logging
from typing import Optional
from dataclasses import dataclass, field
from engine import database as db

logger = logging.getLogger("pylox-v2.events.rules")


@dataclass
class Rule:
    id: str
    name: str
    rule_type: str
    cameras: list  # ["cam1", "cam2"] or ["*"] for all
    conditions: dict
    severity: str = "warning"
    cooldown_sec: int = 60
    enabled: bool = True
    actions: list = field(default_factory=list)  # ["webhook", "log", "alert"]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "rule_type": self.rule_type,
            "cameras": self.cameras,
            "conditions": self.conditions,
            "severity": self.severity,
            "cooldown_sec": self.cooldown_sec,
            "enabled": self.enabled,
            "actions": self.actions,
        }


# ── Zone-aware, camera-specific rules for the warehouse ──
#
# Camera layout:
#   cam1 — Front entrance (zones: front_entrance, door, front_windows)
#   cam2 — Parking lot / vans (zones: van_2, front, side, side_2)
#   cam3 — Back area (zones: back)
#   cam4 — Back door (zones: backdoor, back_door)
#   cam5 — Office/Lobby (couch, shop door, hallway) — "Office-Shop Door"
#   cam6 — Warehouse floor (materials, workbenches, flag) — "Warehouse Cam"
#   cam7 — Office entrance (glass front doors from inside) — "Office Entrance"
#   cam8 — Warehouse high angle (full shop floor, equipment) — Lorex
#   cam9 — Driveway between vans, outdoor 180° — wide angle
#
DEFAULT_RULES = [
    # ── AFTER-HOURS (perimeter cameras are critical, interior is warning) ──
    Rule(
        id="after_hours_perimeter",
        name="After-Hours — Perimeter",
        rule_type="time_restricted",
        cameras=["cam1", "cam2", "cam3", "cam4", "cam9"],
        conditions={
            "restricted_hours": {"start": 22, "end": 6},
            "min_confidence": 0.7,
        },
        severity="critical",
        cooldown_sec=300,
        actions=["log", "alert"],
    ),
    Rule(
        id="after_hours_interior",
        name="After-Hours — Interior",
        rule_type="time_restricted",
        cameras=["cam5", "cam6", "cam7", "cam8"],
        conditions={
            "restricted_hours": {"start": 22, "end": 6},
            "min_confidence": 0.7,
        },
        severity="critical",
        cooldown_sec=60,  # shorter cooldown — interior breach is very serious
        actions=["log", "alert"],
    ),

    # ── FRONT ENTRANCE — tight rules, this is the main entry point ──
    Rule(
        id="front_door_loiter",
        name="Front Door Loitering",
        rule_type="loiter_duration",
        cameras=["cam1"],
        conditions={
            "min_duration_sec": 60,  # 1 min at the front door = suspicious
            "zones": ["door", "front_entrance"],
        },
        severity="warning",
        cooldown_sec=120,
        actions=["log", "alert"],
    ),
    Rule(
        id="front_window_presence",
        name="Front Window Activity",
        rule_type="loiter_duration",
        cameras=["cam1"],
        conditions={
            "min_duration_sec": 30,  # 30 sec near windows = peeping
            "zones": ["front_windows"],
        },
        severity="critical",
        cooldown_sec=60,
        actions=["log", "alert"],
    ),

    # ── PARKING LOT — vehicle and person activity ──
    Rule(
        id="parking_loiter",
        name="Parking Lot Loitering",
        rule_type="loiter_duration",
        cameras=["cam2"],
        conditions={
            "min_duration_sec": 180,  # 3 min in parking lot = suspicious
        },
        severity="warning",
        cooldown_sec=180,
        actions=["log"],
    ),

    # ── BACK AREA — nobody should be here without reason ──
    Rule(
        id="back_area_presence",
        name="Back Area Presence",
        rule_type="loiter_duration",
        cameras=["cam3"],
        conditions={
            "min_duration_sec": 30,  # 30 sec in back area during hours = flag it
        },
        severity="warning",
        cooldown_sec=120,
        actions=["log"],
    ),

    # ── BACK DOOR — entry point, critical ──
    Rule(
        id="backdoor_activity",
        name="Back Door Activity",
        rule_type="loiter_duration",
        cameras=["cam4"],
        conditions={
            "min_duration_sec": 20,  # 20 sec at back door = why?
            "zones": ["backdoor", "back_door"],
        },
        severity="warning",
        cooldown_sec=60,
        actions=["log", "alert"],
    ),

    # ── OFFICE — shouldn't be occupied after hours ──
    Rule(
        id="office_presence",
        name="Office Presence After Hours",
        rule_type="time_restricted",
        cameras=["cam5", "cam7"],
        conditions={
            "restricted_hours": {"start": 20, "end": 6},  # 8pm-6am (earlier than perimeter)
            "min_confidence": 0.7,
        },
        severity="critical",
        cooldown_sec=60,
        actions=["log", "alert"],
    ),

    # ── WAREHOUSE — high value area ──
    Rule(
        id="warehouse_after_hours",
        name="Warehouse After Hours",
        rule_type="time_restricted",
        cameras=["cam6", "cam8"],
        conditions={
            "restricted_hours": {"start": 20, "end": 5},  # 8pm-5am
            "min_confidence": 0.7,
        },
        severity="critical",
        cooldown_sec=60,
        actions=["log", "alert"],
    ),
    Rule(
        id="warehouse_loiter",
        name="Warehouse Extended Presence",
        rule_type="loiter_duration",
        cameras=["cam6", "cam8"],
        conditions={
            "min_duration_sec": 300,  # 5 min on warehouse floor = normal work
        },
        severity="info",
        cooldown_sec=600,
        actions=["log"],
    ),

    # ── DRIVEWAY / VANS — cam9 watches the fleet ──
    Rule(
        id="driveway_presence",
        name="Driveway Activity",
        rule_type="loiter_duration",
        cameras=["cam9"],
        conditions={
            "min_duration_sec": 60,  # 1 min near the vans
        },
        severity="warning",
        cooldown_sec=120,
        actions=["log", "alert"],
    ),

    # ── GLOBAL RULES (all cameras) ──
    Rule(
        id="running",
        name="Running Detected",
        rule_type="speed_threshold",
        cameras=["*"],
        conditions={
            "min_speed_px_sec": 200,
        },
        severity="warning",
        cooldown_sec=60,
        actions=["log"],
    ),
    Rule(
        id="crowding",
        name="Crowding Alert",
        rule_type="person_count",
        cameras=["*"],
        conditions={
            "max_persons": 5,
        },
        severity="warning",
        cooldown_sec=120,
        actions=["log", "alert"],
    ),
    Rule(
        id="anomaly_high",
        name="High Anomaly Score",
        rule_type="anomaly_score",
        cameras=["*"],
        conditions={
            "min_score": 0.8,
        },
        severity="critical",
        cooldown_sec=60,
        actions=["log", "alert"],
    ),
    Rule(
        id="multi_cam_track",
        name="Cross-Camera Movement",
        rule_type="multi_camera",
        cameras=["*"],
        conditions={
            "max_transition_sec": 30,
        },
        severity="info",
        cooldown_sec=60,
        actions=["log"],
    ),
]


class RulesEngine:
    """Evaluates rules against incoming events and detection data."""

    def __init__(self):
        self.rules: dict[str, Rule] = {}
        self._cooldowns: dict[tuple, float] = {}  # (rule_id, camera) -> last_fire
        self._person_counts: dict[str, int] = {}  # camera -> count
        self._recent_tracks: dict[str, dict] = {}  # track_id -> {camera, time}

        # Load default rules
        for rule in DEFAULT_RULES:
            self.rules[rule.id] = rule

        logger.info(f"Rules engine initialized with {len(self.rules)} rules")

    def add_rule(self, rule: Rule):
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str):
        self.rules.pop(rule_id, None)

    def get_rules(self) -> list[dict]:
        return [r.to_dict() for r in self.rules.values()]

    def evaluate_detection(self, camera: str, track_id: str, label: str,
                           score: float, zones: list, speed: float = 0,
                           duration: float = 0) -> list[dict]:
        """Evaluate all rules against a new detection."""
        fired_events = []
        now = time.time()

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # Check camera match
            if "*" not in rule.cameras and camera not in rule.cameras:
                continue

            # Check cooldown
            cooldown_key = (rule.id, camera)
            if cooldown_key in self._cooldowns:
                if now - self._cooldowns[cooldown_key] < rule.cooldown_sec:
                    continue

            # Evaluate rule
            triggered = False
            event_data = {"rule": rule.id, "rule_name": rule.name}

            if rule.rule_type == "time_restricted":
                triggered = self._eval_time_restricted(rule, score)
                if triggered:
                    event_data["reason"] = "Presence detected during restricted hours"

            elif rule.rule_type == "speed_threshold":
                triggered = speed > rule.conditions.get("min_speed_px_sec", 200)
                if triggered:
                    event_data["speed"] = round(speed, 1)

            elif rule.rule_type == "loiter_duration":
                # Check zone filter if specified
                required_zones = rule.conditions.get("zones")
                if required_zones and zones:
                    # Only trigger if person is in one of the required zones
                    in_zone = any(z in required_zones for z in zones)
                    if not in_zone:
                        continue
                elif required_zones and not zones:
                    continue  # Rule requires zones but person has none

                triggered = duration > rule.conditions.get("min_duration_sec", 120)
                if triggered:
                    event_data["duration"] = round(duration, 1)
                    if zones:
                        event_data["zones"] = zones

            elif rule.rule_type == "person_count":
                count = self._person_counts.get(camera, 0)
                triggered = count > rule.conditions.get("max_persons", 5)
                if triggered:
                    event_data["person_count"] = count

            elif rule.rule_type == "multi_camera":
                triggered, transition_data = self._eval_multi_camera(
                    track_id, camera, rule
                )
                if triggered:
                    event_data.update(transition_data)

            if triggered:
                self._cooldowns[cooldown_key] = now
                event = {
                    "type": f"rule_{rule.rule_type}",
                    "severity": rule.severity,
                    "camera": camera,
                    "track_id": track_id,
                    "data": event_data,
                    "actions": rule.actions,
                }
                fired_events.append(event)

                # Store in database
                db.insert_event(
                    camera=camera,
                    event_type=f"rule_{rule.rule_type}",
                    severity=rule.severity,
                    track_id=track_id,
                    data=event_data,
                )

                logger.info(f"Rule fired: {rule.name} on {camera} "
                           f"(severity: {rule.severity})")

        # Track for multi-camera detection
        if track_id and label == "person":
            self._recent_tracks[track_id] = {"camera": camera, "time": now}
            # Cleanup old entries
            cutoff = now - 120
            self._recent_tracks = {
                k: v for k, v in self._recent_tracks.items() if v["time"] > cutoff
            }

        return fired_events

    def evaluate_anomaly(self, camera: str, score: float) -> list[dict]:
        """Evaluate anomaly-specific rules."""
        fired_events = []
        now = time.time()

        for rule in self.rules.values():
            if not rule.enabled or rule.rule_type != "anomaly_score":
                continue
            if "*" not in rule.cameras and camera not in rule.cameras:
                continue

            cooldown_key = (rule.id, camera)
            if cooldown_key in self._cooldowns:
                if now - self._cooldowns[cooldown_key] < rule.cooldown_sec:
                    continue

            if score >= rule.conditions.get("min_score", 0.8):
                self._cooldowns[cooldown_key] = now
                event_data = {
                    "rule": rule.id,
                    "rule_name": rule.name,
                    "anomaly_score": round(score, 3),
                }
                fired_events.append({
                    "type": "rule_anomaly",
                    "severity": rule.severity,
                    "camera": camera,
                    "data": event_data,
                    "actions": rule.actions,
                })
                db.insert_event(
                    camera=camera,
                    event_type="rule_anomaly",
                    severity=rule.severity,
                    data=event_data,
                )

        return fired_events

    def update_person_count(self, camera: str, count: int):
        """Update tracked person count for a camera."""
        self._person_counts[camera] = count

    def _eval_time_restricted(self, rule: Rule, score: float) -> bool:
        """Check if current time falls within restricted hours."""
        from datetime import datetime

        conditions = rule.conditions
        min_confidence = conditions.get("min_confidence", 0.5)
        if score < min_confidence:
            return False

        now = datetime.now()
        start_hour = conditions.get("restricted_hours", {}).get("start", 22)
        end_hour = conditions.get("restricted_hours", {}).get("end", 6)

        current_hour = now.hour
        if start_hour > end_hour:  # Spans midnight
            return current_hour >= start_hour or current_hour < end_hour
        else:
            return start_hour <= current_hour < end_hour

    def _eval_multi_camera(self, track_id: str, camera: str,
                           rule: Rule) -> tuple[bool, dict]:
        """Check if the same person appeared on a different camera recently."""
        if not track_id or track_id not in self._recent_tracks:
            return False, {}

        prev = self._recent_tracks[track_id]
        if prev["camera"] == camera:
            return False, {}

        max_transition = rule.conditions.get("max_transition_sec", 30)
        elapsed = time.time() - prev["time"]

        if elapsed <= max_transition:
            return True, {
                "from_camera": prev["camera"],
                "to_camera": camera,
                "transition_sec": round(elapsed, 1),
            }

        return False, {}
