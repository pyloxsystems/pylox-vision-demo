"""WatchManager — orchestrates active WatchSessions across all cameras.

Handles:
  - Cooldowns (don't restart session immediately on same camera)
  - Circuit breakers (max sessions per night)
  - Session deduplication (one session per active incident)
  - Business hours check (no sessions during business hours)
"""

import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Callable

from engine import database as db
from engine.gemini.connector import GeminiConnector
from engine.gemini.watch_session import WatchSession

logger = logging.getLogger("pylox-v2.watch.manager")

# Cooldowns
SESSION_COOLDOWN_SEC = 60          # Don't restart session on same camera within 60s
MAX_SESSIONS_PER_NIGHT = 20        # Circuit breaker

# Camera descriptions for prompts
CAMERA_DESCRIPTIONS = {
    "cam1": "Front Entrance — main entry door and front windows",
    "cam2": "Parking Lot — company van fleet and customer parking",
    "cam3": "Back Area — rear exterior of the building",
    "cam4": "Back Door — rear entry/exit",
    "cam5": "Office Lobby — interior reception",
    "cam6": "Warehouse — main warehouse floor",
    "cam7": "Office Entrance — glass doors from inside",
    "cam8": "Shop Floor — high angle warehouse view",
    "cam9": "Driveway — outdoor view between vans",
}


class WatchManager:
    """Manages active WatchSessions, prevents duplicates, enforces limits."""

    def __init__(
        self,
        gemini: GeminiConnector,
        signal_collector=None,
        deterrent=None,
        site_config: dict = None,
        on_narration: Callable = None,
        on_action: Callable = None,
        on_end: Callable = None,
        on_start: Callable = None,
    ):
        self.gemini = gemini
        self.signals = signal_collector
        self.deterrent = deterrent
        self.site_config = site_config or {
            "id": "default",
            "name": "ACME-CORP Warehouse",
            "type": "window and door installation warehouse",
            "hours": {"open": 6, "close": 20},
        }
        self.on_narration = on_narration
        self.on_action = on_action
        self.on_end = on_end
        self.on_start = on_start

        self._active_sessions: Dict[str, WatchSession] = {}  # camera -> session
        self._cooldowns: Dict[str, float] = {}  # camera -> last end time
        self._sessions_today = 0
        self._sessions_reset_date = None
        self._stats = {
            "sessions_started": 0,
            "sessions_completed": 0,
            "false_alarms_caught": 0,
            "deterrents_triggered": 0,
            "skipped_business_hours": 0,
            "skipped_cooldown": 0,
            "skipped_circuit_breaker": 0,
            "skipped_already_active": 0,
        }

    def is_business_hours(self) -> bool:
        hour = datetime.now().hour
        open_h = self.site_config.get("hours", {}).get("open", 6)
        close_h = self.site_config.get("hours", {}).get("close", 20)
        return open_h <= hour < close_h

    def _reset_daily_counter(self):
        today = datetime.now().date()
        if self._sessions_reset_date != today:
            self._sessions_today = 0
            self._sessions_reset_date = today

    async def trigger(self, camera: str, trigger_source: str,
                      track_id: str = None, force: bool = False) -> bool:
        """Trigger a watch session on a camera.

        force=True bypasses business hours check (used for door sensors).
        Returns True if session started.
        """
        self._reset_daily_counter()

        # Business hours check
        if not force and self.is_business_hours():
            self._stats["skipped_business_hours"] += 1
            return False

        # Already active session on this camera?
        if camera in self._active_sessions:
            session = self._active_sessions[camera]
            if session.is_active:
                self._stats["skipped_already_active"] += 1
                return False
            else:
                del self._active_sessions[camera]

        # Cooldown check
        if camera in self._cooldowns:
            if time.time() - self._cooldowns[camera] < SESSION_COOLDOWN_SEC:
                self._stats["skipped_cooldown"] += 1
                return False

        # Circuit breaker
        if self._sessions_today >= MAX_SESSIONS_PER_NIGHT:
            self._stats["skipped_circuit_breaker"] += 1
            logger.warning(f"Max sessions per night ({MAX_SESSIONS_PER_NIGHT}) reached")
            return False

        # Create incident in DB
        incident_id = db.create_incident(
            camera=camera,
            trigger_source=trigger_source,
            track_id=track_id,
        )

        # Create session
        session = WatchSession(
            incident_id=incident_id,
            camera=camera,
            trigger_source=trigger_source,
            track_id=track_id,
            gemini=self.gemini,
            signal_collector=self.signals,
            site_config=self.site_config,
            on_narration=self.on_narration,
            on_action=self.on_action,
            on_end=self._handle_session_end,
            deterrent=self.deterrent,
        )

        self._active_sessions[camera] = session
        self._sessions_today += 1
        self._stats["sessions_started"] += 1

        # Broadcast start
        if self.on_start:
            self.on_start({
                "incident_id": incident_id,
                "camera": camera,
                "trigger": trigger_source,
                "timestamp": time.time(),
            })

        # Start the session asynchronously
        await session.start()

        logger.info(f"WatchManager: Started session #{incident_id} on {camera} (trigger: {trigger_source})")
        return True

    def _handle_session_end(self, end_event: dict):
        """Called when a session ends — record cooldown, broadcast."""
        camera = end_event["camera"]
        resolution = end_event.get("resolution", "unknown")

        self._cooldowns[camera] = time.time()
        self._stats["sessions_completed"] += 1

        if resolution == "false_alarm":
            self._stats["false_alarms_caught"] += 1

        # Broadcast to clients
        if self.on_end:
            self.on_end(end_event)

    def get_active_sessions(self) -> dict:
        return {
            camera: {
                "incident_id": s.incident_id,
                "state": s.state,
                "start_time": s.start_time,
                "max_threat": s.max_threat,
                "narration_count": len(s.history),
                "deterrent_active": s.deterrent_active,
            }
            for camera, s in self._active_sessions.items()
            if s.is_active
        }

    async def stop_all(self):
        """Stop all active sessions (shutdown)."""
        for camera, session in list(self._active_sessions.items()):
            await session.stop("shutdown")

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "active_sessions": len(self.get_active_sessions()),
            "sessions_today": self._sessions_today,
            "max_per_night": MAX_SESSIONS_PER_NIGHT,
        }
