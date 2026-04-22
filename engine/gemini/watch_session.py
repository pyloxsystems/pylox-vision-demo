"""WatchSession — manages a single security incident lifecycle.

When triggered, a WatchSession engages Gemini to actively watch the camera
in 5-second intervals, narrating what's happening, until the threat resolves
or times out.

State machine:
  IDLE → ASSESSING (1st Gemini call)
       → not real → CLEARED (false alarm)
       → real → WATCHING
              → loops every 5 sec
              → executes deterrent on threat 7+
              → ends on threat clear, timeout, or manual dismiss
"""

import io
import time
import asyncio
import logging
import urllib.request
import subprocess
from typing import Optional, Callable
from datetime import datetime

from engine import database as db
from engine.config import FRIGATE_API
from engine.gemini.connector import GeminiConnector, GeminiAssessment

logger = logging.getLogger("pylox-v2.watch")

# Session config
WATCH_INTERVAL_SEC = 5
SESSION_MAX_DURATION_SEC = 300  # 5 min hard cap
THREAT_CLEAR_DURATION_SEC = 15  # threat must be cleared for 15s to end session
DETERRENT_THRESHOLD = 7
PANIC_THRESHOLD = 9


class WatchSession:
    """One active incident being watched by Gemini."""

    def __init__(
        self,
        incident_id: int,
        camera: str,
        trigger_source: str,
        track_id: str = None,
        gemini: GeminiConnector = None,
        signal_collector=None,
        site_config: dict = None,
        on_narration: Callable = None,
        on_action: Callable = None,
        on_end: Callable = None,
        deterrent: object = None,
    ):
        self.incident_id = incident_id
        self.camera = camera
        self.trigger_source = trigger_source
        self.track_id = track_id
        self.gemini = gemini
        self.signals = signal_collector
        self.site_config = site_config
        self.on_narration = on_narration
        self.on_action = on_action
        self.on_end = on_end
        self.deterrent = deterrent

        self.start_time = time.time()
        self.history = []  # List of narration dicts
        self.state = "ASSESSING"
        self.max_threat = 0
        self.threat_clear_since = None
        self.deterrent_active = False
        self.notification_sent = False
        self._task = None
        self._stopped = False

    async def start(self):
        """Start watching this incident."""
        logger.info(f"WatchSession START: incident={self.incident_id} camera={self.camera} trigger={self.trigger_source}")
        self._task = asyncio.create_task(self._watch_loop())

    async def stop(self, resolution: str = "manual_dismiss"):
        """Stop the session."""
        if self._stopped:
            return
        self._stopped = True
        if self._task:
            self._task.cancel()
        await self._end(resolution)

    async def _watch_loop(self):
        """Main watch loop — calls Gemini every 5 seconds until threat clears."""
        try:
            while not self._stopped:
                # Check session timeout
                elapsed = time.time() - self.start_time
                if elapsed > SESSION_MAX_DURATION_SEC:
                    logger.info(f"WatchSession TIMEOUT: incident={self.incident_id}")
                    await self._end("timeout")
                    return

                # Grab last 5 seconds of video
                video_bytes = await self._grab_video_clip()
                if not video_bytes:
                    logger.warning(f"No video clip for incident {self.incident_id}, retrying...")
                    await asyncio.sleep(WATCH_INTERVAL_SEC)
                    continue

                # Collect current signals
                signal_context = ""
                if self.signals:
                    collected = self.signals.collect(
                        camera=self.camera, track_id=self.track_id or "watch",
                        label="person", score=0.9, zones=[], duration=elapsed,
                    )
                    signal_context = self.signals.build_prompt_context(collected)

                # Call Gemini
                mode = "initial" if len(self.history) == 0 else "continuation"
                assessment = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.gemini.analyze_video_clip(
                        video_bytes=video_bytes,
                        camera_id=self.camera,
                        camera_name=self.camera,
                        mode=mode,
                        history=self.history,
                        signal_context=signal_context,
                        site_config=self.site_config,
                    )
                )

                if not assessment:
                    logger.warning(f"Gemini returned None for incident {self.incident_id}")
                    await asyncio.sleep(WATCH_INTERVAL_SEC)
                    continue

                # Process assessment
                await self._process_assessment(assessment)

                # Check end conditions
                if not assessment.real and len(self.history) <= 1:
                    # First call confirmed false positive
                    await self._end("false_alarm")
                    return

                if hasattr(assessment, 'raw') and assessment.raw.get("should_continue") is False:
                    await self._end("cleared")
                    return

                # Check threat-clear timer
                if assessment.threat < 4:
                    if self.threat_clear_since is None:
                        self.threat_clear_since = time.time()
                    elif time.time() - self.threat_clear_since > THREAT_CLEAR_DURATION_SEC:
                        await self._end("cleared")
                        return
                else:
                    self.threat_clear_since = None

                # Wait before next call
                await asyncio.sleep(WATCH_INTERVAL_SEC)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WatchSession error: {e}")
            await self._end("error")

    async def _process_assessment(self, assessment: GeminiAssessment):
        """Handle a Gemini assessment — update state, store, broadcast, act."""
        narration_text = assessment.description or "Watching..."

        # Store in history
        narration = {
            "timestamp": time.time(),
            "threat": assessment.threat,
            "narration": narration_text,
            "type": assessment.type,
            "action": assessment.action,
        }
        self.history.append(narration)
        self.max_threat = max(self.max_threat, assessment.threat)

        # Persist narration
        db.add_narration(
            incident_id=self.incident_id,
            threat=assessment.threat,
            narration=narration_text,
            action=assessment.action,
        )

        # Broadcast narration
        if self.on_narration:
            self.on_narration({
                "incident_id": self.incident_id,
                "camera": self.camera,
                "threat": assessment.threat,
                "narration": narration_text,
                "type": assessment.type,
                "timestamp": time.time(),
            })

        # Execute deterrent if threat warrants and not already active
        if assessment.threat >= DETERRENT_THRESHOLD and not self.deterrent_active:
            await self._trigger_deterrent()

        # Send notification (once per incident)
        if assessment.threat >= 4 and not self.notification_sent:
            await self._send_notification(assessment)

        # Transition to WATCHING state after first real assessment
        if self.state == "ASSESSING" and assessment.real:
            self.state = "WATCHING"

    async def _trigger_deterrent(self):
        """Activate strobe on this camera."""
        if not self.deterrent:
            return

        if not self.deterrent.can_strobe(self.camera):
            logger.warning(f"Camera {self.camera} cannot strobe")
            return

        success = await self.deterrent.strobe_on(self.camera, duration_sec=15)
        if success:
            self.deterrent_active = True
            db.update_incident(self.incident_id, deterrent_triggered=1)

            if self.on_action:
                self.on_action({
                    "incident_id": self.incident_id,
                    "camera": self.camera,
                    "action": "strobe",
                    "timestamp": time.time(),
                })

    async def _send_notification(self, assessment: GeminiAssessment):
        """Trigger notification (push first, SMS fallback)."""
        if self.on_action:
            self.on_action({
                "incident_id": self.incident_id,
                "camera": self.camera,
                "action": "notification",
                "threat": assessment.threat,
                "description": assessment.description,
                "timestamp": time.time(),
            })
        self.notification_sent = True
        db.update_incident(self.incident_id, notification_sent=1)

    async def _grab_video_clip(self, duration: int = 5) -> Optional[bytes]:
        """Grab the last N seconds of video from Frigate as MP4 bytes."""
        loop = asyncio.get_event_loop()
        try:
            # Use Frigate's clip API
            end_time = int(time.time())
            start_time = end_time - duration
            url = f"{FRIGATE_API}/api/{self.camera}/start/{start_time}/end/{end_time}/clip.mp4?download=true"

            def fetch():
                req = urllib.request.urlopen(url, timeout=10)
                return req.read()

            data = await loop.run_in_executor(None, fetch)
            if data and len(data) > 1000:
                return data
        except Exception as e:
            logger.debug(f"Clip fetch failed: {e}")

        return None

    async def _end(self, resolution: str):
        """Finalize the session."""
        if self._stopped:
            return
        self._stopped = True

        # Update incident in DB
        db.update_incident(
            self.incident_id,
            max_threat=self.max_threat,
            total_gemini_calls=len(self.history),
        )
        db.end_incident(self.incident_id, resolution)

        # Turn off any active deterrent
        if self.deterrent_active and self.deterrent:
            await self.deterrent.strobe_off(self.camera)

        # Generate police artifacts (narrative + tactical brief) in the background
        # so we don't block the state transition. Only if this was a real incident
        # with enough observations to be useful.
        if resolution not in ("false_alarm", "error") and len(self.history) > 0 and self.max_threat >= 4:
            try:
                asyncio.create_task(self._generate_police_artifacts())
            except Exception as e:
                logger.error(f"Failed to queue police artifacts: {e}")

        logger.info(f"WatchSession END: incident={self.incident_id} resolution={resolution} max_threat={self.max_threat} narrations={len(self.history)}")

        # Broadcast end
        if self.on_end:
            self.on_end({
                "incident_id": self.incident_id,
                "camera": self.camera,
                "resolution": resolution,
                "max_threat": self.max_threat,
                "narration_count": len(self.history),
                "duration": time.time() - self.start_time,
                "timestamp": time.time(),
            })

    async def _generate_police_artifacts(self):
        """Background task — generate report narrative + tactical brief after incident closes.

        Runs after _end() so it doesn't block the state transition. Results are
        stored in incident_police table and broadcast via WebSocket so the UI
        can light up the "prosecutor-ready evidence available" indicator.
        """
        try:
            import json as _json
            from engine.police import report_narrative as narrative_mod
            from engine.police import tactical_brief as tactical_mod

            # Rehydrate the persisted incident so we get final times/threat/resolution
            incident = db.get_incident(self.incident_id)
            if not incident:
                return

            loop = asyncio.get_event_loop()

            narrative = await loop.run_in_executor(
                None,
                lambda: narrative_mod.generate_narrative(
                    incident=incident,
                    history=self.history,
                    site_config=self.site_config or {},
                    gemini_connector=self.gemini,
                    camera_name=self.camera,
                ),
            )

            brief = await loop.run_in_executor(
                None,
                lambda: tactical_mod.generate_brief(
                    incident=incident,
                    history=self.history,
                    site_config=self.site_config or {},
                    gemini_connector=self.gemini,
                    cameras_involved=[self.camera],
                    camera_name=self.camera,
                ),
            )

            db.save_incident_police_data(
                incident_id=self.incident_id,
                report_narrative=narrative,
                tactical_brief=_json.dumps(brief) if brief else None,
            )

            logger.info(
                f"Police artifacts generated: incident={self.incident_id} "
                f"narrative={'yes' if narrative else 'no'} "
                f"tactical={'yes' if brief else 'no'}"
            )

            # Broadcast police-ready event so UI can light up the indicator
            if self.on_action and (narrative or brief):
                self.on_action({
                    "incident_id": self.incident_id,
                    "camera": self.camera,
                    "action": "police_ready",
                    "has_narrative": bool(narrative),
                    "has_tactical_brief": bool(brief),
                    "timestamp": time.time(),
                })

        except Exception as e:
            logger.error(f"Police artifact generation failed for incident {self.incident_id}: {e}")

    @property
    def is_active(self) -> bool:
        return not self._stopped
