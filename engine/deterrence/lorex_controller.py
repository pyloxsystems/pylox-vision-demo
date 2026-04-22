"""Lorex/Dahua NVR Coaxial Control — strobe lights via HTTP digest auth.

Active deterrence: when a confirmed threat is detected, flash the white light
on the camera. Most intruders flee when they realize they're being filmed.

API verified working on Lorex N910A6 NVR with cameras 1, 2, 3, 4, 9.
Cameras 5-8 (interior) don't have built-in strobes.

Reference: Dahua coaxial control HTTP API
  GET http://{nvr}/cgi-bin/coaxialControlIO.cgi?action=control&channel={N}&info[0].Type={T}&info[0].IO={S}
    Type=1: White light
    Type=2: Audible alarm (speaker)
    IO=1: ON
    IO=2: OFF
"""

import os
import time
import asyncio
import logging
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger("pylox-v2.deterrence.lorex")

NVR_HOST = os.getenv("NVR_HOST", "192.168.1.100")
NVR_USER = os.getenv("NVR_USER", "admin")
NVR_PASS = os.getenv("NVR_PASS", "YOUR_NVR_PASSWORD")

# Cameras with verified strobe hardware (outdoor cams)
CAMERAS_WITH_STROBE = {"cam1", "cam2", "cam3", "cam4", "cam9"}

# Map cam names to NVR channel numbers
CAMERA_CHANNELS = {
    "cam1": 1, "cam2": 2, "cam3": 3, "cam4": 4, "cam5": 5,
    "cam6": 6, "cam7": 7, "cam8": 8, "cam9": 9,
}


class LorexController:
    """Controls Lorex/Dahua NVR coaxial deterrence (strobe lights)."""

    def __init__(self, host: str = None, user: str = None, password: str = None):
        self.host = host or NVR_HOST
        self.user = user or NVR_USER
        self.password = password or NVR_PASS
        self._active_strobes = {}  # camera -> end_timestamp
        self._auto_off_tasks = {}  # camera -> asyncio.Task
        self._stats = {
            "strobes_triggered": 0,
            "strobes_active": 0,
            "errors": 0,
        }

    def can_strobe(self, camera: str) -> bool:
        """Check if a camera supports strobe."""
        return camera in CAMERAS_WITH_STROBE

    async def strobe_on(self, camera: str, duration_sec: int = 15) -> bool:
        """Turn on strobe for a camera. Auto-off after duration."""
        if not self.can_strobe(camera):
            logger.warning(f"Camera {camera} does not support strobe")
            return False

        channel = CAMERA_CHANNELS.get(camera)
        if not channel:
            return False

        success = await self._send_control(channel, type_id=1, io=1)
        if success:
            self._stats["strobes_triggered"] += 1
            self._active_strobes[camera] = time.time() + duration_sec
            logger.warning(f"⚡ STROBE ON: {camera} for {duration_sec}s")

            # Cancel any existing auto-off
            if camera in self._auto_off_tasks:
                self._auto_off_tasks[camera].cancel()

            # Schedule auto-off
            self._auto_off_tasks[camera] = asyncio.create_task(
                self._auto_off(camera, duration_sec)
            )

        self._stats["strobes_active"] = len(self._active_strobes)
        return success

    async def strobe_off(self, camera: str) -> bool:
        """Turn off strobe for a camera."""
        if not self.can_strobe(camera):
            return False

        channel = CAMERA_CHANNELS.get(camera)
        if not channel:
            return False

        success = await self._send_control(channel, type_id=1, io=2)
        if success:
            self._active_strobes.pop(camera, None)
            if camera in self._auto_off_tasks:
                self._auto_off_tasks[camera].cancel()
                del self._auto_off_tasks[camera]
            logger.info(f"Strobe OFF: {camera}")

        self._stats["strobes_active"] = len(self._active_strobes)
        return success

    async def strobe_all_off(self):
        """Emergency: turn off all strobes."""
        for camera in list(self._active_strobes.keys()):
            await self.strobe_off(camera)

    async def _auto_off(self, camera: str, duration_sec: int):
        """Auto-turn-off after duration."""
        try:
            await asyncio.sleep(duration_sec)
            await self.strobe_off(camera)
        except asyncio.CancelledError:
            pass

    async def _send_control(self, channel: int, type_id: int, io: int) -> bool:
        """Send a coaxial control HTTP request to the NVR."""
        url = (f"http://{self.host}/cgi-bin/coaxialControlIO.cgi"
               f"?action=control&channel={channel}"
               f"&info[0].Type={type_id}&info[0].IO={io}")

        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._do_request, url)
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Lorex API error: {e}")
            return False

    def _do_request(self, url: str) -> bool:
        """Synchronous HTTP digest request (run in thread executor)."""
        import urllib.request
        import urllib.error

        # Build digest auth handler
        password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, url, self.user, self.password)
        handler = urllib.request.HTTPDigestAuthHandler(password_mgr)
        opener = urllib.request.build_opener(handler)

        try:
            resp = opener.open(url, timeout=5)
            return resp.status == 200
        except urllib.error.HTTPError as e:
            # 401 on first request is normal for digest — opener handles retry
            return e.code in (200, 204)
        except Exception as e:
            logger.error(f"Lorex request failed: {e}")
            return False

    def is_active(self, camera: str) -> bool:
        """Check if a strobe is currently active on a camera."""
        if camera not in self._active_strobes:
            return False
        return time.time() < self._active_strobes[camera]

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "active_cameras": list(self._active_strobes.keys()),
            "supported_cameras": list(CAMERAS_WITH_STROBE),
        }
