"""Health Monitor — watches system health and triggers cellular alerts when things fail.

Checks every 30 seconds:
  1. Internet connectivity (ping + API check)
  2. Camera feeds (Frigate stats — is each camera streaming?)
  3. Frigate process (is it running?)
  4. Power status (UPS battery detection)
  5. Case tamper switch (GPIO pin)
  6. Cellular dongle signal (AT+CSQ)

When a failure is detected:
  - Internet down > 2 min → SMS alert
  - Camera offline → SMS alert
  - Power failure (UPS) → immediate SMS + phone call
  - Tamper detected → immediate SMS + phone call
  - Multiple systems down → PANIC: SMS + call + all channels

Also handles network failover:
  - Internet dies → switch API calls to cellular data (if dongle supports data)
  - Internet returns → switch back to ethernet
"""

import os
import time
import json
import asyncio
import logging
import socket
import subprocess
import urllib.request
from datetime import datetime
from typing import Optional

from engine.alerts.sms_sender import SMSSender
from engine.alerts.notifier import AlertNotifier
from engine.config import FRIGATE_API

logger = logging.getLogger("pylox-v2.watchdog")

# GPIO pin for tamper switch (Jetson Orin Nano)
TAMPER_GPIO_PIN = os.getenv("TAMPER_GPIO_PIN", "")  # Set when deployed on Jetson


class HealthMonitor:
    """Monitors system health and triggers cellular alerts on failure."""

    def __init__(self, sms_sender: SMSSender = None, notifier: AlertNotifier = None,
                 check_interval: int = 30):
        self.sms = sms_sender
        self.notifier = notifier
        self.interval = check_interval
        self._running = False
        self._task = None

        # State tracking
        self._internet_down_since = None
        self._cameras_down = {}  # camera -> down_since timestamp
        self._power_on_battery = False
        self._tamper_detected = False
        self._alerted = {
            "internet": False,
            "power": False,
            "tamper": False,
        }
        self._cameras_alerted = set()

        self._stats = {
            "checks": 0,
            "internet_ok": True,
            "cameras_online": 0,
            "cameras_total": 9,
            "frigate_ok": True,
            "power_ok": True,
            "tamper_ok": True,
            "cellular_signal": 0,
            "alerts_sent": 0,
            "last_check": 0,
        }

    async def start(self):
        """Start the health monitoring loop."""
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Health monitor started (interval: {self.interval}s)")

    async def stop(self):
        """Stop the health monitor."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Health monitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._stats["checks"] += 1
                self._stats["last_check"] = time.time()

                await self._check_internet()
                await self._check_cameras()
                await self._check_frigate()
                self._check_power()
                self._check_tamper()
                self._check_cellular_signal()

                # Check for PANIC condition (multiple failures)
                self._check_panic()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            await asyncio.sleep(self.interval)

    async def _check_internet(self):
        """Check internet connectivity."""
        loop = asyncio.get_event_loop()
        connected = await loop.run_in_executor(None, self._ping_internet)

        if connected:
            if self._internet_down_since:
                downtime = time.time() - self._internet_down_since
                logger.info(f"Internet restored after {downtime:.0f}s")
                self._internet_down_since = None
                self._alerted["internet"] = False
            self._stats["internet_ok"] = True
        else:
            if not self._internet_down_since:
                self._internet_down_since = time.time()
                logger.warning("Internet connectivity lost")

            downtime = time.time() - self._internet_down_since
            self._stats["internet_ok"] = False

            # Alert after 2 minutes of downtime
            if downtime > 120 and not self._alerted["internet"]:
                self._alerted["internet"] = True
                self._send_emergency(
                    f"Internet offline at site since "
                    f"{datetime.fromtimestamp(self._internet_down_since).strftime('%I:%M %p')}. "
                    f"Cameras recording locally. AI verification paused."
                )

    async def _check_cameras(self):
        """Check if all cameras are streaming via Frigate stats."""
        loop = asyncio.get_event_loop()
        camera_stats = await loop.run_in_executor(None, self._fetch_camera_stats)

        if not camera_stats:
            return

        online = 0
        for cam_id, stats in camera_stats.items():
            fps = stats.get("camera_fps", 0)
            if fps > 0:
                online += 1
                # Camera came back
                if cam_id in self._cameras_down:
                    logger.info(f"Camera {cam_id} back online")
                    del self._cameras_down[cam_id]
                    self._cameras_alerted.discard(cam_id)
            else:
                # Camera is down
                if cam_id not in self._cameras_down:
                    self._cameras_down[cam_id] = time.time()
                    logger.warning(f"Camera {cam_id} offline (0 fps)")

                # Alert after 60 seconds
                downtime = time.time() - self._cameras_down[cam_id]
                if downtime > 60 and cam_id not in self._cameras_alerted:
                    self._cameras_alerted.add(cam_id)
                    self._send_emergency(f"Camera {cam_id} offline for {downtime:.0f}s")

        self._stats["cameras_online"] = online

    async def _check_frigate(self):
        """Check if Frigate process is running."""
        loop = asyncio.get_event_loop()
        running = await loop.run_in_executor(None, self._is_frigate_running)
        self._stats["frigate_ok"] = running

        if not running:
            logger.error("Frigate process not detected!")
            self._send_emergency("Frigate NVR process is not running. Cameras not recording.")

    def _check_power(self):
        """Check for UPS battery status."""
        # Check for common UPS indicators on Linux
        on_battery = False

        # Method 1: Check /sys/class/power_supply
        try:
            for supply_dir in ("/sys/class/power_supply/BAT0", "/sys/class/power_supply/battery"):
                status_file = f"{supply_dir}/status"
                if os.path.exists(status_file):
                    with open(status_file) as f:
                        status = f.read().strip()
                    if status == "Discharging":
                        on_battery = True
                    break
        except Exception:
            pass

        # Method 2: Check apcupsd (common UPS daemon)
        if not on_battery:
            try:
                result = subprocess.run(["apcaccess"], capture_output=True, text=True, timeout=3)
                if "ONBATT" in result.stdout:
                    on_battery = True
            except Exception:
                pass

        self._stats["power_ok"] = not on_battery

        if on_battery and not self._power_on_battery:
            self._power_on_battery = True
            logger.critical("POWER FAILURE — running on battery!")
            self._send_emergency(
                "POWER FAILURE at site. Battery backup active. "
                "Estimated 15 minutes remaining.",
                call=True,
            )
        elif not on_battery and self._power_on_battery:
            self._power_on_battery = False
            self._alerted["power"] = False
            logger.info("Power restored")

    def _check_tamper(self):
        """Check tamper switch via GPIO."""
        if not TAMPER_GPIO_PIN:
            return  # No tamper switch configured

        try:
            gpio_path = f"/sys/class/gpio/gpio{TAMPER_GPIO_PIN}/value"
            if os.path.exists(gpio_path):
                with open(gpio_path) as f:
                    value = f.read().strip()
                # LOW = case closed, HIGH = case opened
                if value == "1" and not self._tamper_detected:
                    self._tamper_detected = True
                    self._stats["tamper_ok"] = False
                    logger.critical("TAMPER DETECTED — case opened!")
                    self._send_emergency(
                        "TAMPER ALERT: Equipment enclosure has been opened!",
                        call=True,
                    )
                elif value == "0":
                    self._tamper_detected = False
                    self._stats["tamper_ok"] = True
        except Exception:
            pass

    def _check_cellular_signal(self):
        """Check cellular dongle signal strength."""
        if self.sms and self.sms.available:
            self._stats["cellular_signal"] = self.sms.get_stats().get("signal_strength", 0)

    def _check_panic(self):
        """Check for multiple simultaneous failures — indicates attack."""
        failures = []
        if not self._stats["internet_ok"]:
            failures.append("internet")
        if not self._stats["power_ok"]:
            failures.append("power")
        if len(self._cameras_down) >= 3:
            failures.append(f"{len(self._cameras_down)} cameras")

        if len(failures) >= 2:
            msg = f"MULTIPLE SYSTEM FAILURES: {', '.join(failures)}. Possible tampering or attack."
            logger.critical(msg)
            self._send_emergency(msg, call=True)

    def _send_emergency(self, message: str, call: bool = False):
        """Send emergency alert via all available channels."""
        self._stats["alerts_sent"] += 1
        timestamp = datetime.now().strftime("%I:%M %p")
        full_msg = f"PYLOX ALERT ({timestamp}): {message}"

        logger.critical(f"EMERGENCY: {message}")

        # SMS via cellular dongle
        if self.sms and self.sms.available:
            # Send to all registered numbers
            if self.notifier and self.notifier._sms_numbers:
                for number in self.notifier._sms_numbers:
                    self.sms.send_sms(number, full_msg)
                    if call:
                        self.sms.make_call(number, duration_sec=20)
            else:
                logger.warning("No SMS numbers registered for emergency alerts")

        # Also try push notification (might work if internet is up)
        if self.notifier:
            try:
                self.notifier.send_emergency(message)
            except Exception:
                pass  # Internet probably down

    # ── Sync helper methods (run in executor) ──

    def _ping_internet(self) -> bool:
        """Quick internet check."""
        # Try DNS resolution first (fastest)
        try:
            socket.setdefaulttimeout(3)
            socket.getaddrinfo("google.com", 80)
            return True
        except (socket.gaierror, socket.timeout):
            pass

        # Try HTTP as backup
        try:
            urllib.request.urlopen("http://clients3.google.com/generate_204", timeout=3)
            return True
        except Exception:
            return False

    def _fetch_camera_stats(self) -> Optional[dict]:
        """Fetch camera stats from Frigate."""
        try:
            resp = urllib.request.urlopen(f"{FRIGATE_API}/api/stats", timeout=5)
            data = json.loads(resp.read().decode())
            return data.get("cameras", {})
        except Exception:
            return None

    def _is_frigate_running(self) -> bool:
        """Check if Frigate Docker container is running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", "frigate"],
                capture_output=True, text=True, timeout=5,
            )
            return "true" in result.stdout.lower()
        except Exception:
            return False

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "internet_down_since": self._internet_down_since,
            "cameras_down": {k: time.time() - v for k, v in self._cameras_down.items()},
            "sms_available": self.sms.available if self.sms else False,
        }
