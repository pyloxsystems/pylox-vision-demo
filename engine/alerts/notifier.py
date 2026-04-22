"""Alert Notifier — sends push notifications and SMS alerts.

Two channels:
  1. Push notifications via Expo push API (app — always)
  2. SMS via USB cellular dongle AT commands (if client opted in)

No Twilio. No WhatsApp. No third-party messaging.
"""

import os
import json
import time
import logging
import urllib.request
from typing import Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("pylox-v2.alerts")

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
SITE_DATA_DIR = Path(__file__).parent.parent.parent / "data"


class AlertNotifier:
    """Handles push notifications and SMS alerts."""

    def __init__(self, sms_sender=None):
        self.sms_sender = sms_sender  # SMSSender instance (when dongle available)
        self._push_tokens = []  # Expo push tokens for this site
        self._sms_numbers = []  # Phone numbers for SMS alerts
        self._stats = {
            "push_sent": 0,
            "push_failed": 0,
            "sms_sent": 0,
            "sms_failed": 0,
            "alerts_total": 0,
        }
        self._load_config()

    def _load_config(self):
        """Load push tokens and SMS numbers from site config."""
        config_file = SITE_DATA_DIR / "alert_config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text())
                self._push_tokens = data.get("push_tokens", [])
                self._sms_numbers = data.get("sms_numbers", [])
                logger.info(f"Loaded {len(self._push_tokens)} push tokens, "
                           f"{len(self._sms_numbers)} SMS numbers")
            except Exception as e:
                logger.warning(f"Failed to load alert config: {e}")

    def save_config(self):
        """Save current config to disk."""
        config_file = SITE_DATA_DIR / "alert_config.json"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps({
            "push_tokens": self._push_tokens,
            "sms_numbers": self._sms_numbers,
        }, indent=2))

    def register_push_token(self, token: str):
        """Register an Expo push token (called from app on login)."""
        if token not in self._push_tokens:
            self._push_tokens.append(token)
            self.save_config()
            logger.info(f"Registered push token: {token[:20]}...")

    def register_sms_number(self, number: str):
        """Register a phone number for SMS alerts."""
        # Normalize
        number = number.strip().replace(" ", "").replace("-", "")
        if not number.startswith("+"):
            number = "+1" + number  # Default US
        if number not in self._sms_numbers:
            self._sms_numbers.append(number)
            self.save_config()
            logger.info(f"Registered SMS number: {number}")

    def remove_sms_number(self, number: str):
        """Remove a phone number from SMS alerts."""
        number = number.strip().replace(" ", "").replace("-", "")
        if not number.startswith("+"):
            number = "+1" + number
        self._sms_numbers = [n for n in self._sms_numbers if n != number]
        self.save_config()

    def _send_sms_fallback(self, text: str):
        """SMS fallback when push fails."""
        if not self._sms_numbers or not self.sms_sender:
            return
        for number in self._sms_numbers:
            try:
                self.sms_sender.send_sms(number, text)
                self._stats["sms_sent"] += 1
                logger.info(f"SMS fallback sent to {number}")
            except Exception as e:
                self._stats["sms_failed"] += 1
                logger.error(f"SMS fallback failed: {e}")

    def send_alert(self, alert: dict):
        """Send alert via all configured channels.

        alert format:
        {
            "camera": "cam4",
            "camera_name": "Back Door",
            "threat": 8,
            "type": "suspicious",
            "description": "Man in dark hoodie tried back door handle...",
            "action": "alert_owner",
            "timestamp": 1234567890,
        }
        """
        self._stats["alerts_total"] += 1

        camera_name = alert.get("camera_name", alert.get("camera", "Unknown"))
        threat = alert.get("threat", 0)
        description = alert.get("description", "Detection event")
        timestamp = alert.get("timestamp", time.time())
        time_str = datetime.fromtimestamp(timestamp).strftime("%I:%M %p")

        # Push notification first
        push_title = f"Threat {threat}/10 — {camera_name}"
        push_body = description
        push_succeeded = self._send_push(push_title, push_body, alert)

        # SMS only as fallback if push fails AND we have numbers
        if not push_succeeded and self._sms_numbers and self.sms_sender:
            sms_text = (
                f"PYLOX: {camera_name} {time_str}\n"
                f"{description}\n"
                f"Threat: {threat}/10"
            )
            self._send_sms_fallback(sms_text)

    def send_emergency(self, message: str):
        """Send emergency alert (system failure, tamper, etc.)."""
        self._send_push("PYLOX EMERGENCY", message, {"emergency": True})
        if self._sms_numbers:
            self._send_sms(f"PYLOX EMERGENCY: {message}")

    def send_morning_summary(self, summary: str):
        """Send morning summary as a push notification."""
        self._send_push("Pylox — Overnight Summary", summary, {"type": "summary"})

    def _send_push(self, title: str, body: str, data: dict = None) -> bool:
        """Send push notification via Expo push API. Returns True if at least one succeeded."""
        if not self._push_tokens:
            return False

        succeeded = False
        for token in self._push_tokens:
            try:
                payload = json.dumps({
                    "to": token,
                    "title": title,
                    "body": body[:200],
                    "sound": "default",
                    "priority": "high",
                    "data": data or {},
                }).encode()

                req = urllib.request.Request(
                    EXPO_PUSH_URL,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                resp = urllib.request.urlopen(req, timeout=5)
                if resp.status == 200:
                    self._stats["push_sent"] += 1
                    succeeded = True
                    logger.info(f"Push sent: {title}")
                else:
                    self._stats["push_failed"] += 1

            except Exception as e:
                self._stats["push_failed"] += 1
                logger.error(f"Push notification failed: {e}")

        return succeeded

    def _send_sms(self, text: str):
        """Send SMS via USB cellular dongle."""
        if not self.sms_sender:
            logger.warning("SMS sender not available (no cellular dongle)")
            return

        for number in self._sms_numbers:
            try:
                self.sms_sender.send_sms(number, text)
                self._stats["sms_sent"] += 1
                logger.info(f"SMS sent to {number}")
            except Exception as e:
                self._stats["sms_failed"] += 1
                logger.error(f"SMS failed to {number}: {e}")

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "push_tokens": len(self._push_tokens),
            "sms_numbers": len(self._sms_numbers),
            "sms_available": self.sms_sender is not None,
        }
