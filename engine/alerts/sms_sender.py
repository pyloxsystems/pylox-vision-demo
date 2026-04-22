"""SMS Sender — sends SMS and makes calls via USB cellular dongle (AT commands).

Uses serial communication with a USB GSM/LTE dongle (SIM7600, Huawei E3372, etc.)
to send SMS messages and make automated voice calls without internet.

No Twilio. No API. Direct cellular network access.
"""

import os
import time
import serial
import logging
from typing import Optional

logger = logging.getLogger("pylox-v2.alerts.sms")

# Common USB modem serial paths on Linux
MODEM_PATHS = [
    "/dev/ttyUSB0",
    "/dev/ttyUSB1",
    "/dev/ttyUSB2",
    "/dev/ttyACM0",
    "/dev/ttyACM1",
]


class SMSSender:
    """Sends SMS and makes calls via USB cellular dongle."""

    def __init__(self, port: str = None, baud: int = 115200):
        self.port = port or self._find_modem()
        self.baud = baud
        self._serial: Optional[serial.Serial] = None
        self._available = False
        self._stats = {
            "sms_sent": 0,
            "sms_failed": 0,
            "calls_made": 0,
            "signal_strength": 0,
        }

        if self.port:
            self._connect()

    def _find_modem(self) -> Optional[str]:
        """Auto-detect USB modem serial port."""
        for path in MODEM_PATHS:
            if os.path.exists(path):
                logger.info(f"Found modem at {path}")
                return path
        logger.warning("No USB modem found")
        return None

    def _connect(self):
        """Connect to the modem."""
        try:
            self._serial = serial.Serial(
                self.port, self.baud, timeout=3,
                xonxoff=False, rtscts=False, dsrdtr=False,
            )
            time.sleep(1)

            # Test with AT command
            resp = self._send_at("AT")
            if "OK" in resp:
                self._available = True
                # Set text mode for SMS
                self._send_at("AT+CMGF=1")
                # Check signal
                self._check_signal()
                logger.info(f"Modem connected on {self.port}")
            else:
                logger.error(f"Modem not responding on {self.port}")
                self._available = False

        except Exception as e:
            logger.error(f"Modem connection failed: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        return self._available and self._serial is not None

    def send_sms(self, number: str, message: str) -> bool:
        """Send an SMS message."""
        if not self.available:
            logger.error("Modem not available")
            self._stats["sms_failed"] += 1
            return False

        try:
            # Set text mode
            self._send_at("AT+CMGF=1")
            time.sleep(0.5)

            # Set recipient
            self._send_at(f'AT+CMGS="{number}"', wait_for=">")
            time.sleep(0.5)

            # Send message (Ctrl+Z = chr(26) to send)
            self._serial.write((message + chr(26)).encode())
            time.sleep(3)

            # Read response
            resp = self._serial.read(self._serial.in_waiting or 1).decode(errors="ignore")
            if "+CMGS:" in resp or "OK" in resp:
                self._stats["sms_sent"] += 1
                logger.info(f"SMS sent to {number}: {message[:50]}...")
                return True
            else:
                self._stats["sms_failed"] += 1
                logger.error(f"SMS send failed: {resp}")
                return False

        except Exception as e:
            self._stats["sms_failed"] += 1
            logger.error(f"SMS error: {e}")
            return False

    def make_call(self, number: str, duration_sec: int = 15) -> bool:
        """Make a voice call (for emergency alerts)."""
        if not self.available:
            return False

        try:
            # Dial
            self._send_at(f"ATD{number};")
            self._stats["calls_made"] += 1
            logger.info(f"Calling {number}...")

            # Wait for call duration
            time.sleep(duration_sec)

            # Hang up
            self._send_at("ATH")
            return True

        except Exception as e:
            logger.error(f"Call error: {e}")
            return False

    def _check_signal(self):
        """Check cellular signal strength."""
        try:
            resp = self._send_at("AT+CSQ")
            # Response: +CSQ: 18,0 (18 = signal strength, 0-31 scale)
            if "+CSQ:" in resp:
                parts = resp.split("+CSQ:")[1].strip().split(",")
                strength = int(parts[0])
                # Convert to percentage (31 = max)
                self._stats["signal_strength"] = round(strength / 31 * 100)
        except Exception:
            pass

    def _send_at(self, command: str, wait_for: str = "OK",
                  timeout: float = 3) -> str:
        """Send an AT command and read response."""
        if not self._serial:
            return ""

        self._serial.write((command + "\r\n").encode())
        time.sleep(0.5)

        response = ""
        end_time = time.time() + timeout
        while time.time() < end_time:
            if self._serial.in_waiting:
                response += self._serial.read(self._serial.in_waiting).decode(errors="ignore")
                if wait_for in response:
                    break
            time.sleep(0.1)

        return response

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "available": self.available,
            "port": self.port,
        }
