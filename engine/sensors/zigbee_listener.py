"""Zigbee Sensor Listener — listens to Zigbee2MQTT for door/window sensor events.

Subscribes to zigbee2mqtt/# and processes contact sensor events.
When a door opens after hours → feeds into detection handler as context
for Gemini analysis.

Zigbee2MQTT publishes sensor state on:
  zigbee2mqtt/{friendly_name}  →  {"contact": true/false, "battery": 95, ...}

  contact: true = door CLOSED
  contact: false = door OPEN

Sensor naming convention (set in Zigbee2MQTT):
  front_door, back_door, warehouse_door, office_door, etc.
"""

import json
import time
import logging
from typing import Callable, Optional

import paho.mqtt.client as mqtt

from engine.config import MQTT_HOST, MQTT_PORT

logger = logging.getLogger("pylox-v2.sensors.zigbee")

# Map sensor friendly names to camera IDs
SENSOR_TO_CAMERA = {
    "front_door": "cam1",
    "front_entrance": "cam1",
    "back_door": "cam4",
    "backdoor": "cam4",
    "warehouse_door": "cam6",
    "shop_door": "cam5",
    "office_door": "cam7",
    "side_door": "cam3",
    "garage_door": "cam9",
    "driveway_gate": "cam9",
}


class ZigbeeListener:
    """Listens to Zigbee2MQTT door/window sensor events."""

    def __init__(self, on_door_event: Callable = None,
                 on_sensor_update: Callable = None):
        self.on_door_event = on_door_event  # Called when door opens/closes
        self.on_sensor_update = on_sensor_update  # Called for any sensor update
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="pylox-v2-zigbee",
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self._running = False
        self._sensor_states = {}  # sensor_name -> {contact, battery, last_seen}
        self._stats = {
            "messages_received": 0,
            "door_events": 0,
            "sensors_active": 0,
        }

    def start(self):
        """Connect to MQTT and start listening for Zigbee events."""
        logger.info("Connecting to Zigbee2MQTT...")
        try:
            self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            self._running = True
            self.client.loop_start()
        except Exception as e:
            logger.warning(f"Zigbee MQTT connect failed: {e}")

    def stop(self):
        """Disconnect."""
        self._running = False
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Zigbee listener stopped")

    def _on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info("Connected to Zigbee2MQTT")
        client.subscribe("zigbee2mqtt/+")
        client.subscribe("zigbee2mqtt/bridge/+")

    def _on_disconnect(self, client, userdata, flags, rc, properties=None):
        if self._running:
            logger.warning("Zigbee MQTT disconnected, will reconnect...")

    def _on_message(self, client, userdata, msg):
        """Process Zigbee2MQTT messages."""
        self._stats["messages_received"] += 1

        topic = msg.topic

        # Skip bridge messages (status, config, etc.)
        if "/bridge/" in topic:
            return

        # Extract sensor friendly name from topic
        # zigbee2mqtt/{friendly_name} → friendly_name
        parts = topic.split("/")
        if len(parts) < 2:
            return

        sensor_name = parts[1]

        try:
            payload = json.loads(msg.payload.decode())
        except json.JSONDecodeError:
            return

        # Process contact sensors (door/window)
        if "contact" in payload:
            self._handle_contact_sensor(sensor_name, payload)

        # Process any sensor update
        if self.on_sensor_update:
            self.on_sensor_update({
                "sensor": sensor_name,
                "data": payload,
                "timestamp": time.time(),
            })

        # Update sensor state
        self._sensor_states[sensor_name] = {
            "contact": payload.get("contact"),
            "battery": payload.get("battery"),
            "last_seen": time.time(),
            "data": payload,
        }
        self._stats["sensors_active"] = len(self._sensor_states)

    def _handle_contact_sensor(self, sensor_name: str, payload: dict):
        """Handle a contact sensor (door/window) state change."""
        contact = payload.get("contact")  # true = closed, false = open
        state = "closed" if contact else "open"
        battery = payload.get("battery", "?")

        # Check if this is a state CHANGE
        prev = self._sensor_states.get(sensor_name, {})
        prev_contact = prev.get("contact")

        if prev_contact is not None and prev_contact == contact:
            return  # No change

        self._stats["door_events"] += 1
        logger.info(f"Door sensor: {sensor_name} → {state} (battery: {battery}%)")

        # Find associated camera
        camera_id = SENSOR_TO_CAMERA.get(sensor_name)

        # Fire door event callback
        if self.on_door_event:
            self.on_door_event({
                "sensor": sensor_name,
                "state": state,
                "camera": camera_id,
                "battery": battery,
                "timestamp": time.time(),
            })

    def get_door_states(self) -> dict:
        """Get current state of all door sensors."""
        states = {}
        for name, data in self._sensor_states.items():
            if data.get("contact") is not None:
                states[name] = "closed" if data["contact"] else "open"
        return states

    def get_all_sensors(self) -> dict:
        """Get all sensor states."""
        return {
            name: {
                "state": "closed" if data.get("contact") else "open" if data.get("contact") is not None else "unknown",
                "battery": data.get("battery"),
                "last_seen": data.get("last_seen"),
            }
            for name, data in self._sensor_states.items()
        }

    def get_stats(self) -> dict:
        return {**self._stats}
