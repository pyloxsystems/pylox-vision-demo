"""Pylox V2 Engine Configuration"""

import os

# MQTT
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Frigate topics
FRIGATE_EVENTS_TOPIC = "frigate/events"
FRIGATE_DETECTIONS_TOPIC = "frigate/+/person"  # per-camera person detections

# Cameras (from frigate-config.yml)
CAMERAS = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8", "cam9"]

# Behavior thresholds
LOITER_TIME_SECONDS = 30        # person in same area for 30s = loitering
LOITER_RADIUS_PIXELS = 80       # pixel radius for "same area"
SPEED_FAST_THRESHOLD = 150      # pixels/sec = running
SPEED_SLOW_THRESHOLD = 5        # pixels/sec = stationary

# Zone dwell thresholds
ZONE_DWELL_WARNING = 60         # 60s in a zone = warning
ZONE_DWELL_ALERT = 180          # 3min in a zone = alert

# Event engine
EVENT_COOLDOWN_SECONDS = 30     # don't re-fire same event type within 30s

# API
API_HOST = "0.0.0.0"
API_PORT = int(os.getenv("V2_PORT", "3450"))

# Data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "v2.db")

# Frigate API (for snapshots)
FRIGATE_API = os.getenv("FRIGATE_API", "http://localhost:5000")

# GPU queue
MAX_GPU_CONCURRENT = 2  # max concurrent GPU tasks
