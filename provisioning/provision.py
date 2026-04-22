#!/usr/bin/env python3
"""Pylox Vision Provisioner — one JSON config → fully deployed Jetson.

Usage:
  python provision.py client_config.json

Generates:
  1. Frigate config (frigate-config.yml)
  2. Docker compose (docker-compose.yml)
  3. V2 engine config (ecosystem.config.js)
  4. Zigbee2MQTT config (zigbee2mqtt/configuration.yaml)
  5. Training file (training/training.json)
  6. Alert config (alert_config.json)
  7. Sensor-to-camera mapping
  8. Install script (install.sh) — one command to deploy everything

Run install.sh on a fresh Jetson → system is live in 10 minutes.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

TEMPLATE_DIR = Path(__file__).parent / "templates"
OUTPUT_DIR = Path(__file__).parent / "output"


def provision(config_path: str):
    """Generate all deployment files from a client config."""
    with open(config_path) as f:
        config = json.load(f)

    site = config["site"]
    site_id = site["id"]
    output = OUTPUT_DIR / site_id
    output.mkdir(parents=True, exist_ok=True)

    print(f"\n  PYLOX VISION PROVISIONER")
    print(f"  Site: {site['name']} ({site_id})")
    print(f"  Cameras: {len(config['cameras'])}")
    print(f"  Hours: {site['hours']['open']}:00 - {site['hours']['close']}:00")
    print()

    # 1. Generate Frigate config
    generate_frigate_config(config, output)

    # 2. Generate Docker compose
    generate_docker_compose(config, output)

    # 3. Generate V2 ecosystem config
    generate_ecosystem(config, output)

    # 4. Generate Zigbee2MQTT config
    generate_zigbee_config(config, output)

    # 5. Generate training file
    generate_training_file(config, output)

    # 6. Generate alert config
    generate_alert_config(config, output)

    # 7. Generate sensor mapping
    generate_sensor_mapping(config, output)

    # 8. Generate install script
    generate_install_script(config, output)

    print(f"\n  All files generated in: {output}")
    print(f"  Copy to Jetson and run: sudo bash install.sh")
    print()


def generate_frigate_config(config: dict, output: Path):
    """Generate frigate-config.yml from camera definitions."""
    site = config["site"]
    cameras = config["cameras"]

    lines = [
        "version: 0.16-0",
        "",
        "mqtt:",
        "  enabled: true",
        "  host: localhost",
        "  port: 1883",
        "",
        "detectors:",
        "  cpu:",
        "    type: cpu",
        "    num_threads: 4",
        "",
        "detect:",
        "  enabled: false",
        "",
        "genai:",
        "  enabled: false",
        "",
        "auth:",
        "  enabled: false",
        "",
        "go2rtc:",
        "  streams:",
    ]

    # Add go2rtc streams
    for cam_id, cam in cameras.items():
        lines.append(f"    {cam_id}:")
        lines.append(f"      {cam['rtsp']}")

    lines.append("")
    lines.append("cameras:")

    # Add camera configs
    for cam_id, cam in cameras.items():
        detect = cam.get("detect", {})
        track = cam.get("track", ["person"])
        zones = cam.get("zones", {})
        detect_enabled = detect.get("enabled", True)
        width = detect.get("width", 640)
        height = detect.get("height", 480)
        fps = detect.get("fps", 3)

        lines.extend([
            f"  {cam_id}:",
            f"    enabled: true",
            f"    ffmpeg:",
            f"      inputs:",
            f"        - path: {cam['rtsp']}",
            f"          input_args: preset-rtsp-generic",
            f"          roles:",
        ])

        if detect_enabled:
            lines.append(f"            - detect")
        lines.append(f"            - record")

        lines.extend([
            f"    detect:",
            f"      enabled: {'true' if detect_enabled else 'false'}",
            f"      width: {width}",
            f"      height: {height}",
            f"      fps: {fps}",
            f"    objects:",
            f"      track:",
        ])

        for obj in track:
            lines.append(f"        - {obj}")

        lines.extend([
            f"      filters:",
            f"        person:",
            f"          min_score: 0.4",
            f"          threshold: 0.5",
            f"          min_area: 500",
            f"    snapshots:",
            f"      enabled: true",
            f"      retain:",
            f"        default: 3",
            f"    record:",
            f"      enabled: true",
            f"      retain:",
            f"        days: 7",
        ])

        # Add zones if defined
        if zones:
            lines.append(f"    zones:")
            for zone_name, zone_config in zones.items():
                lines.append(f"      {zone_name}:")
                lines.append(f"        coordinates: {zone_config['coordinates']}")
                if "loitering_time" in zone_config:
                    lines.append(f"        loitering_time: {zone_config['loitering_time']}")
                lines.append(f"        inertia: 4")

        lines.append("")

    (output / "frigate-config.yml").write_text("\n".join(lines))
    print(f"  ✓ frigate-config.yml ({len(cameras)} cameras)")


def generate_docker_compose(config: dict, output: Path):
    """Generate docker-compose.yml."""
    network = config.get("network", {})
    zigbee_port = network.get("zigbee_dongle", "/dev/ttyACM0")

    compose = f"""services:
  frigate:
    image: ghcr.io/blakeblackshear/frigate:stable
    container_name: frigate
    privileged: true
    network_mode: host
    shm_size: "256mb"
    volumes:
      - ./frigate-config.yml:/config/config.yml
      - ./frigate-storage:/media/frigate
      - /etc/localtime:/etc/localtime:ro
    restart: unless-stopped

  zigbee2mqtt:
    image: koenkk/zigbee2mqtt:latest
    container_name: pylox-zigbee
    network_mode: host
    volumes:
      - ./zigbee2mqtt:/app/data
      - /run/udev:/run/udev:ro
    devices:
      - {zigbee_port}:{zigbee_port}
    environment:
      - TZ={config['site'].get('timezone', 'America/New_York')}
    restart: unless-stopped

  mosquitto:
    image: eclipse-mosquitto:2
    container_name: mosquitto
    network_mode: host
    volumes:
      - ./mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf
    restart: unless-stopped
"""

    (output / "docker-compose.yml").write_text(compose)

    # Mosquitto config
    mosquitto_dir = output / "mosquitto"
    mosquitto_dir.mkdir(exist_ok=True)
    (mosquitto_dir / "mosquitto.conf").write_text(
        "listener 1883\nallow_anonymous true\npersistence false\n"
    )

    print(f"  ✓ docker-compose.yml (frigate + zigbee2mqtt + mosquitto)")


def generate_ecosystem(config: dict, output: Path):
    """Generate PM2 ecosystem config for V2 engine."""
    site = config["site"]
    network = config.get("network", {})

    eco = f"""module.exports = {{
  apps: [
    {{
      name: "pylox-v2",
      script: "./venv/bin/uvicorn",
      args: "engine.app:app --host 0.0.0.0 --port 3450",
      cwd: "/opt/pylox-v2",
      interpreter: "none",
      env: {{
        MQTT_HOST: "localhost",
        MQTT_PORT: "1883",
        FRIGATE_API: "http://localhost:5000",
        V2_PORT: "3450",
        GEMINI_API_KEY: process.env.GEMINI_API_KEY || "",
        SITE_ID: "{site['id']}",
      }},
      max_restarts: 10,
      restart_delay: 3000,
    }},
    {{
      name: "pylox-vision",
      script: "/opt/pylox-vision/web/serve.cjs",
      cwd: "/opt/pylox-vision/web",
      env: {{
        NODE_ENV: "production",
      }},
      max_restarts: 10,
    }},
  ],
}};
"""

    (output / "ecosystem.config.js").write_text(eco)
    print(f"  ✓ ecosystem.config.js (V2 engine + vision UI)")


def generate_zigbee_config(config: dict, output: Path):
    """Generate Zigbee2MQTT configuration."""
    network = config.get("network", {})
    zigbee_port = network.get("zigbee_dongle", "/dev/ttyACM0")

    z2m_dir = output / "zigbee2mqtt"
    z2m_dir.mkdir(exist_ok=True)

    z2m_config = f"""homeassistant: false
permit_join: true
mqtt:
  base_topic: zigbee2mqtt
  server: mqtt://localhost:1883
serial:
  port: {zigbee_port}
frontend:
  port: 8099
advanced:
  log_level: info
  network_key: GENERATE
"""

    (z2m_dir / "configuration.yaml").write_text(z2m_config)
    print(f"  ✓ zigbee2mqtt/configuration.yaml")


def generate_training_file(config: dict, output: Path):
    """Generate initial training file for the site."""
    site = config["site"]
    cameras = config["cameras"]

    training = {
        "site": {
            "id": site["id"],
            "name": site["name"],
            "type": site["type"],
            "hours": site["hours"],
            "address": site.get("address", ""),
        },
        "cameras": {
            cam_id: {
                "name": cam["name"],
                "false_positives": [],
                "known_people": [],
            }
            for cam_id, cam in cameras.items()
        },
        "learned": [],
        "stats": {
            "total_feedback": 0,
            "false_positives_reported": 0,
            "correct_alerts": 0,
        },
    }

    training_dir = output / "training" / site["id"]
    training_dir.mkdir(parents=True, exist_ok=True)
    (training_dir / "training.json").write_text(json.dumps(training, indent=2))
    print(f"  ✓ training/{site['id']}/training.json")


def generate_alert_config(config: dict, output: Path):
    """Generate alert configuration."""
    alerts = config.get("alerts", {})

    alert_config = {
        "push_tokens": alerts.get("push_tokens", []),
        "sms_numbers": alerts.get("sms_numbers", []),
    }

    (output / "alert_config.json").write_text(json.dumps(alert_config, indent=2))
    print(f"  ✓ alert_config.json ({len(alert_config['sms_numbers'])} SMS numbers)")


def generate_sensor_mapping(config: dict, output: Path):
    """Generate sensor-to-camera mapping."""
    sensors = config.get("sensors", {})
    (output / "sensor_mapping.json").write_text(json.dumps(sensors, indent=2))
    print(f"  ✓ sensor_mapping.json ({len(sensors)} sensors)")


def generate_install_script(config: dict, output: Path):
    """Generate the one-command install script for a fresh Jetson."""
    site = config["site"]

    script = f"""#!/bin/bash
# Pylox Vision Installer — {site['name']}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
# Run: sudo bash install.sh

set -e

echo ""
echo "  PYLOX VISION INSTALLER"
echo "  Site: {site['name']}"
echo ""

# 1. System packages
echo "[1/8] Installing system packages..."
apt-get update -qq
apt-get install -y -qq docker.io docker-compose nodejs npm mosquitto python3-venv python3-pip > /dev/null 2>&1

# 2. Create directories
echo "[2/8] Creating directories..."
mkdir -p /opt/pylox-v2 /opt/pylox-vision /opt/pylox-v2/data /opt/pylox-v2/models

# 3. Copy files
echo "[3/8] Copying configuration..."
cp frigate-config.yml /opt/pylox-v2/
cp docker-compose.yml /opt/pylox-v2/
cp ecosystem.config.js /opt/pylox-v2/
cp alert_config.json /opt/pylox-v2/data/
cp sensor_mapping.json /opt/pylox-v2/data/
cp -r zigbee2mqtt /opt/pylox-v2/
cp -r mosquitto /opt/pylox-v2/
cp -r training /opt/pylox-v2/data/

# 4. Start Docker containers
echo "[4/8] Starting Docker containers..."
cd /opt/pylox-v2
docker-compose up -d

# 5. Set up Python environment
echo "[5/8] Setting up Python environment..."
cd /opt/pylox-v2
python3 -m venv venv
source venv/bin/activate
pip install -q fastapi uvicorn paho-mqtt aiosqlite websockets numpy pillow mediapipe google-genai pyserial requests

# 6. Install PM2
echo "[6/8] Installing PM2..."
npm install -g pm2 > /dev/null 2>&1

# 7. Start services
echo "[7/8] Starting Pylox services..."
pm2 start ecosystem.config.js
pm2 save
pm2 startup systemd -u $USER --hp /home/$USER > /dev/null 2>&1

# 8. Install Tailscale
echo "[8/8] Setting up Tailscale..."
curl -fsSL https://tailscale.com/install.sh | sh
echo "Run 'tailscale up --auth-key=YOUR_KEY' to connect to VPN"

echo ""
echo "  ✅ PYLOX VISION INSTALLED"
echo ""
echo "  Frigate: http://localhost:5000"
echo "  Pylox Vision: http://localhost:3335"
echo "  V2 Engine: http://localhost:3450"
echo "  Zigbee2MQTT: http://localhost:8099"
echo ""
echo "  Next steps:"
echo "  1. Connect Tailscale: tailscale up --auth-key=YOUR_KEY"
echo "  2. Pair door sensors via http://localhost:8099"
echo "  3. System will start learning automatically"
echo ""
"""

    install_path = output / "install.sh"
    install_path.write_text(script)
    install_path.chmod(0o755)
    print(f"  ✓ install.sh (one-command installer)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python provision.py client_config.json")
        sys.exit(1)

    provision(sys.argv[1])
