#!/bin/bash
# Pylox Vision Installer — ACME-CORP Warehouse
# Generated: 2026-04-06 23:55
# Run: sudo bash install.sh

set -e

echo ""
echo "  PYLOX VISION INSTALLER"
echo "  Site: ACME-CORP Warehouse"
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
