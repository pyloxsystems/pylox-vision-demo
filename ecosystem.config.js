module.exports = {
  apps: [
    {
      name: "pylox-v2",
      script: "./venv/bin/uvicorn",
      args: "engine.app:app --host 0.0.0.0 --port 3450",
      cwd: "/home/acme-corpai/pylox-v2",
      interpreter: "none",
      env: {
        MQTT_HOST: "localhost",
        MQTT_PORT: "1883",
        FRIGATE_API: "http://localhost:5000",
        V2_PORT: "3450",
        GEMINI_API_KEY: "YOUR_GEMINI_API_KEY",
        GEMINI_MODEL: "gemini-3.1-pro-preview",
      },
      max_restarts: 10,
      restart_delay: 3000,
    },
  ],
};
