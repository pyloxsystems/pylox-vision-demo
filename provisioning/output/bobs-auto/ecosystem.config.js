module.exports = {
  apps: [
    {
      name: "pylox-v2",
      script: "./venv/bin/uvicorn",
      args: "engine.app:app --host 0.0.0.0 --port 3450",
      cwd: "/opt/pylox-v2",
      interpreter: "none",
      env: {
        MQTT_HOST: "localhost",
        MQTT_PORT: "1883",
        FRIGATE_API: "http://localhost:5000",
        V2_PORT: "3450",
        GEMINI_API_KEY: process.env.GEMINI_API_KEY || "",
        SITE_ID: "bobs-auto",
      },
      max_restarts: 10,
      restart_delay: 3000,
    },
    {
      name: "pylox-vision",
      script: "/opt/pylox-vision/web/serve.cjs",
      cwd: "/opt/pylox-vision/web",
      env: {
        NODE_ENV: "production",
      },
      max_restarts: 10,
    },
  ],
};
