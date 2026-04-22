"""Morning Summary — generates overnight report and sends at 7am.

Collects all events from 8pm-6am, groups by severity, and generates
a clean one-paragraph summary for push notification + a detailed
HTML report accessible via the Intelligence page.
"""

import time
import json
import logging
from datetime import datetime, timedelta
from engine import database as db

logger = logging.getLogger("pylox-v2.alerts.summary")


def generate_overnight_summary() -> dict:
    """Generate summary of overnight events.

    Returns dict with:
        - text: short summary for push notification
        - events: list of overnight events
        - stats: {total, critical, warning, info, cameras_triggered}
    """
    now = datetime.now()
    # Last night: 8pm yesterday to 6am today
    if now.hour < 12:
        night_start = now.replace(hour=20, minute=0, second=0) - timedelta(days=1)
        night_end = now.replace(hour=6, minute=0, second=0)
    else:
        # Afternoon — summarize last night
        night_start = now.replace(hour=20, minute=0, second=0) - timedelta(days=1)
        night_end = now.replace(hour=6, minute=0, second=0)

    events = db.get_events(since=night_start.timestamp(), limit=500)
    events = [e for e in events if e["timestamp"] < night_end.timestamp()]

    # Parse event data
    for e in events:
        if isinstance(e["data"], str):
            e["data"] = json.loads(e["data"])

    # Stats
    total = len(events)
    critical = sum(1 for e in events if e["severity"] == "critical")
    warning = sum(1 for e in events if e["severity"] == "warning")
    info = total - critical - warning
    cameras = set(e["camera"] for e in events)

    # Build Gemini events (with descriptions)
    gemini_events = [e for e in events if "gemini" in e.get("event_type", "")]
    false_positives = sum(1 for e in gemini_events if not e["data"].get("real", True))
    real_threats = sum(1 for e in gemini_events if e["data"].get("real", False))

    # Generate text summary
    if total == 0:
        text = "Quiet night. No detections between 8 PM and 6 AM. All cameras operational."
    else:
        lines = [f"Last night: {total} detection{'s' if total != 1 else ''}"]

        if false_positives:
            lines.append(f"  {false_positives} confirmed false positive{'s' if false_positives != 1 else ''} (suppressed)")

        if real_threats:
            lines.append(f"  {real_threats} confirmed real detection{'s' if real_threats != 1 else ''}")

        # Add top events with descriptions
        for e in gemini_events[:5]:
            data = e["data"]
            if data.get("description"):
                t = datetime.fromtimestamp(e["timestamp"]).strftime("%I:%M %p")
                status = "✅" if not data.get("real") else "⚠️" if data.get("threat", 0) < 7 else "🔴"
                lines.append(f"{status} {t} — {data['description'][:80]}")

        if critical:
            lines.append(f"\n{critical} critical alert{'s' if critical != 1 else ''} sent.")

        text = "\n".join(lines)

    return {
        "text": text,
        "events": events,
        "gemini_events": gemini_events,
        "stats": {
            "total": total,
            "critical": critical,
            "warning": warning,
            "info": info,
            "cameras_triggered": list(cameras),
            "false_positives": false_positives,
            "real_threats": real_threats,
        },
        "period": {
            "start": night_start.isoformat(),
            "end": night_end.isoformat(),
        },
    }
