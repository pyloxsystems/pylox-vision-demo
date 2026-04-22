"""Training Manager — manages per-site learning files.

Each client site has one training.json file that grows smarter over time.
The file is injected into every Gemini prompt as context.

Structure:
{
    "site": {
        "id": "bobs-auto",
        "name": "Bob's Auto Dealership",
        "type": "car dealership",
        "hours": {"open": 7, "close": 20},
        "address": "123 Main St, Pompano Beach, FL"
    },
    "cameras": {
        "cam1": {
            "name": "Front Entrance",
            "false_positives": [
                {"description": "Tree shadow at 4pm looks like person", "added": "2026-04-05", "count": 3}
            ],
            "known_people": [
                {"description": "Owner wears red hat, arrives 6:30am", "added": "2026-04-05"}
            ]
        }
    },
    "learned": [
        {"pattern": "FedEx delivers Mon/Wed/Fri around 2pm at front door", "added": "2026-04-05"},
    ],
    "stats": {
        "total_feedback": 0,
        "false_positives_reported": 0,
        "correct_alerts": 0
    }
}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("pylox-v2.training")

TRAINING_DIR = Path(__file__).parent.parent.parent / "data" / "training"


class TrainingManager:
    """Manages per-site training files."""

    def __init__(self, site_id: str = "default"):
        self.site_id = site_id
        self._file = TRAINING_DIR / site_id / "training.json"
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                return json.loads(self._file.read_text())
            except Exception:
                pass
        return {
            "site": {"id": self.site_id, "name": "", "type": "", "hours": {"open": 6, "close": 20}},
            "cameras": {},
            "learned": [],
            "stats": {"total_feedback": 0, "false_positives_reported": 0, "correct_alerts": 0},
        }

    def _save(self):
        self._file.write_text(json.dumps(self._data, indent=2))

    def set_site_info(self, name: str = None, site_type: str = None,
                       hours: dict = None, address: str = None):
        """Set site information."""
        site = self._data.setdefault("site", {"id": self.site_id})
        if name:
            site["name"] = name
        if site_type:
            site["type"] = site_type
        if hours:
            site["hours"] = hours
        if address:
            site["address"] = address
        self._save()

    def add_false_positive(self, camera_id: str, description: str):
        """Add a false positive for a camera."""
        cam = self._data.setdefault("cameras", {}).setdefault(camera_id, {})
        fps = cam.setdefault("false_positives", [])

        # Check if similar already exists
        for fp in fps:
            if fp["description"].lower() == description.lower():
                fp["count"] = fp.get("count", 1) + 1
                stats = self._data.setdefault("stats", {"total_feedback": 0, "false_positives_reported": 0, "correct_alerts": 0})
                stats["false_positives_reported"] = stats.get("false_positives_reported", 0) + 1
                stats["total_feedback"] = stats.get("total_feedback", 0) + 1
                self._save()
                return

        fps.append({
            "description": description,
            "added": datetime.now().strftime("%Y-%m-%d"),
            "count": 1,
        })

        # Cap at 10 per camera — drop least frequent
        if len(fps) > 10:
            fps.sort(key=lambda x: x.get("count", 1), reverse=True)
            fps.pop()

        stats = self._data.setdefault("stats", {"total_feedback": 0, "false_positives_reported": 0, "correct_alerts": 0})
        stats["false_positives_reported"] = stats.get("false_positives_reported", 0) + 1
        stats["total_feedback"] = stats.get("total_feedback", 0) + 1
        self._save()
        logger.info(f"False positive added for {camera_id}: {description}")

    def add_known_person(self, camera_id: str, description: str):
        """Add a known person for a camera."""
        cam = self._data.setdefault("cameras", {}).setdefault(camera_id, {})
        known = cam.setdefault("known_people", [])

        if any(k["description"].lower() == description.lower() for k in known):
            return

        known.append({
            "description": description,
            "added": datetime.now().strftime("%Y-%m-%d"),
        })

        if len(known) > 5:
            known.pop(0)

        self._save()

    def add_pattern(self, pattern: str):
        """Add a site-wide learned pattern."""
        learned = self._data.setdefault("learned", [])

        if any(l["pattern"].lower() == pattern.lower() for l in learned):
            return

        learned.append({
            "pattern": pattern,
            "added": datetime.now().strftime("%Y-%m-%d"),
        })

        if len(learned) > 20:
            learned.pop(0)

        self._save()

    def record_correct_alert(self):
        """Record that a client confirmed an alert was correct."""
        stats = self._data.setdefault("stats", {"total_feedback": 0, "false_positives_reported": 0, "correct_alerts": 0})
        stats["correct_alerts"] = stats.get("correct_alerts", 0) + 1
        stats["total_feedback"] = stats.get("total_feedback", 0) + 1
        self._save()

    def process_feedback(self, event_id: int, feedback: str,
                          camera_id: str = None, gemini_description: str = None):
        """Process client feedback on an alert.

        feedback: "false_alarm" | "correct" | "known_person"
        """
        if feedback == "false_alarm":
            if camera_id and gemini_description:
                self.add_false_positive(camera_id, gemini_description)
            elif camera_id:
                self.add_false_positive(camera_id, f"Event #{event_id} marked as false alarm")

        elif feedback == "correct":
            self.record_correct_alert()

        elif feedback == "known_person":
            if camera_id and gemini_description:
                self.add_known_person(camera_id, gemini_description)

        logger.info(f"Feedback processed: event={event_id} feedback={feedback} camera={camera_id}")

    def get_prompt_context(self, camera_id: str = None) -> str:
        """Get the training context string for Gemini prompts."""
        lines = []

        # Camera-specific context
        if camera_id:
            cam = self._data.get("cameras", {}).get(camera_id, {})

            fps = cam.get("false_positives", [])
            if fps:
                lines.append("KNOWN FALSE POSITIVES for this camera:")
                for fp in fps:
                    count = fp.get("count", 1)
                    lines.append(f"  - {fp['description']}" +
                               (f" (reported {count}x)" if count > 1 else ""))

            known = cam.get("known_people", [])
            if known:
                lines.append("KNOWN PEOPLE for this camera:")
                for k in known:
                    lines.append(f"  - {k['description']}")

        # Site-wide patterns
        learned = self._data.get("learned", [])
        if learned:
            lines.append("LEARNED PATTERNS for this site:")
            for l in learned:
                lines.append(f"  - {l['pattern']}")

        return "\n".join(lines) if lines else ""

    def get_data(self) -> dict:
        """Get full training data (for API)."""
        return self._data

    def get_stats(self) -> dict:
        """Get training stats."""
        total_fps = sum(
            len(cam.get("false_positives", []))
            for cam in self._data.get("cameras", {}).values()
        )
        return {
            **self._data.get("stats", {}),
            "total_false_positives": total_fps,
            "total_patterns": len(self._data.get("learned", [])),
            "cameras_with_data": list(self._data.get("cameras", {}).keys()),
        }
