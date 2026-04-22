"""Tactical Briefing Generator.

Produces a structured JSON tactical brief from incident observations so that
responding officers can walk into the scene with full pre-arrival situational
awareness — suspect count, descriptions, tools visible, direction of travel,
entry point, accomplices, weapons.

This is the #1 officer safety improvement possible: cops knowing exactly what
they're walking into before they arrive.

Output schema (version 1):
{
  "incident_id": int,
  "generated_at": float,
  "site": {"name": str, "address": str, "type": str},
  "camera": {"id": str, "name": str},
  "timeline": {
    "start_edt": str,
    "duration_seconds": int,
    "event_type": str
  },
  "suspects": [
    {
      "id": "p1",
      "description": "Male, ~6ft, dark hoodie, blue jeans, white sneakers",
      "armed": false,
      "weapons": [],
      "tools_visible": ["screwdriver"],
      "direction_of_travel": "northeast",
      "speed": "walking|running|stationary",
      "time_on_scene_seconds": 47,
      "distinguishing_features": ""
    }
  ],
  "entry_point": "east fence at 03:14:08",
  "exit_route_predicted": "Federal Hwy northbound",
  "property_threat_level": 0-10,
  "cameras_involved": ["cam2", "cam3"],
  "last_known_location": str,
  "officer_safety_notes": str,
  "recommended_response": "silent approach|code 3|caution|standard"
}
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger("pylox-v2.police.tactical")


TACTICAL_SYSTEM_PROMPT = """You are generating a tactical briefing for a police officer
responding to an in-progress incident. Extract structured, actionable details from the
AI observation history.

Accuracy is critical — officer safety depends on this briefing. If a detail was not
directly observed, use null or an empty string. Never invent details.

Output STRICT JSON matching this schema exactly — no markdown, no code blocks, no prose:

{
  "suspects": [
    {
      "id": "p1",
      "description": "1-sentence physical description (height, clothing, shoes, headwear)",
      "armed": boolean,
      "weapons": ["list any weapons observed, or empty"],
      "tools_visible": ["list any tools observed, or empty"],
      "direction_of_travel": "north|south|east|west|northeast|etc|unknown",
      "speed": "walking|running|stationary|crouching|unknown",
      "time_on_scene_seconds": integer,
      "distinguishing_features": "tattoos, limp, mask, etc. or empty string"
    }
  ],
  "entry_point": "where and when subject(s) entered, or empty",
  "exit_route_predicted": "best guess direction and route, or empty",
  "property_threat_level": integer 0-10,
  "last_known_location": "where on the property the subject was last seen",
  "officer_safety_notes": "specific hazards officers should know (e.g. 'subject may be armed', 'multiple suspects', 'hiding behind vehicle row')",
  "recommended_response": "silent approach|code 3|caution|standard",
  "accomplices_nearby": integer count of additional people within view but not identified as primary suspects
}

Rules:
- If the subject was never clearly seen, description can be generic ("unknown subject").
- time_on_scene_seconds must be based on the span of observation timestamps.
- property_threat_level should match the max threat in the history.
- recommended_response guide:
  * silent approach: suspect is unaware, active crime in progress, stealth needed
  * code 3: armed suspect, violent crime, immediate danger
  * caution: suspect may be concealed, exit route unclear, multiple unknowns
  * standard: suspect already fled, property needs clearing
- Output ONLY the JSON object. No explanation. No markdown fences."""


def _parse_history_to_context(history: list, incident: dict) -> str:
    lines = []
    if not history:
        return "(no observations)"
    lines.append("OBSERVATIONS (chronological):")
    for h in history:
        t = datetime.fromtimestamp(h.get("timestamp", 0)).strftime("%H:%M:%S")
        threat = h.get("threat", "?")
        typ = h.get("type", "?")
        text = h.get("narration", "")
        action = h.get("action", "none")
        lines.append(f"  [{t}] threat={threat}/10 type={typ} action={action}")
        lines.append(f"           {text}")

    lines.append("")
    lines.append("INCIDENT METADATA:")
    lines.append(f"  max_threat: {incident.get('max_threat', 0)}/10")
    lines.append(f"  duration_seconds: {int((incident.get('end_time') or time.time()) - incident.get('start_time', 0))}")
    lines.append(f"  trigger_source: {incident.get('trigger_source', 'unknown')}")
    lines.append(f"  resolution: {incident.get('resolution', 'active')}")
    return "\n".join(lines)


def generate_brief(
    incident: dict,
    history: list,
    site_config: dict,
    gemini_connector,
    cameras_involved: list = None,
    camera_name: str = None,
) -> Optional[dict]:
    """Generate tactical briefing dict from incident observations.

    Returns the full brief dict (ready to serialize to JSON), or None on failure.
    """
    if not gemini_connector or not getattr(gemini_connector, "available", False):
        logger.warning("Gemini unavailable — cannot generate tactical brief")
        return None

    if not history:
        logger.info(f"Incident {incident.get('id')} has no history — skipping brief")
        return None

    incident_id = incident.get("id")
    camera = incident.get("camera", "unknown")
    cameras = cameras_involved or [camera]
    start_ts = incident.get("start_time", 0)
    end_ts = incident.get("end_time") or time.time()

    site = site_config or {}
    site_info = {
        "name": site.get("name", "Commercial Property"),
        "address": site.get("address", ""),
        "type": site.get("type", "commercial"),
    }

    context = _parse_history_to_context(history, incident)

    try:
        from google.genai import types
        client = gemini_connector._client
        if not client:
            return None

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[TACTICAL_SYSTEM_PROMPT + "\n\n" + context],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
                response_mime_type="application/json",
            ),
        )

        raw_text = (response.text or "").strip() if response else ""
        if not raw_text:
            logger.warning(f"Empty tactical response for incident {incident_id}")
            return None

        # Strip any markdown fences
        if raw_text.startswith("```"):
            lines = [l for l in raw_text.split("\n") if not l.strip().startswith("```")]
            raw_text = "\n".join(lines).strip()

        parsed = json.loads(raw_text)

        # Build the final brief with metadata
        brief = {
            "incident_id": incident_id,
            "generated_at": time.time(),
            "schema_version": 1,
            "site": site_info,
            "camera": {
                "id": camera,
                "name": camera_name or camera,
            },
            "timeline": {
                "start_edt": datetime.fromtimestamp(start_ts).strftime("%H:%M:%S EDT on %B %d, %Y"),
                "end_edt": datetime.fromtimestamp(end_ts).strftime("%H:%M:%S EDT on %B %d, %Y"),
                "duration_seconds": int(end_ts - start_ts),
                "status": incident.get("resolution", "active"),
            },
            "suspects": parsed.get("suspects", []),
            "entry_point": parsed.get("entry_point", ""),
            "exit_route_predicted": parsed.get("exit_route_predicted", ""),
            "property_threat_level": parsed.get("property_threat_level", incident.get("max_threat", 0)),
            "last_known_location": parsed.get("last_known_location", ""),
            "officer_safety_notes": parsed.get("officer_safety_notes", ""),
            "recommended_response": parsed.get("recommended_response", "standard"),
            "accomplices_nearby": parsed.get("accomplices_nearby", 0),
            "cameras_involved": cameras,
        }

        logger.info(
            f"Tactical brief generated for incident {incident_id}: "
            f"{len(brief['suspects'])} suspects, threat={brief['property_threat_level']}/10, "
            f"response={brief['recommended_response']}"
        )
        return brief

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tactical JSON for incident {incident_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to generate tactical brief for incident {incident_id}: {e}")
        return None


def brief_to_summary_string(brief: dict) -> str:
    """One-line summary for logs and SMS alerts."""
    if not brief:
        return ""
    suspects = brief.get("suspects", [])
    n = len(suspects)
    threat = brief.get("property_threat_level", 0)
    response = brief.get("recommended_response", "standard")
    dirs = [s.get("direction_of_travel", "") for s in suspects if s.get("direction_of_travel")]
    dir_str = f" heading {dirs[0]}" if dirs else ""
    return f"{n} subject(s){dir_str}, threat {threat}/10, response: {response}"
