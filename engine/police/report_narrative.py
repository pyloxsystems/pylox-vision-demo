"""Report Narrative Generator.

Generates a prosecutor-ready neutral-witness paragraph describing an incident
in the exact voice used in Florida police reports. Officers copy this text
directly into their incident reports, saving ~15 minutes of writing per report.

Flow:
    WatchSession._end() →
        generate_narrative(incident_data, site_config) →
            Gemini prompt with history + site details →
                returns narrative string →
                    stored in incident_police.report_narrative

The narrative is written as though the author is an uninvolved technical
witness — not first person, no editorializing, just the observed facts in
the order they occurred. Usable verbatim in a Florida incident report.
"""

import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("pylox-v2.police.narrative")


NARRATIVE_SYSTEM_PROMPT = """You are generating a paragraph for a Florida police incident report.

Write in the neutral voice of a technical witness describing recorded events.
Use the past tense. Do NOT use first person. Do NOT editorialize or speculate.
Report only observable facts — clothing, actions, direction of travel, tools,
objects touched, duration. Use 24-hour military time with EST/EDT timezone.
Include precise timestamps and camera identifiers.

Follow this structure strictly:

  1. Opening: "At approximately [HH:MM:SS EDT] on [Date], the Pylox Vision
     AI security system at [Site Name] ([Address]) detected [event type]
     at [Camera Location]."

  2. Subject description: physical appearance (height, clothing, accessories,
     distinguishing features), in the order they appear in the footage.

  3. Actions taken: chronological sequence of what the subject did, with
     timestamps for each phase. Name the cameras that captured each phase.

  4. Tools/weapons visible (if any): describe without assumption of intent.

  5. Resolution: how and when the subject left, direction of travel, last
     timestamp on camera.

  6. Cameras involved: list all camera IDs that recorded the incident.

Do NOT include:
  - First person ("I saw")
  - Opinion or conclusion ("clearly a burglar")
  - Speculation about intent
  - Identification by name (unless provided in signals)
  - Any language that is not strictly factual

The paragraph should read like this example:

"At approximately 03:14:22 EDT on April 11, 2026, the Pylox Vision AI security
system at Ocean Motors (1234 Federal Highway, Pompano Beach, FL) detected
unauthorized activity in the front parking lot. A male subject, approximately
six feet in height, wearing a dark hooded sweatshirt, denim jeans, and white
athletic shoes, entered the property by scaling the east perimeter fence at
03:14:08 EDT. The subject approached a white 2024 BMW 530i parked in the
customer display area at 03:14:31 EDT. The subject produced a hand tool and
attempted to pry the driver-side door for a duration of 2 minutes 47 seconds.
The subject fled northbound on Federal Highway at 03:17:33 EDT. All events
were recorded on camera 2, camera 3, and camera 5."

Output ONLY the paragraph. No preamble. No markdown. No bullet points. No quotes."""


def _format_history(history: list) -> str:
    if not history:
        return "(no narration history available)"
    lines = []
    for h in history:
        t = datetime.fromtimestamp(h.get("timestamp", 0)).strftime("%H:%M:%S")
        threat = h.get("threat", "?")
        text = h.get("narration", "")
        lines.append(f"  [{t}] threat={threat}/10: {text}")
    return "\n".join(lines)


def _format_site(site_config: dict) -> str:
    if not site_config:
        return "a commercial property"
    name = site_config.get("name") or "Commercial Property"
    addr = site_config.get("address") or "address on file"
    stype = site_config.get("type") or "commercial site"
    return f"{name} ({addr}) — {stype}"


def generate_narrative(
    incident: dict,
    history: list,
    site_config: dict,
    gemini_connector,
    camera_name: str = None,
) -> Optional[str]:
    """Generate the prosecutor-ready narrative paragraph for an incident.

    Args:
        incident: dict from database with id, camera, start_time, end_time,
                  trigger_source, max_threat, resolution.
        history: list of narration dicts from WatchSession (timestamp, threat,
                 narration, type, action).
        site_config: site information (name, address, type, hours).
        gemini_connector: the shared GeminiConnector instance from app.py.
        camera_name: human-friendly camera label ("Front Entrance", etc.).

    Returns:
        Narrative paragraph string, or None if generation failed.
    """
    if not gemini_connector or not getattr(gemini_connector, "available", False):
        logger.warning("Gemini unavailable — cannot generate report narrative")
        return None

    if not history:
        logger.info(f"Incident {incident.get('id')} has no history — skipping narrative")
        return None

    incident_id = incident.get("id")
    camera = incident.get("camera", "unknown")
    camera_label = camera_name or camera
    start_ts = incident.get("start_time", 0)
    end_ts = incident.get("end_time") or time.time()
    duration = end_ts - start_ts
    trigger = incident.get("trigger_source", "unknown")
    max_threat = incident.get("max_threat", 0)
    resolution = incident.get("resolution", "unknown")

    start_dt = datetime.fromtimestamp(start_ts)
    end_dt = datetime.fromtimestamp(end_ts)

    context = f"""INCIDENT DATA FOR REPORT NARRATIVE

Site: {_format_site(site_config)}
Camera: {camera_label} ({camera})
Incident ID: {incident_id}
Start: {start_dt.strftime('%H:%M:%S EDT on %B %d, %Y')}
End: {end_dt.strftime('%H:%M:%S EDT on %B %d, %Y')}
Duration: {duration:.0f} seconds
Trigger: {trigger}
Max threat level: {max_threat}/10
Resolution: {resolution}

CHRONOLOGICAL OBSERVATIONS FROM AI WATCH SESSION:
{_format_history(history)}

Using the above observations, write the incident paragraph following the
exact format specified in the system prompt. Output ONLY the paragraph."""

    try:
        from google.genai import types
        client = gemini_connector._client
        if not client:
            return None

        full_prompt = NARRATIVE_SYSTEM_PROMPT + "\n\n" + context

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[full_prompt],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )

        text = (response.text or "").strip() if response else None
        if not text:
            logger.warning(f"Gemini returned empty narrative for incident {incident_id}")
            return None

        # Clean up formatting artifacts
        text = text.strip().strip('"').strip("'").strip()
        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        logger.info(
            f"Narrative generated for incident {incident_id} "
            f"({len(text)} chars, {len(history)} observations)"
        )
        return text

    except Exception as e:
        logger.error(f"Failed to generate narrative for incident {incident_id}: {e}")
        return None
