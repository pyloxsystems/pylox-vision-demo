"""Vehicle Make/Model/Color AI Enrichment.

Secondary Gemini pass that extracts structured vehicle details from a detected
vehicle track — make, model, color, year range, body type, license plate
(if visible). Critical for stolen vehicle reports, BOLO matching, and officer
briefings.

Instead of Frigate saying "car detected", Pylox says
"2024 White BMW 530i sedan, Florida plate ABC-1234".

Input: a vehicle snapshot image bytes + basic track info.
Output: structured dict with make, model, color, year_range, body_type, plate, confidence.

Can also run on every car event that enters a protected lot after hours,
feeding the BOLO scan index with enriched tags.
"""

import base64
import json
import logging
import time
from typing import Optional

logger = logging.getLogger("pylox-v2.police.vehicle_ai")


VEHICLE_SYSTEM_PROMPT = """You are extracting structured vehicle details from a surveillance
snapshot for use in a police BOLO (Be On the Look Out) database.

Look at the vehicle in the image and return STRICT JSON matching this schema exactly.
Use null for any field you cannot determine with high confidence. Never guess.

Output JSON format (no markdown, no code fences, no prose):

{
  "make": "BMW|Ford|Toyota|...|null",
  "model": "530i|F-150|Camry|...|null",
  "year_range": "2020-2024|null",
  "color": "white|black|silver|red|...|null",
  "body_type": "sedan|suv|pickup|van|coupe|convertible|hatchback|wagon|null",
  "license_plate": "text if visible and readable, otherwise null",
  "plate_state": "FL|NY|...|null",
  "distinguishing_features": "damage, decals, aftermarket parts, etc. or empty",
  "confidence": 0.0-1.0
}

Rules:
- Only fill fields you can verify from the image. Never guess make/model from shape alone.
- license_plate should be the actual characters, in uppercase, no dashes or spaces
  (e.g. "ABC1234" not "ABC-1234"). Use null if blurry, partial, or angle-obscured.
- confidence is your overall confidence in the extraction (0.0-1.0).
- Output ONLY the JSON object. No explanation."""


def extract_vehicle_details(
    image_bytes: bytes,
    gemini_connector,
    context: str = "",
) -> Optional[dict]:
    """Extract structured vehicle details from a snapshot.

    Args:
        image_bytes: JPEG/PNG bytes of the vehicle snapshot
        gemini_connector: shared GeminiConnector from app.py
        context: optional extra context (time, camera, etc.)

    Returns:
        dict with vehicle details, or None on failure.
    """
    if not gemini_connector or not getattr(gemini_connector, "available", False):
        logger.warning("Gemini unavailable — cannot extract vehicle details")
        return None

    if not image_bytes:
        return None

    try:
        from google.genai import types
        client = gemini_connector._client
        if not client:
            return None

        prompt = VEHICLE_SYSTEM_PROMPT
        if context:
            prompt += f"\n\nAdditional context: {context}"

        contents = [
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        text = (response.text or "").strip() if response else ""
        if not text:
            return None

        if text.startswith("```"):
            lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        parsed = json.loads(text)

        # Normalize output
        result = {
            "make": parsed.get("make"),
            "model": parsed.get("model"),
            "year_range": parsed.get("year_range"),
            "color": parsed.get("color"),
            "body_type": parsed.get("body_type"),
            "license_plate": (parsed.get("license_plate") or "").upper().replace("-", "").replace(" ", "") or None,
            "plate_state": parsed.get("plate_state"),
            "distinguishing_features": parsed.get("distinguishing_features") or "",
            "confidence": float(parsed.get("confidence") or 0.0),
            "extracted_at": time.time(),
        }

        logger.info(
            f"Vehicle extracted: {result.get('color') or '?'} "
            f"{result.get('year_range') or '?'} "
            f"{result.get('make') or '?'} {result.get('model') or '?'} "
            f"plate={result.get('license_plate') or 'unknown'} "
            f"conf={result['confidence']:.2f}"
        )

        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse vehicle JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Vehicle extraction failed: {e}")
        return None


def format_vehicle_summary(details: dict) -> str:
    """Human-readable one-line summary for reports and alerts."""
    if not details:
        return "unidentified vehicle"
    parts = []
    if details.get("color"):
        parts.append(details["color"])
    if details.get("year_range"):
        parts.append(details["year_range"])
    if details.get("make"):
        parts.append(details["make"])
    if details.get("model"):
        parts.append(details["model"])
    if details.get("body_type"):
        parts.append(details["body_type"])
    base = " ".join(parts) if parts else "unidentified vehicle"
    plate = details.get("license_plate")
    state = details.get("plate_state")
    if plate:
        tag = f"{state} {plate}" if state else plate
        base += f" ({tag})"
    return base
