"""Case Management + Silent Evidence Hold.

Binds agency case numbers (BSO-26-48271, MDPD-26-12345, etc.) to Pylox
incidents so that footage can be recalled years later by case number, and
places "silent holds" on time windows so footage in those windows is
immutable until explicitly released.

Two features:

1. Case number binding
   - Officer enters their case number in the Pylox UI
   - One or more incidents get tagged to that case
   - The tagged footage is preserved outside the normal retention window
   - Any user (dealer, LEO, prosecutor) can recall the full evidence package
     by case number at any time

2. Silent evidence hold
   - Authorized user flags a camera + time window as evidence
   - Purge logic honors the flag: matching segments are NEVER deleted
   - Holds can be released explicitly (with audit log entry)
   - Holds have an optional case_number link

Retention enforcement:
    Before Frigate's normal retention purge runs (or before we manually purge),
    call is_footage_preserved(camera, start, end) — returns True if any hold
    or case binding protects the window.
"""

import logging
import time
from typing import Optional

from engine import database as db

logger = logging.getLogger("pylox-v2.police.cases")

DEFAULT_AGENCY = "Broward Sheriff's Office"


def bind_case(
    case_number: str,
    agency: str,
    incident_ids: list,
    officer_name: str = None,
    officer_badge: str = None,
    description: str = None,
    bound_by: str = None,
) -> dict:
    """Create (or reuse) a case and bind one or more incidents to it.

    Each bound incident's footage is preserved via auto-generated evidence holds
    covering its full timeframe.

    Returns {"case_id": int, "case_number": str, "incidents_bound": int, "holds_created": int}
    """
    case_id = db.create_case(
        case_number=case_number,
        agency=agency,
        officer_name=officer_name,
        officer_badge=officer_badge,
        description=description,
    )
    if not case_id:
        logger.error(f"Failed to create case {case_number}")
        return {
            "case_id": None,
            "case_number": case_number,
            "incidents_bound": 0,
            "holds_created": 0,
        }

    bound = 0
    holds_created = 0
    for incident_id in incident_ids:
        incident = db.get_incident(incident_id)
        if not incident:
            logger.warning(f"Incident {incident_id} not found — skipping bind")
            continue

        db.bind_incident_to_case(case_id, incident_id, bound_by=bound_by)
        bound += 1

        # Auto-create an evidence hold covering the incident window
        start = incident.get("start_time") or 0
        end = incident.get("end_time") or time.time()
        # Pad the window by 60 seconds on each side for context footage
        padded_start = max(0, start - 60)
        padded_end = end + 60
        camera = incident.get("camera", "unknown")
        db.create_evidence_hold(
            camera=camera,
            start_time=padded_start,
            end_time=padded_end,
            reason=f"Case binding: {case_number}",
            case_number=case_number,
            created_by=bound_by or officer_name or "system",
        )
        holds_created += 1

    db.log_leo_action(
        badge_number=officer_badge or "",
        action="case_bind",
        resource=f"case:{case_number}",
        details=f"Bound {bound} incidents to case {case_number} ({agency})",
    )

    logger.info(
        f"Case bound: {case_number} ({agency}) — "
        f"{bound} incidents, {holds_created} holds created"
    )

    return {
        "case_id": case_id,
        "case_number": case_number,
        "incidents_bound": bound,
        "holds_created": holds_created,
    }


def lookup_case(case_number: str) -> Optional[dict]:
    """Fetch a case with all bound incidents attached."""
    return db.get_case(case_number)


def list_open_cases(agency: str = None, limit: int = 100) -> list:
    return db.list_cases(agency=agency, status="open", limit=limit)


# --- Silent evidence holds ---

def place_hold(
    camera: str,
    start_time: float,
    end_time: float,
    reason: str,
    case_number: str = None,
    created_by: str = None,
) -> int:
    """Place a preservation hold on a camera/time-window."""
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    hold_id = db.create_evidence_hold(
        camera=camera,
        start_time=start_time,
        end_time=end_time,
        reason=reason,
        case_number=case_number,
        created_by=created_by,
    )

    db.log_leo_action(
        badge_number="",
        action="evidence_hold_create",
        resource=f"camera:{camera}",
        details=f"Hold {hold_id}: {reason} ({start_time}-{end_time})",
    )

    logger.info(
        f"Evidence hold placed: id={hold_id} camera={camera} "
        f"window={end_time - start_time:.0f}s reason={reason}"
    )
    return hold_id


def release_hold(hold_id: int, released_by: str = None):
    """Release a previously placed hold."""
    db.release_evidence_hold(hold_id, released_by=released_by)
    db.log_leo_action(
        badge_number="",
        action="evidence_hold_release",
        resource=f"hold:{hold_id}",
        details=f"Released by {released_by or 'system'}",
    )
    logger.info(f"Evidence hold released: id={hold_id} by={released_by}")


def list_active_holds(camera: str = None) -> list:
    return db.get_active_holds(camera=camera)


def is_footage_preserved(camera: str, start: float, end: float) -> bool:
    """Called by retention/purge logic before deleting a segment.

    Returns True if the time window overlaps any active hold and must
    NOT be deleted.
    """
    return db.is_timeframe_held(camera, start, end)


def filter_purgeable(camera: str, candidate_segments: list) -> list:
    """Given a list of {start, end, path} segments, return only those
    NOT protected by any active hold. Use this to filter Frigate segments
    before running purge.
    """
    return [
        s for s in candidate_segments
        if not is_footage_preserved(camera, s["start"], s["end"])
    ]
