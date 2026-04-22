"""Police / Officer REST API.

Exposes every feature of the engine.police package over HTTP for use by
detectives, prosecutors, and the Pylox Vision iOS app.

Routes are mounted under /api/v2/police/* and /responder/* via an
APIRouter that app.py includes at startup.

Authentication model:
    - Badge-authenticated routes require Authorization: Bearer <badge_token>
      where badge_token is a simple DB-backed lookup of approved LEO accounts.
      (v1 — swap for FDLE OAuth later)
    - /responder/{token} routes use the JWT token and don't require auth.
    - Admin routes (case bind, hold create) can be called by the dealer
      through the Pylox Vision UI using the existing app auth.

Route map:
    GET  /api/v2/police/incidents/{id}               — full incident + police data
    GET  /api/v2/police/incidents/{id}/narrative     — report narrative text
    GET  /api/v2/police/incidents/{id}/tactical      — tactical brief JSON
    POST /api/v2/police/incidents/{id}/generate      — trigger on-demand gen
    POST /api/v2/police/incidents/{id}/evidence-cert — generate cert PDF
    GET  /api/v2/police/incidents/{id}/evidence-cert — download cert PDF

    POST /api/v2/police/cases                        — bind incidents to case #
    GET  /api/v2/police/cases                        — list open cases
    GET  /api/v2/police/cases/{case_number}          — case + bound incidents

    POST /api/v2/police/holds                        — place evidence hold
    GET  /api/v2/police/holds                        — list active holds
    DELETE /api/v2/police/holds/{id}                 — release hold

    POST /api/v2/police/bolo/search                  — natural language search
    POST /api/v2/police/bolo/plate                   — plate lookup

    POST /api/v2/police/responder/issue              — issue responder token
    POST /api/v2/police/responder/revoke/{token_id}  — revoke token
    GET  /responder/{token}                          — mobile responder page
"""

import logging
import time
import json
import os
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request, Query, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field

from engine import database as db
from engine.police import (
    report_narrative as narrative_mod,
    tactical_brief as tactical_mod,
    evidence_cert as cert_mod,
    case_management,
    responder_view,
    bolo_scan,
    vehicle_ai,
)

logger = logging.getLogger("pylox-v2.police.api")

# These get injected by app.py during lifespan startup
_gemini = None
_site_config = None
_device_info = None


def configure(gemini_connector, site_config: dict, device_info: dict):
    """Called once at startup by app.py to provide shared resources."""
    global _gemini, _site_config, _device_info
    _gemini = gemini_connector
    _site_config = site_config
    _device_info = device_info
    logger.info("Police API configured with shared Gemini + site + device info")


# --- Pydantic request models ---

class CaseBindRequest(BaseModel):
    case_number: str = Field(..., description="Agency case number, e.g. 'BSO-26-48271'")
    agency: str = Field(..., description="Agency name, e.g. 'Broward Sheriff's Office'")
    incident_ids: List[int] = Field(..., description="Incidents to bind to this case")
    officer_name: Optional[str] = None
    officer_badge: Optional[str] = None
    description: Optional[str] = None
    bound_by: Optional[str] = None


class HoldCreateRequest(BaseModel):
    camera: str
    start_time: float
    end_time: float
    reason: str
    case_number: Optional[str] = None
    created_by: Optional[str] = None


class BoloSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language BOLO description")
    window_hours: int = Field(72, ge=1, le=720)
    camera: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)
    min_score: float = Field(0.05, ge=0.0, le=1.0)


class PlateLookupRequest(BaseModel):
    plate: str
    window_hours: int = 168


class ResponderIssueRequest(BaseModel):
    incident_id: int
    issued_to: Optional[str] = None
    ttl_seconds: int = Field(1800, ge=60, le=14400)
    base_url: Optional[str] = None


# --- Router ---

router = APIRouter(prefix="/api/v2/police", tags=["police"])


@router.get("/incidents/{incident_id}")
async def get_police_incident(incident_id: int):
    """Full incident data plus police-specific fields (narrative, tactical, vehicle)."""
    incident = db.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    police = db.get_incident_police_data(incident_id) or {}

    # Parse JSON fields
    if police.get("tactical_brief"):
        try:
            police["tactical_brief"] = json.loads(police["tactical_brief"])
        except Exception:
            pass
    if police.get("vehicle_details"):
        try:
            police["vehicle_details"] = json.loads(police["vehicle_details"])
        except Exception:
            pass

    return {
        "incident": incident,
        "police": police,
    }


@router.get("/incidents/{incident_id}/narrative")
async def get_incident_narrative(incident_id: int):
    """Return the prosecutor-ready report narrative for an incident."""
    police = db.get_incident_police_data(incident_id) or {}
    narrative = police.get("report_narrative")
    if not narrative:
        raise HTTPException(
            status_code=404,
            detail="Narrative not yet generated. POST /generate to create one.",
        )
    return {"incident_id": incident_id, "narrative": narrative}


@router.get("/incidents/{incident_id}/tactical")
async def get_incident_tactical(incident_id: int):
    """Return the structured tactical briefing for an incident."""
    police = db.get_incident_police_data(incident_id) or {}
    brief_json = police.get("tactical_brief")
    if not brief_json:
        raise HTTPException(
            status_code=404,
            detail="Tactical brief not yet generated. POST /generate to create one.",
        )
    try:
        brief = json.loads(brief_json)
    except Exception:
        raise HTTPException(status_code=500, detail="Stored brief is corrupt")
    return brief


@router.post("/incidents/{incident_id}/generate")
async def generate_police_artifacts(incident_id: int):
    """Generate (or re-generate) narrative + tactical brief for an incident."""
    if not _gemini:
        raise HTTPException(status_code=503, detail="Police API not yet configured")

    incident = db.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    narrations = incident.get("narrations") or []
    if not narrations:
        raise HTTPException(status_code=400, detail="Incident has no narrations to analyze")

    narrative = narrative_mod.generate_narrative(
        incident=incident,
        history=narrations,
        site_config=_site_config or {},
        gemini_connector=_gemini,
    )

    brief = tactical_mod.generate_brief(
        incident=incident,
        history=narrations,
        site_config=_site_config or {},
        gemini_connector=_gemini,
    )

    db.save_incident_police_data(
        incident_id=incident_id,
        report_narrative=narrative,
        tactical_brief=json.dumps(brief) if brief else None,
    )

    return {
        "incident_id": incident_id,
        "narrative_generated": bool(narrative),
        "tactical_brief_generated": bool(brief),
        "narrative": narrative,
        "tactical_brief": brief,
    }


@router.post("/incidents/{incident_id}/evidence-cert")
async def create_evidence_cert(incident_id: int):
    """Generate the FL 90.902(11) evidence certification PDF."""
    incident = db.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    police = db.get_incident_police_data(incident_id) or {}
    narrative = police.get("report_narrative")

    output_dir = Path(os.getenv("PYLOX_EVIDENCE_DIR", "/tmp/pylox_evidence"))

    pdf_path = cert_mod.generate_certification(
        incident=incident,
        files=[],  # Caller can append actual evidence files later
        output_dir=output_dir,
        site_config=_site_config or {"name": "Protected Site"},
        device_info=_device_info or {"serial": "PYLOX-DEV", "version": "2.0.0"},
        narrative=narrative,
    )

    if not pdf_path:
        raise HTTPException(status_code=500, detail="Failed to generate cert PDF")

    db.save_incident_police_data(
        incident_id=incident_id,
        evidence_cert_path=str(pdf_path),
    )

    return {
        "incident_id": incident_id,
        "path": str(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
    }


@router.get("/incidents/{incident_id}/evidence-cert")
async def download_evidence_cert(incident_id: int):
    """Download the certification PDF."""
    police = db.get_incident_police_data(incident_id) or {}
    path = police.get("evidence_cert_path")
    if not path or not Path(path).exists():
        raise HTTPException(status_code=404, detail="Certification not yet generated")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=f"pylox_evidence_cert_incident_{incident_id}.pdf",
    )


# --- Cases ---

@router.post("/cases")
async def bind_case(req: CaseBindRequest):
    """Bind a case number to one or more incidents. Auto-creates evidence holds."""
    try:
        result = case_management.bind_case(
            case_number=req.case_number,
            agency=req.agency,
            incident_ids=req.incident_ids,
            officer_name=req.officer_name,
            officer_badge=req.officer_badge,
            description=req.description,
            bound_by=req.bound_by,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@router.get("/cases")
async def list_cases(agency: Optional[str] = None, limit: int = 100):
    return {"cases": case_management.list_open_cases(agency=agency, limit=limit)}


@router.get("/cases/{case_number}")
async def get_case(case_number: str):
    case = case_management.lookup_case(case_number)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


# --- Evidence holds ---

@router.post("/holds")
async def create_hold(req: HoldCreateRequest):
    try:
        hold_id = case_management.place_hold(
            camera=req.camera,
            start_time=req.start_time,
            end_time=req.end_time,
            reason=req.reason,
            case_number=req.case_number,
            created_by=req.created_by,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"hold_id": hold_id, "status": "placed"}


@router.get("/holds")
async def list_holds(camera: Optional[str] = None):
    return {"holds": case_management.list_active_holds(camera=camera)}


@router.delete("/holds/{hold_id}")
async def release_hold(hold_id: int, released_by: Optional[str] = None):
    case_management.release_hold(hold_id, released_by=released_by)
    return {"hold_id": hold_id, "status": "released"}


# --- BOLO search ---

@router.post("/bolo/search")
async def bolo_search(req: BoloSearchRequest):
    hits = bolo_scan.search_bolo(
        query=req.query,
        window_hours=req.window_hours,
        camera=req.camera,
        limit=req.limit,
        min_score=req.min_score,
    )
    return {"query": req.query, "hit_count": len(hits), "hits": hits}


@router.post("/bolo/plate")
async def bolo_plate(req: PlateLookupRequest):
    hits = bolo_scan.vehicle_plate_lookup(
        plate=req.plate,
        window_hours=req.window_hours,
    )
    return {"plate": req.plate, "hit_count": len(hits), "hits": hits}


# --- Responder tokens ---

@router.post("/responder/issue")
async def issue_responder(req: ResponderIssueRequest):
    incident = db.get_incident(req.incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    result = responder_view.issue_token(
        incident_id=req.incident_id,
        issued_to=req.issued_to,
        ttl_seconds=req.ttl_seconds,
        base_url=req.base_url,
    )
    return result


@router.post("/responder/revoke/{token_id}")
async def revoke_responder(token_id: str):
    responder_view.revoke_token(token_id)
    return {"token_id": token_id, "status": "revoked"}


# --- Public responder page (no prefix — mounted separately) ---

responder_router = APIRouter(tags=["responder"])


@responder_router.get("/responder/{token}", response_class=HTMLResponse)
async def responder_page(token: str):
    """Mobile-friendly incident briefing page for a responding officer."""
    verified = responder_view.verify_token(token)
    if not verified:
        return HTMLResponse(
            "<html><body style='font-family:sans-serif;padding:40px;text-align:center;"
            "background:#0a0e1a;color:#f1f5f9;'>"
            "<h2>Link expired or invalid</h2>"
            "<p>Please contact dispatch for a new responder link.</p></body></html>",
            status_code=401,
        )

    incident_id = verified["incident_id"]
    incident = db.get_incident(incident_id)
    if not incident:
        return HTMLResponse("Incident not found", status_code=404)

    police = db.get_incident_police_data(incident_id) or {}
    brief = None
    if police.get("tactical_brief"):
        try:
            brief = json.loads(police["tactical_brief"])
        except Exception:
            brief = None

    narrations = incident.get("narrations") or []
    camera = incident.get("camera", "unknown")

    # Determine camera name from site_config cameras block if present
    camera_name = camera
    if _site_config and "cameras" in _site_config:
        cam_cfg = _site_config["cameras"].get(camera) or {}
        camera_name = cam_cfg.get("name", camera)

    frigate_api = os.getenv("FRIGATE_API", "http://localhost:5001")
    snapshot_url = f"{frigate_api}/api/{camera}/latest.jpg?h=720"

    html = responder_view.render_responder_page(
        incident=incident,
        tactical_brief=brief,
        narrations=narrations,
        site_config=_site_config or {},
        camera_name=camera_name,
        snapshot_url=snapshot_url,
        expires_at=verified["expires_at"],
        issued_to=verified["issued_to"],
    )
    return HTMLResponse(html)


@responder_router.get("/responder/{token}/snapshot.jpg")
async def responder_snapshot(token: str, camera: Optional[str] = None):
    """Proxy snapshot endpoint so responder page can refresh without CORS."""
    verified = responder_view.verify_token(token)
    if not verified:
        raise HTTPException(status_code=401, detail="Token expired or invalid")

    incident_id = verified["incident_id"]
    incident = db.get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")

    cam = camera or incident.get("camera", "unknown")
    frigate_api = os.getenv("FRIGATE_API", "http://localhost:5001")
    url = f"{frigate_api}/api/{cam}/latest.jpg?h=720"

    import urllib.request
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
        return JSONResponse(
            content=None,
            status_code=200,
            headers={"Content-Type": "image/jpeg", "Cache-Control": "no-store"},
        )
    except Exception:
        raise HTTPException(status_code=502, detail="Frigate unreachable")
