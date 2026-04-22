"""Responder View — signed time-limited live camera access for responding officers.

When Gemini confirms a threat, Pylox can issue a short-lived JWT that grants
a responding officer live view + tactical brief access on their phone for
30 minutes. The token is included in the push to BSO RTCC (or SMS'd to a
pre-registered responder phone number).

No cloud dependency. No login required on the responder side. Just a link
that works until it expires.

Token lifecycle:
    issue_token(incident_id, issued_to="Deputy Smith, BSO-12345", ttl_seconds=1800)
        → returns (token_string, full_url)
    verify_token(token_string)
        → returns payload dict if valid, else None
    revoke_token(token_id)
        → marks revoked in DB

Routes are added in officer_api.py so this module stays transport-agnostic.
"""

import os
import json
import time
import uuid
import logging
from typing import Optional

import jwt

from engine import database as db

logger = logging.getLogger("pylox-v2.police.responder")

# Load (or generate) the signing key. In production the key should live in
# an env var; for dev we persist it alongside the database.
_KEY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "data",
    "responder_signing_key.txt",
)


def _load_or_create_key() -> str:
    env_key = os.getenv("PYLOX_RESPONDER_KEY")
    if env_key:
        return env_key
    try:
        if os.path.exists(_KEY_FILE):
            with open(_KEY_FILE, "r") as f:
                k = f.read().strip()
                if k:
                    return k
        new_key = uuid.uuid4().hex + uuid.uuid4().hex
        os.makedirs(os.path.dirname(_KEY_FILE), exist_ok=True)
        with open(_KEY_FILE, "w") as f:
            f.write(new_key)
        os.chmod(_KEY_FILE, 0o600)
        logger.info("Generated new responder signing key")
        return new_key
    except Exception as e:
        logger.error(f"Failed to load/create responder key: {e}")
        return "pylox-fallback-key-change-me"


SIGNING_KEY = _load_or_create_key()
ALGORITHM = "HS256"
DEFAULT_TTL = 30 * 60  # 30 minutes


def issue_token(
    incident_id: int,
    issued_to: str = None,
    ttl_seconds: int = DEFAULT_TTL,
    base_url: str = None,
) -> dict:
    """Issue a responder access token for an incident.

    Returns {"token_id": str, "token": str, "url": str, "expires_at": float, "ttl_seconds": int}
    """
    now = time.time()
    token_id = uuid.uuid4().hex[:16]
    expires_at = now + ttl_seconds

    payload = {
        "jti": token_id,
        "sub": str(incident_id),
        "iat": int(now),
        "exp": int(expires_at),
        "iss": "pylox-vision",
        "aud": "responder",
        "issued_to": issued_to or "",
    }

    token = jwt.encode(payload, SIGNING_KEY, algorithm=ALGORITHM)
    if isinstance(token, bytes):
        token = token.decode("utf-8")

    db.save_responder_token(
        token_id=token_id,
        incident_id=incident_id,
        expires_at=expires_at,
        issued_to=issued_to,
    )

    base = base_url or os.getenv("PYLOX_PUBLIC_URL") or "http://localhost:3450"
    url = f"{base.rstrip('/')}/responder/{token}"

    logger.info(
        f"Responder token issued: incident={incident_id} "
        f"token_id={token_id} ttl={ttl_seconds}s issued_to={issued_to or '-'}"
    )
    return {
        "token_id": token_id,
        "token": token,
        "url": url,
        "expires_at": expires_at,
        "ttl_seconds": ttl_seconds,
    }


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a responder token.

    Returns the token's database record (with audit tracking) if valid,
    or None if expired, revoked, or invalid.
    """
    try:
        payload = jwt.decode(
            token,
            SIGNING_KEY,
            algorithms=[ALGORITHM],
            audience="responder",
            issuer="pylox-vision",
        )
    except jwt.ExpiredSignatureError:
        logger.info("Responder token rejected: expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Responder token rejected: invalid ({e})")
        return None

    token_id = payload.get("jti")
    if not token_id:
        return None

    record = db.get_responder_token(token_id)
    if not record:
        logger.warning(f"Responder token {token_id} not in database — rejected")
        return None
    if record.get("revoked"):
        logger.info(f"Responder token {token_id} revoked — rejected")
        return None
    if record.get("expires_at", 0) < time.time():
        logger.info(f"Responder token {token_id} DB-expired — rejected")
        return None

    # Touch access counter
    db.touch_responder_token(token_id)

    return {
        "token_id": token_id,
        "incident_id": int(payload["sub"]),
        "issued_to": payload.get("issued_to", ""),
        "expires_at": payload.get("exp", 0),
        "record": record,
    }


def revoke_token(token_id: str):
    db.revoke_responder_token(token_id)
    logger.info(f"Responder token {token_id} revoked")


# --- HTML template for the responder view page ---

RESPONDER_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pylox Vision — Active Incident</title>
<style>
  :root {{
    --bg: #0a0e1a;
    --card: #141a2e;
    --accent: #3b82f6;
    --danger: #ef4444;
    --text: #f1f5f9;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 12px;
  }}
  .header {{
    background: linear-gradient(135deg, var(--danger) 0%, #991b1b 100%);
    padding: 14px 16px;
    border-radius: 12px;
    margin-bottom: 12px;
  }}
  .header h1 {{
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .pulse {{
    width: 10px;
    height: 10px;
    background: #fff;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
  }}
  .header .sub {{
    margin-top: 4px;
    font-size: 13px;
    opacity: 0.9;
  }}
  .card {{
    background: var(--card);
    padding: 14px;
    border-radius: 12px;
    margin-bottom: 12px;
  }}
  .card h2 {{
    margin: 0 0 10px 0;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--muted);
    letter-spacing: 0.5px;
  }}
  .feed {{
    width: 100%;
    border-radius: 8px;
    background: #000;
    min-height: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 12px;
  }}
  .feed img {{
    width: 100%;
    border-radius: 8px;
    display: block;
  }}
  .threat-bar {{
    height: 14px;
    background: #1e293b;
    border-radius: 7px;
    overflow: hidden;
    margin: 8px 0 4px;
  }}
  .threat-fill {{
    height: 100%;
    background: linear-gradient(90deg, #fbbf24, #ef4444);
    transition: width 0.5s ease;
  }}
  .suspect {{
    border-left: 3px solid var(--accent);
    padding: 8px 12px;
    margin: 8px 0;
    background: rgba(59, 130, 246, 0.08);
    border-radius: 0 8px 8px 0;
  }}
  .suspect .id {{
    color: var(--accent);
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
  }}
  .suspect .desc {{
    margin-top: 4px;
    font-size: 14px;
    line-height: 1.4;
  }}
  .meta {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    font-size: 12px;
  }}
  .meta > div {{
    padding: 6px 8px;
    background: rgba(255, 255, 255, 0.04);
    border-radius: 6px;
  }}
  .meta .label {{
    color: var(--muted);
    font-size: 10px;
    text-transform: uppercase;
    margin-bottom: 2px;
  }}
  .safety {{
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 12px;
  }}
  .narration {{
    font-size: 13px;
    line-height: 1.5;
    color: var(--muted);
    max-height: 180px;
    overflow-y: auto;
  }}
  .narration .entry {{
    padding: 6px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
  }}
  .narration .entry:last-child {{
    border-bottom: 0;
  }}
  .narration .ts {{
    color: var(--accent);
    font-weight: 600;
    font-size: 11px;
  }}
  .footer {{
    text-align: center;
    padding: 16px 12px;
    font-size: 10px;
    color: var(--muted);
  }}
</style>
</head>
<body>
  <div class="header">
    <h1><span class="pulse"></span> LIVE INCIDENT — {site_name}</h1>
    <div class="sub">Camera: {camera_name} &middot; Started: {start_time}</div>
  </div>

  {safety_notes}

  <div class="card">
    <h2>Live View</h2>
    <div class="feed">
      <img id="feed" src="{snapshot_url}" alt="Live camera feed" onerror="this.style.display='none'; this.parentElement.innerHTML='(feed unavailable)';">
    </div>
  </div>

  <div class="card">
    <h2>Threat Level</h2>
    <div class="threat-bar">
      <div class="threat-fill" style="width: {threat_pct}%"></div>
    </div>
    <div style="font-size: 12px; color: var(--muted); margin-top: 4px;">
      {threat}/10 &middot; {recommended_response}
    </div>
  </div>

  <div class="card">
    <h2>Suspects ({suspect_count})</h2>
    {suspects_html}
  </div>

  <div class="card">
    <h2>Tactical Details</h2>
    <div class="meta">
      <div><div class="label">Entry Point</div>{entry_point}</div>
      <div><div class="label">Direction</div>{primary_direction}</div>
      <div><div class="label">Exit Route</div>{exit_route}</div>
      <div><div class="label">Last Seen</div>{last_known}</div>
    </div>
  </div>

  <div class="card">
    <h2>AI Observations</h2>
    <div class="narration">{narration_html}</div>
  </div>

  <div class="footer">
    Pylox Vision &middot; Token expires in <span id="ttl">{ttl_display}</span><br>
    Issued to: {issued_to}
  </div>

<script>
  // Auto-refresh snapshot every 2 seconds
  const feed = document.getElementById('feed');
  const base = feed && feed.src.split('?')[0];
  if (feed) {{
    setInterval(() => {{
      feed.src = base + '?t=' + Date.now();
    }}, 2000);
  }}

  // Countdown timer
  const exp = {expires_at};
  const ttlEl = document.getElementById('ttl');
  function updateTTL() {{
    const remaining = Math.max(0, exp - (Date.now() / 1000));
    const mins = Math.floor(remaining / 60);
    const secs = Math.floor(remaining % 60);
    if (ttlEl) {{
      ttlEl.textContent = mins + "m " + secs.toString().padStart(2, '0') + "s";
    }}
    if (remaining <= 0) {{
      document.body.innerHTML = '<div style="padding:40px;text-align:center;color:#94a3b8;">Token expired. Contact dispatch for a new link.</div>';
    }}
  }}
  updateTTL();
  setInterval(updateTTL, 1000);
</script>
</body>
</html>"""


def render_responder_page(
    incident: dict,
    tactical_brief: dict,
    narrations: list,
    site_config: dict,
    camera_name: str,
    snapshot_url: str,
    expires_at: float,
    issued_to: str,
) -> str:
    """Render the mobile-friendly responder HTML page."""
    from datetime import datetime

    site = site_config or {}
    brief = tactical_brief or {}
    suspects = brief.get("suspects", [])

    suspects_html = ""
    if suspects:
        for s in suspects:
            desc = s.get("description", "Unknown subject")
            weapons = s.get("weapons", [])
            tools = s.get("tools_visible", [])
            extras = []
            if weapons:
                extras.append(f"ARMED: {', '.join(weapons)}")
            if tools:
                extras.append(f"Tools: {', '.join(tools)}")
            if s.get("distinguishing_features"):
                extras.append(s["distinguishing_features"])
            extras_str = (" &middot; ".join(extras)) if extras else ""
            suspects_html += f"""
            <div class="suspect">
              <div class="id">{s.get('id', 'Subject')}</div>
              <div class="desc">{desc}</div>
              {f'<div style="font-size: 11px; color: var(--muted); margin-top: 4px;">{extras_str}</div>' if extras_str else ''}
            </div>
            """
    else:
        suspects_html = '<div style="color: var(--muted); font-size: 12px;">No suspects detected in briefing.</div>'

    narration_html = ""
    if narrations:
        for n in narrations[-10:]:
            t = datetime.fromtimestamp(n.get("timestamp", 0)).strftime("%H:%M:%S")
            txt = n.get("narration", "")
            narration_html += f'<div class="entry"><span class="ts">{t}</span> {txt}</div>'
    else:
        narration_html = '<div>No observations available.</div>'

    safety = brief.get("officer_safety_notes", "")
    safety_html = ""
    if safety:
        safety_html = f'<div class="safety"><strong>OFFICER SAFETY:</strong> {safety}</div>'

    threat = brief.get("property_threat_level") or incident.get("max_threat", 0)
    primary_direction = "unknown"
    if suspects and suspects[0].get("direction_of_travel"):
        primary_direction = suspects[0]["direction_of_travel"]

    ttl_seconds = max(0, int(expires_at - time.time()))
    ttl_display = f"{ttl_seconds // 60}m {ttl_seconds % 60:02d}s"

    return RESPONDER_PAGE_HTML.format(
        site_name=site.get("name", "Protected Site"),
        camera_name=camera_name,
        start_time=datetime.fromtimestamp(incident.get("start_time", 0)).strftime("%H:%M:%S EDT"),
        safety_notes=safety_html,
        snapshot_url=snapshot_url,
        threat=threat,
        threat_pct=int(threat) * 10,
        recommended_response=brief.get("recommended_response", "standard").upper(),
        suspect_count=len(suspects),
        suspects_html=suspects_html,
        entry_point=brief.get("entry_point") or "unknown",
        primary_direction=primary_direction,
        exit_route=brief.get("exit_route_predicted") or "unknown",
        last_known=brief.get("last_known_location") or "unknown",
        narration_html=narration_html,
        ttl_display=ttl_display,
        expires_at=expires_at,
        issued_to=issued_to or "authorized responder",
    )
