"""Incident Report Generator — auto-generates reports from V2 events.

Produces HTML incident reports that include:
  - Event timeline
  - Camera snapshots at time of event
  - Behavior analysis (trajectory, speed, duration)
  - Anomaly scores
  - Rule violations
  - Person count history

Reports are generated on-demand or scheduled (daily summary).
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from engine import database as db
from engine.config import FRIGATE_API, DATA_DIR, CAMERAS

logger = logging.getLogger("pylox-v2.reports")

REPORTS_DIR = Path(DATA_DIR) / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_incident_report(event_id: int = None, camera: str = None,
                              since: float = None, until: float = None,
                              title: str = None) -> str:
    """Generate an HTML incident report.

    Can be scoped to:
      - A single event (event_id)
      - A camera + time range
      - A global time range

    Returns: path to generated HTML file
    """
    now = time.time()

    # Gather events
    if event_id:
        conn = db.get_db()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        conn.close()
        if not row:
            raise ValueError(f"Event {event_id} not found")
        events = [dict(row)]
        camera = events[0]["camera"]
        title = title or f"Incident Report — Event #{event_id}"
    else:
        since = since or (now - 3600)  # default last hour
        until = until or now
        events = db.get_events(camera=camera, since=since, limit=500)
        title = title or f"Security Report — {camera or 'All Cameras'}"

    # Gather related tracks
    track_ids = set(e.get("track_id") for e in events if e.get("track_id"))
    tracks = {}
    conn = db.get_db()
    for tid in track_ids:
        row = conn.execute("SELECT * FROM tracks WHERE id = ?", (tid,)).fetchone()
        if row:
            tracks[tid] = dict(row)
    conn.close()

    # Gather behavior stats for involved tracks
    behaviors = {}
    conn = db.get_db()
    for tid in track_ids:
        rows = conn.execute(
            "SELECT * FROM behavior_stats WHERE track_id = ? ORDER BY timestamp",
            (tid,)
        ).fetchall()
        if rows:
            behaviors[tid] = [dict(r) for r in rows]
    conn.close()

    # Generate HTML
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"report_{int(now)}_{camera or 'all'}"
    report_path = REPORTS_DIR / f"{report_id}.html"

    html = _render_html(
        title=title,
        report_time=report_time,
        events=events,
        tracks=tracks,
        behaviors=behaviors,
        camera=camera,
    )

    report_path.write_text(html)
    logger.info(f"Report generated: {report_path}")

    return str(report_path)


def generate_daily_summary(date: str = None) -> str:
    """Generate a daily summary report.

    Args:
        date: Date string YYYY-MM-DD (default: today)
    """
    if date:
        day = datetime.strptime(date, "%Y-%m-%d")
    else:
        day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    start = day.timestamp()
    end = (day + timedelta(days=1)).timestamp()

    events = db.get_events(since=start, limit=1000)
    events = [e for e in events if e["timestamp"] < end]

    # Stats
    total_events = len(events)
    by_severity = {}
    by_type = {}
    by_camera = {}
    for e in events:
        sev = e.get("severity", "info")
        by_severity[sev] = by_severity.get(sev, 0) + 1
        etype = e.get("event_type", "unknown")
        by_type[etype] = by_type.get(etype, 0) + 1
        cam = e.get("camera", "unknown")
        by_camera[cam] = by_camera.get(cam, 0) + 1

    title = f"Daily Security Summary — {day.strftime('%B %d, %Y')}"
    report_id = f"daily_{day.strftime('%Y%m%d')}"
    report_path = REPORTS_DIR / f"{report_id}.html"

    html = _render_daily_html(
        title=title,
        date=day.strftime("%B %d, %Y"),
        total_events=total_events,
        by_severity=by_severity,
        by_type=by_type,
        by_camera=by_camera,
        events=events[:50],  # Top 50 events
    )

    report_path.write_text(html)
    logger.info(f"Daily summary generated: {report_path}")
    return str(report_path)


def _render_html(title, report_time, events, tracks, behaviors, camera) -> str:
    """Render incident report HTML."""
    events_html = ""
    for e in events:
        data = json.loads(e["data"]) if isinstance(e["data"], str) else e["data"]
        severity_color = {"critical": "#ef4444", "warning": "#eab308", "info": "#3b82f6"}.get(
            e.get("severity", "info"), "#888"
        )
        ts = datetime.fromtimestamp(e["timestamp"]).strftime("%H:%M:%S")

        events_html += f"""
        <div class="event-row">
          <span class="event-time">{ts}</span>
          <span class="event-severity" style="color:{severity_color}">{e.get('severity','info').upper()}</span>
          <span class="event-type">{e.get('event_type','')}</span>
          <span class="event-camera">{e.get('camera','')}</span>
          <span class="event-detail">{json.dumps(data, indent=None)[:120]}</span>
        </div>"""

    tracks_html = ""
    for tid, track in tracks.items():
        duration = track.get("last_seen", 0) - track.get("first_seen", 0)
        tracks_html += f"""
        <div class="track-row">
          <span class="track-id">{tid[:12]}...</span>
          <span class="track-camera">{track.get('camera','')}</span>
          <span class="track-label">{track.get('label','')}</span>
          <span class="track-duration">{duration:.0f}s</span>
          <span class="track-zones">{track.get('zones','[]')}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0a0a0a; color: #e5e5e5; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; padding: 40px; }}
    .header {{ border-bottom: 2px solid #dc2626; padding-bottom: 20px; margin-bottom: 30px; }}
    .header h1 {{ font-size: 24px; font-weight: 600; color: #fff; }}
    .header .meta {{ font-size: 13px; color: #888; margin-top: 8px; }}
    .section {{ margin-bottom: 30px; }}
    .section h2 {{ font-size: 16px; font-weight: 600; color: #dc2626; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
    .event-row, .track-row {{ display: flex; gap: 16px; padding: 8px 12px; border-bottom: 1px solid #1a1a1a; font-size: 13px; align-items: center; }}
    .event-row:hover, .track-row:hover {{ background: #111; }}
    .event-time, .track-id {{ color: #666; font-family: monospace; min-width: 80px; }}
    .event-severity {{ font-weight: 600; min-width: 70px; }}
    .event-type, .track-label {{ color: #fff; min-width: 120px; }}
    .event-camera, .track-camera {{ color: #888; min-width: 60px; }}
    .event-detail {{ color: #666; font-size: 12px; overflow: hidden; text-overflow: ellipsis; }}
    .track-duration {{ color: #eab308; min-width: 60px; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; }}
    .stat-card {{ background: #111; border: 1px solid #222; border-radius: 8px; padding: 16px; }}
    .stat-value {{ font-size: 28px; font-weight: 700; color: #fff; }}
    .stat-label {{ font-size: 12px; color: #888; margin-top: 4px; text-transform: uppercase; }}
    .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #222; font-size: 11px; color: #444; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <div class="meta">Generated: {report_time} | Camera: {camera or 'All'} | Events: {len(events)}</div>
  </div>

  <div class="section">
    <h2>Summary</h2>
    <div class="stats-grid">
      <div class="stat-card"><div class="stat-value">{len(events)}</div><div class="stat-label">Total Events</div></div>
      <div class="stat-card"><div class="stat-value">{sum(1 for e in events if e.get('severity') == 'critical')}</div><div class="stat-label">Critical</div></div>
      <div class="stat-card"><div class="stat-value">{sum(1 for e in events if e.get('severity') == 'warning')}</div><div class="stat-label">Warnings</div></div>
      <div class="stat-card"><div class="stat-value">{len(tracks)}</div><div class="stat-label">Persons Tracked</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Event Timeline</h2>
    {events_html or '<div style="color:#666;padding:12px">No events in this period.</div>'}
  </div>

  <div class="section">
    <h2>Tracked Persons</h2>
    {tracks_html or '<div style="color:#666;padding:12px">No tracks in this period.</div>'}
  </div>

  <div class="footer">
    Pylox Vision V2 Intelligence Engine — Confidential Security Report<br>
    Generated by Pylox Systems
  </div>
</body>
</html>"""


def _render_daily_html(title, date, total_events, by_severity, by_type,
                       by_camera, events) -> str:
    """Render daily summary HTML."""
    severity_bars = ""
    for sev in ["critical", "warning", "info"]:
        count = by_severity.get(sev, 0)
        color = {"critical": "#ef4444", "warning": "#eab308", "info": "#3b82f6"}[sev]
        width = min(100, count * 2) if total_events > 0 else 0
        severity_bars += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
          <span style="min-width:70px;font-size:13px;color:{color}">{sev.upper()}</span>
          <div style="flex:1;background:#1a1a1a;height:20px;border-radius:4px;overflow:hidden">
            <div style="width:{width}%;height:100%;background:{color};border-radius:4px"></div>
          </div>
          <span style="min-width:40px;text-align:right;font-size:13px;color:#888">{count}</span>
        </div>"""

    camera_bars = ""
    max_cam = max(by_camera.values()) if by_camera else 1
    for cam in sorted(by_camera.keys()):
        count = by_camera[cam]
        width = (count / max_cam * 100) if max_cam > 0 else 0
        camera_bars += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">
          <span style="min-width:60px;font-size:13px;color:#888">{cam}</span>
          <div style="flex:1;background:#1a1a1a;height:16px;border-radius:4px;overflow:hidden">
            <div style="width:{width}%;height:100%;background:#dc2626;border-radius:4px"></div>
          </div>
          <span style="min-width:30px;text-align:right;font-size:12px;color:#666">{count}</span>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ background: #0a0a0a; color: #e5e5e5; font-family: -apple-system, BlinkMacSystemFont, 'SF Pro', system-ui, sans-serif; padding: 40px; max-width: 800px; margin: 0 auto; }}
    .header {{ border-bottom: 2px solid #dc2626; padding-bottom: 20px; margin-bottom: 30px; }}
    .header h1 {{ font-size: 24px; font-weight: 600; color: #fff; }}
    .header .meta {{ font-size: 13px; color: #888; margin-top: 8px; }}
    .section {{ margin-bottom: 30px; }}
    .section h2 {{ font-size: 14px; font-weight: 600; color: #dc2626; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }}
    .big-number {{ font-size: 48px; font-weight: 700; color: #fff; }}
    .footer {{ margin-top: 40px; padding-top: 16px; border-top: 1px solid #222; font-size: 11px; color: #444; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <div class="meta">{date}</div>
  </div>

  <div style="text-align:center;margin-bottom:30px">
    <div class="big-number">{total_events}</div>
    <div style="font-size:14px;color:#888;text-transform:uppercase">Total Events</div>
  </div>

  <div class="section">
    <h2>By Severity</h2>
    {severity_bars}
  </div>

  <div class="section">
    <h2>By Camera</h2>
    {camera_bars}
  </div>

  <div class="footer">
    Pylox Vision V2 — Daily Security Summary<br>
    Generated by Pylox Systems
  </div>
</body>
</html>"""
