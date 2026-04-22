"""SQLite database for V2 events, tracks, and behavior data."""

import sqlite3
import json
import time
import os
from engine.config import DB_PATH, DATA_DIR


def get_db() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracks (
            id TEXT PRIMARY KEY,
            camera TEXT NOT NULL,
            label TEXT NOT NULL,
            first_seen REAL NOT NULL,
            last_seen REAL NOT NULL,
            positions TEXT NOT NULL DEFAULT '[]',
            zones TEXT NOT NULL DEFAULT '[]',
            thumbnail_path TEXT,
            reid_id TEXT,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            camera TEXT NOT NULL,
            event_type TEXT NOT NULL,
            severity TEXT NOT NULL DEFAULT 'info',
            track_id TEXT,
            data TEXT NOT NULL DEFAULT '{}',
            acknowledged INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS behavior_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            camera TEXT NOT NULL,
            track_id TEXT NOT NULL,
            behavior_type TEXT NOT NULL,
            value REAL,
            data TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS anomaly_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            camera TEXT NOT NULL,
            score REAL NOT NULL,
            frame_path TEXT,
            data TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL,
            trigger_source TEXT NOT NULL,
            track_id TEXT,
            initial_threat INTEGER DEFAULT 0,
            max_threat INTEGER DEFAULT 0,
            resolution TEXT,
            total_gemini_calls INTEGER DEFAULT 0,
            deterrent_triggered INTEGER DEFAULT 0,
            notification_sent INTEGER DEFAULT 0,
            snapshot_path TEXT,
            data TEXT NOT NULL DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS incident_narrations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            threat INTEGER,
            narration TEXT,
            action TEXT,
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        );

        /* Pylox Vision — Law Enforcement tables */

        CREATE TABLE IF NOT EXISTS incident_police (
            incident_id INTEGER PRIMARY KEY,
            report_narrative TEXT,
            tactical_brief TEXT,
            vehicle_details TEXT,
            evidence_cert_path TEXT,
            generated_at REAL,
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        );

        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_number TEXT NOT NULL UNIQUE,
            agency TEXT NOT NULL,
            officer_name TEXT,
            officer_badge TEXT,
            opened_at REAL NOT NULL,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'open'
        );

        CREATE TABLE IF NOT EXISTS case_incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            incident_id INTEGER NOT NULL,
            bound_at REAL NOT NULL,
            bound_by TEXT,
            FOREIGN KEY (case_id) REFERENCES cases(id),
            FOREIGN KEY (incident_id) REFERENCES incidents(id),
            UNIQUE(case_id, incident_id)
        );

        CREATE TABLE IF NOT EXISTS evidence_holds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            reason TEXT NOT NULL,
            case_number TEXT,
            created_at REAL NOT NULL,
            created_by TEXT,
            released_at REAL,
            released_by TEXT
        );

        CREATE TABLE IF NOT EXISTS leo_accounts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            badge_number TEXT NOT NULL UNIQUE,
            agency TEXT NOT NULL,
            full_name TEXT NOT NULL,
            rank TEXT,
            email TEXT,
            phone TEXT,
            jurisdiction TEXT,
            password_hash TEXT,
            approved INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            last_login REAL
        );

        CREATE TABLE IF NOT EXISTS leo_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            badge_number TEXT,
            action TEXT NOT NULL,
            resource TEXT,
            ip_address TEXT,
            details TEXT
        );

        CREATE TABLE IF NOT EXISTS responder_tokens (
            token_id TEXT PRIMARY KEY,
            incident_id INTEGER NOT NULL,
            issued_at REAL NOT NULL,
            expires_at REAL NOT NULL,
            issued_to TEXT,
            revoked INTEGER NOT NULL DEFAULT 0,
            access_count INTEGER NOT NULL DEFAULT 0,
            last_accessed REAL,
            FOREIGN KEY (incident_id) REFERENCES incidents(id)
        );

        CREATE INDEX IF NOT EXISTS idx_tracks_camera ON tracks(camera);
        CREATE INDEX IF NOT EXISTS idx_tracks_active ON tracks(active);
        CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera);
        CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
        CREATE INDEX IF NOT EXISTS idx_behavior_track ON behavior_stats(track_id);
        CREATE INDEX IF NOT EXISTS idx_anomaly_camera ON anomaly_scores(camera);
        CREATE INDEX IF NOT EXISTS idx_incidents_camera ON incidents(camera);
        CREATE INDEX IF NOT EXISTS idx_incidents_start ON incidents(start_time);
        CREATE INDEX IF NOT EXISTS idx_narrations_incident ON incident_narrations(incident_id);
        CREATE INDEX IF NOT EXISTS idx_cases_number ON cases(case_number);
        CREATE INDEX IF NOT EXISTS idx_case_incidents_case ON case_incidents(case_id);
        CREATE INDEX IF NOT EXISTS idx_case_incidents_incident ON case_incidents(incident_id);
        CREATE INDEX IF NOT EXISTS idx_holds_camera ON evidence_holds(camera);
        CREATE INDEX IF NOT EXISTS idx_holds_active ON evidence_holds(released_at);
        CREATE INDEX IF NOT EXISTS idx_leo_badge ON leo_accounts(badge_number);
        CREATE INDEX IF NOT EXISTS idx_leo_audit_time ON leo_audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_responder_incident ON responder_tokens(incident_id);
    """)
    conn.close()


# --- Police module operations ---

def save_incident_police_data(incident_id: int, report_narrative: str = None,
                               tactical_brief: str = None, vehicle_details: str = None,
                               evidence_cert_path: str = None):
    """Upsert police-specific data for an incident."""
    conn = get_db()
    existing = conn.execute(
        "SELECT incident_id FROM incident_police WHERE incident_id = ?",
        (incident_id,)
    ).fetchone()
    now = time.time()
    if existing:
        updates = []
        params = []
        if report_narrative is not None:
            updates.append("report_narrative = ?")
            params.append(report_narrative)
        if tactical_brief is not None:
            updates.append("tactical_brief = ?")
            params.append(tactical_brief)
        if vehicle_details is not None:
            updates.append("vehicle_details = ?")
            params.append(vehicle_details)
        if evidence_cert_path is not None:
            updates.append("evidence_cert_path = ?")
            params.append(evidence_cert_path)
        if updates:
            updates.append("generated_at = ?")
            params.append(now)
            params.append(incident_id)
            conn.execute(
                f"UPDATE incident_police SET {', '.join(updates)} WHERE incident_id = ?",
                params,
            )
    else:
        conn.execute("""
            INSERT INTO incident_police
            (incident_id, report_narrative, tactical_brief, vehicle_details,
             evidence_cert_path, generated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (incident_id, report_narrative, tactical_brief, vehicle_details,
              evidence_cert_path, now))
    conn.commit()
    conn.close()


def get_incident_police_data(incident_id: int) -> dict:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM incident_police WHERE incident_id = ?",
        (incident_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Case management ---

def create_case(case_number: str, agency: str, officer_name: str = None,
                officer_badge: str = None, description: str = None) -> int:
    conn = get_db()
    cursor = conn.execute("""
        INSERT OR IGNORE INTO cases
        (case_number, agency, officer_name, officer_badge, opened_at, description)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (case_number, agency, officer_name, officer_badge, time.time(), description))
    case_id = cursor.lastrowid
    if not case_id:
        row = conn.execute(
            "SELECT id FROM cases WHERE case_number = ?", (case_number,)
        ).fetchone()
        case_id = row["id"] if row else None
    conn.commit()
    conn.close()
    return case_id


def bind_incident_to_case(case_id: int, incident_id: int, bound_by: str = None):
    conn = get_db()
    conn.execute("""
        INSERT OR IGNORE INTO case_incidents
        (case_id, incident_id, bound_at, bound_by)
        VALUES (?, ?, ?, ?)
    """, (case_id, incident_id, time.time(), bound_by))
    conn.commit()
    conn.close()


def get_case(case_number: str) -> dict:
    conn = get_db()
    case = conn.execute(
        "SELECT * FROM cases WHERE case_number = ?", (case_number,)
    ).fetchone()
    if not case:
        conn.close()
        return None
    case_id = case["id"]
    incidents = conn.execute("""
        SELECT i.* FROM incidents i
        JOIN case_incidents ci ON ci.incident_id = i.id
        WHERE ci.case_id = ?
        ORDER BY i.start_time
    """, (case_id,)).fetchall()
    conn.close()
    result = dict(case)
    result["incidents"] = [dict(i) for i in incidents]
    return result


def list_cases(agency: str = None, status: str = "open", limit: int = 100) -> list:
    conn = get_db()
    query = "SELECT * FROM cases WHERE 1=1"
    params = []
    if agency:
        query += " AND agency = ?"
        params.append(agency)
    if status:
        query += " AND status = ?"
        params.append(status)
    query += " ORDER BY opened_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Evidence holds ---

def create_evidence_hold(camera: str, start_time: float, end_time: float,
                          reason: str, case_number: str = None,
                          created_by: str = None) -> int:
    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO evidence_holds
        (camera, start_time, end_time, reason, case_number, created_at, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (camera, start_time, end_time, reason, case_number, time.time(), created_by))
    hold_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return hold_id


def release_evidence_hold(hold_id: int, released_by: str = None):
    conn = get_db()
    conn.execute("""
        UPDATE evidence_holds SET released_at = ?, released_by = ? WHERE id = ?
    """, (time.time(), released_by, hold_id))
    conn.commit()
    conn.close()


def get_active_holds(camera: str = None) -> list:
    conn = get_db()
    query = "SELECT * FROM evidence_holds WHERE released_at IS NULL"
    params = []
    if camera:
        query += " AND camera = ?"
        params.append(camera)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def is_timeframe_held(camera: str, start: float, end: float) -> bool:
    """Return True if ANY active hold overlaps the given window — blocks purge."""
    conn = get_db()
    row = conn.execute("""
        SELECT COUNT(*) as n FROM evidence_holds
        WHERE released_at IS NULL
          AND camera = ?
          AND start_time < ?
          AND end_time > ?
    """, (camera, end, start)).fetchone()
    conn.close()
    return (row["n"] or 0) > 0


# --- LEO accounts + audit ---

def log_leo_action(badge_number: str, action: str, resource: str = None,
                    ip_address: str = None, details: str = None):
    conn = get_db()
    conn.execute("""
        INSERT INTO leo_audit_log (timestamp, badge_number, action, resource, ip_address, details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time.time(), badge_number, action, resource, ip_address, details))
    conn.commit()
    conn.close()


def create_leo_account(badge_number: str, agency: str, full_name: str,
                        rank: str = None, email: str = None, phone: str = None,
                        jurisdiction: str = None, password_hash: str = None) -> int:
    conn = get_db()
    cursor = conn.execute("""
        INSERT OR IGNORE INTO leo_accounts
        (badge_number, agency, full_name, rank, email, phone, jurisdiction,
         password_hash, approved, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
    """, (badge_number, agency, full_name, rank, email, phone, jurisdiction,
          password_hash, time.time()))
    acc_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return acc_id


def get_leo_by_badge(badge_number: str) -> dict:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM leo_accounts WHERE badge_number = ?", (badge_number,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Responder tokens ---

def save_responder_token(token_id: str, incident_id: int, expires_at: float,
                          issued_to: str = None):
    conn = get_db()
    conn.execute("""
        INSERT OR REPLACE INTO responder_tokens
        (token_id, incident_id, issued_at, expires_at, issued_to)
        VALUES (?, ?, ?, ?, ?)
    """, (token_id, incident_id, time.time(), expires_at, issued_to))
    conn.commit()
    conn.close()


def get_responder_token(token_id: str) -> dict:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM responder_tokens WHERE token_id = ?", (token_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def touch_responder_token(token_id: str):
    conn = get_db()
    conn.execute("""
        UPDATE responder_tokens
        SET access_count = access_count + 1, last_accessed = ?
        WHERE token_id = ?
    """, (time.time(), token_id))
    conn.commit()
    conn.close()


def revoke_responder_token(token_id: str):
    conn = get_db()
    conn.execute(
        "UPDATE responder_tokens SET revoked = 1 WHERE token_id = ?", (token_id,)
    )
    conn.commit()
    conn.close()


# --- Incident operations ---

def create_incident(camera: str, trigger_source: str, track_id: str = None,
                    snapshot_path: str = None) -> int:
    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO incidents (camera, start_time, trigger_source, track_id, snapshot_path)
        VALUES (?, ?, ?, ?, ?)
    """, (camera, time.time(), trigger_source, track_id, snapshot_path))
    incident_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return incident_id


def update_incident(incident_id: int, **fields):
    if not fields:
        return
    conn = get_db()
    keys = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [incident_id]
    conn.execute(f"UPDATE incidents SET {keys} WHERE id = ?", values)
    conn.commit()
    conn.close()


def add_narration(incident_id: int, threat: int, narration: str, action: str = None):
    conn = get_db()
    conn.execute("""
        INSERT INTO incident_narrations (incident_id, timestamp, threat, narration, action)
        VALUES (?, ?, ?, ?, ?)
    """, (incident_id, time.time(), threat, narration, action))
    conn.commit()
    conn.close()


def end_incident(incident_id: int, resolution: str):
    conn = get_db()
    conn.execute("""
        UPDATE incidents SET end_time = ?, resolution = ? WHERE id = ?
    """, (time.time(), resolution, incident_id))
    conn.commit()
    conn.close()


def get_incidents(camera: str = None, since: float = None, limit: int = 50) -> list:
    conn = get_db()
    query = "SELECT * FROM incidents WHERE 1=1"
    params = []
    if camera:
        query += " AND camera = ?"
        params.append(camera)
    if since:
        query += " AND start_time > ?"
        params.append(since)
    query += " ORDER BY start_time DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_incident(incident_id: int) -> dict:
    conn = get_db()
    inc = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if not inc:
        conn.close()
        return None
    narrations = conn.execute(
        "SELECT * FROM incident_narrations WHERE incident_id = ? ORDER BY timestamp",
        (incident_id,)
    ).fetchall()
    conn.close()
    result = dict(inc)
    result["narrations"] = [dict(n) for n in narrations]
    return result


# --- Track operations ---

def upsert_track(track_id: str, camera: str, label: str, x: float, y: float,
                 w: float, h: float, zones: list = None, reid_id: str = None):
    now = time.time()
    conn = get_db()
    existing = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()

    if existing:
        positions = json.loads(existing["positions"])
        positions.append({"x": x, "y": y, "w": w, "h": h, "t": now})
        # Keep last 500 positions max
        if len(positions) > 500:
            positions = positions[-500:]

        zone_list = json.loads(existing["zones"])
        if zones:
            for z in zones:
                if z not in zone_list:
                    zone_list.append(z)

        conn.execute("""
            UPDATE tracks SET last_seen = ?, positions = ?, zones = ?,
                              reid_id = COALESCE(?, reid_id)
            WHERE id = ?
        """, (now, json.dumps(positions), json.dumps(zone_list), reid_id, track_id))
    else:
        positions = [{"x": x, "y": y, "w": w, "h": h, "t": now}]
        conn.execute("""
            INSERT INTO tracks (id, camera, label, first_seen, last_seen, positions, zones, reid_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (track_id, camera, label, now, now, json.dumps(positions),
              json.dumps(zones or []), reid_id))

    conn.commit()
    conn.close()


def deactivate_track(track_id: str):
    conn = get_db()
    conn.execute("UPDATE tracks SET active = 0 WHERE id = ?", (track_id,))
    conn.commit()
    conn.close()


def get_active_tracks(camera: str = None) -> list:
    conn = get_db()
    if camera:
        rows = conn.execute(
            "SELECT * FROM tracks WHERE active = 1 AND camera = ? ORDER BY last_seen DESC",
            (camera,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM tracks WHERE active = 1 ORDER BY last_seen DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_track(track_id: str) -> dict:
    conn = get_db()
    row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# --- Event operations ---

def insert_event(camera: str, event_type: str, severity: str = "info",
                 track_id: str = None, data: dict = None) -> int:
    conn = get_db()
    cursor = conn.execute("""
        INSERT INTO events (timestamp, camera, event_type, severity, track_id, data)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time.time(), camera, event_type, severity, track_id, json.dumps(data or {})))
    event_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return event_id


def get_events(camera: str = None, event_type: str = None,
               since: float = None, limit: int = 100) -> list:
    conn = get_db()
    query = "SELECT * FROM events WHERE 1=1"
    params = []
    if camera:
        query += " AND camera = ?"
        params.append(camera)
    if event_type:
        query += " AND event_type = ?"
        params.append(event_type)
    if since:
        query += " AND timestamp > ?"
        params.append(since)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# --- Behavior stats ---

def insert_behavior(camera: str, track_id: str, behavior_type: str,
                    value: float = None, data: dict = None):
    conn = get_db()
    conn.execute("""
        INSERT INTO behavior_stats (timestamp, camera, track_id, behavior_type, value, data)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (time.time(), camera, track_id, behavior_type, value, json.dumps(data or {})))
    conn.commit()
    conn.close()


# --- Anomaly scores ---

def insert_anomaly_score(camera: str, score: float, frame_path: str = None, data: dict = None):
    conn = get_db()
    conn.execute("""
        INSERT INTO anomaly_scores (timestamp, camera, score, frame_path, data)
        VALUES (?, ?, ?, ?, ?)
    """, (time.time(), camera, score, frame_path, json.dumps(data or {})))
    conn.commit()
    conn.close()
