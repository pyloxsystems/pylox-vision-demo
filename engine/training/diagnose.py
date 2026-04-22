"""YOLO fine-tuning readiness diagnostic.

Reads training.json, v2.db, frigate.db, and the Frigate clips directory
to report exactly how much labeled data we have available for fine-tuning
a per-site YOLO model.

Run with:
    python3 /home/acme-corpai/pylox-v2/engine/training/diagnose.py

Or as a module:
    python3 -m engine.training.diagnose (from pylox-v2/)
"""

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

# Known paths on this Spark
PYLOX_V2_DB = Path("/home/acme-corpai/pylox-v2/data/v2.db")
TRAINING_JSON = Path("/home/acme-corpai/pylox-v2/data/training/default/training.json")
FRIGATE_DB = Path("/tmp/pylox-training/frigate.db")  # Copied from container
FRIGATE_CLIPS = Path("/home/acme-corpai/security-monitor/frigate-storage/clips")


def header(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def read_training_json():
    header("TrainingManager (prompt-learning layer)")
    if not TRAINING_JSON.exists():
        print(f"  MISSING: {TRAINING_JSON}")
        return {}
    data = json.loads(TRAINING_JSON.read_text())
    site = data.get("site", {})
    cameras = data.get("cameras", {})
    learned = data.get("learned", [])
    stats = data.get("stats", {})

    print(f"  Site: {site.get('name') or '(unnamed)'} ({site.get('type') or 'unknown type'})")
    print(f"  Cameras with data: {len(cameras)}")

    fp_by_cam = {}
    kp_by_cam = {}
    for cam_id, cam in cameras.items():
        fps = cam.get("false_positives", [])
        kps = cam.get("known_people", [])
        fp_by_cam[cam_id] = len(fps)
        kp_by_cam[cam_id] = len(kps)
        if fps or kps:
            print(f"    {cam_id}: {len(fps)} FPs, {len(kps)} known people")

    print(f"  Site-wide learned patterns: {len(learned)}")
    print(f"  Stats: {stats}")
    return {"fp_by_cam": fp_by_cam, "kp_by_cam": kp_by_cam, "cameras": list(cameras.keys())}


def read_pylox_v2_db():
    header("Pylox V2 database (watch session history)")
    if not PYLOX_V2_DB.exists():
        print(f"  MISSING: {PYLOX_V2_DB}")
        return {}
    conn = sqlite3.connect(str(PYLOX_V2_DB))
    c = conn.cursor()

    # Incidents: watch sessions that actually opened
    c.execute("SELECT COUNT(*) FROM incidents")
    total_incidents = c.fetchone()[0]

    c.execute("""
        SELECT camera, resolution, COUNT(*)
        FROM incidents
        GROUP BY camera, resolution
        ORDER BY camera, resolution
    """)
    incidents_by = c.fetchall()

    # Narrations: Gemini's per-tick verdicts inside sessions
    c.execute("SELECT COUNT(*) FROM incident_narrations")
    total_narrations = c.fetchone()[0]

    # Events: Pylox V2 internal rule/anomaly events
    c.execute("SELECT COUNT(*) FROM events")
    total_events = c.fetchone()[0]

    c.execute("""
        SELECT camera, event_type, severity, COUNT(*)
        FROM events
        GROUP BY camera, event_type, severity
        ORDER BY COUNT(*) DESC
        LIMIT 15
    """)
    top_events = c.fetchall()

    # Tracks: Frigate object detection tracks ingested into V2
    c.execute("SELECT COUNT(*) FROM tracks")
    total_tracks = c.fetchone()[0]

    print(f"  Total incidents (watch sessions opened): {total_incidents}")
    print(f"  Total narrations (Gemini per-tick verdicts): {total_narrations}")
    print(f"  Total V2 internal events (rule/anomaly triggers): {total_events}")
    print(f"  Total tracks (Frigate object tracks ingested): {total_tracks}")
    print()
    print("  Incident breakdown by camera + resolution:")
    for cam, res, n in incidents_by:
        print(f"    {cam:10s} {str(res):20s} {n}")
    print()
    print("  Top 15 V2 event counts (camera, event_type, severity):")
    for cam, et, sev, n in top_events:
        print(f"    {cam:10s} {et:30s} {sev:10s} {n}")

    conn.close()
    return {
        "incidents": total_incidents,
        "narrations": total_narrations,
        "events": total_events,
        "tracks": total_tracks,
        "top_events": top_events,
    }


def read_frigate_db():
    header("Frigate database (raw YOLO detections)")
    if not FRIGATE_DB.exists():
        print(f"  MISSING: {FRIGATE_DB} (copy with: docker cp frigate:/config/frigate.db {FRIGATE_DB})")
        return {}
    conn = sqlite3.connect(str(FRIGATE_DB))
    c = conn.cursor()

    c.execute("""
        SELECT COUNT(*) FROM event
        WHERE start_time > strftime('%s', 'now', '-7 days')
    """)
    last7 = c.fetchone()[0]

    c.execute("""
        SELECT camera, label, COUNT(*)
        FROM event
        WHERE start_time > strftime('%s', 'now', '-7 days')
        GROUP BY camera, label
        ORDER BY COUNT(*) DESC
    """)
    breakdown = c.fetchall()

    # How many have has_snapshot = 1
    c.execute("""
        SELECT has_snapshot, COUNT(*) FROM event
        WHERE start_time > strftime('%s', 'now', '-7 days')
        GROUP BY has_snapshot
    """)
    snap_counts = dict(c.fetchall())

    # Confidence distribution (using top_score where available)
    c.execute("""
        SELECT
            CASE
                WHEN top_score IS NULL THEN 'no_score'
                WHEN top_score >= 0.85 THEN 'high_>=0.85'
                WHEN top_score >= 0.5 THEN 'mid_0.5-0.85'
                ELSE 'low_<0.5'
            END AS bucket,
            COUNT(*)
        FROM event
        WHERE start_time > strftime('%s', 'now', '-7 days')
        GROUP BY bucket
    """)
    confidence_buckets = dict(c.fetchall())

    print(f"  Events in last 7 days: {last7}")
    print(f"  Snapshot presence: {snap_counts}")
    print(f"  Confidence distribution: {confidence_buckets}")
    print()
    print("  Per-camera/label breakdown:")
    for cam, lbl, n in breakdown:
        print(f"    {cam:10s} {lbl:12s} {n}")

    conn.close()
    return {
        "last7": last7,
        "breakdown": breakdown,
        "snap_counts": snap_counts,
        "confidence": confidence_buckets,
    }


def read_clip_files():
    header("Frigate clips on disk")
    if not FRIGATE_CLIPS.exists():
        print(f"  MISSING: {FRIGATE_CLIPS}")
        return {}

    jpgs = list(FRIGATE_CLIPS.glob("cam*.jpg"))
    jpg_by_cam = Counter()
    for p in jpgs:
        cam = p.name.split("-", 1)[0]
        jpg_by_cam[cam] += 1

    total_size = sum(p.stat().st_size for p in jpgs[:5000])  # sample to keep fast
    sample = len(jpgs[:5000])
    avg_kb = (total_size / sample / 1024) if sample else 0

    print(f"  Total JPG snapshots: {len(jpgs)}")
    print(f"  Average size: {avg_kb:.1f} KB (from first {sample} files)")
    print()
    print("  By camera:")
    for cam in sorted(jpg_by_cam.keys()):
        print(f"    {cam}: {jpg_by_cam[cam]} snapshots")

    return {"total_jpgs": len(jpgs), "by_cam": dict(jpg_by_cam)}


def readiness_assessment(tj, v2, frig, clips):
    header("YOLO fine-tune readiness assessment")

    total_available_frames = clips.get("total_jpgs", 0)
    person_events = sum(n for cam, lbl, n in frig.get("breakdown", []) if lbl == "person")
    car_events = sum(n for cam, lbl, n in frig.get("breakdown", []) if lbl == "car")
    text_fps = sum(tj.get("fp_by_cam", {}).values())

    # A reasonable training dataset target: ~2000 labeled frames balanced across cameras
    TARGET = 2000

    print(f"  Available raw material:")
    print(f"    - {total_available_frames} snapshot files on disk")
    print(f"    - {person_events} person events in last 7 days")
    print(f"    - {car_events} car events in last 7 days")
    print(f"    - {text_fps} text-labeled false positives (from training.json)")
    print(f"    - {v2.get('narrations', 0)} Gemini per-tick narrations (direct FP verdicts)")
    print()

    enough_data = total_available_frames >= TARGET * 2
    has_seed_labels = text_fps > 0 or v2.get("narrations", 0) > 0

    if enough_data:
        print(f"  [OK]   Enough snapshot volume for a {TARGET}-frame training set")
    else:
        print(f"  [WARN] Only {total_available_frames} snapshots, target is {TARGET*2}+ for robust training")

    if has_seed_labels:
        print(f"  [OK]   Has seed labels from TrainingManager + V2 narrations")
    else:
        print(f"  [WARN] No seed labels — qwen3-vl would label from scratch")

    print()
    print("  Recommended next steps:")
    print("    1. Sample ~2000 frames balanced across (camera, label, confidence bucket)")
    print("    2. Use training.json FP text + narrations as seed labels")
    print("    3. Batch through qwen3-vl:32b for the rest")
    print("    4. Human spot-check 100 random samples before training")
    print("    5. Fine-tune YOLOv8n on the Spark GPU (~30 min)")
    print("    6. Export ONNX, hot-swap into Frigate")
    print("    7. Measure FP rate before/after over 48h")


def main():
    print("Pylox YOLO fine-tune readiness diagnostic")
    print(f"  Training JSON: {TRAINING_JSON}")
    print(f"  Pylox V2 DB:   {PYLOX_V2_DB}")
    print(f"  Frigate DB:    {FRIGATE_DB}")
    print(f"  Clips dir:     {FRIGATE_CLIPS}")

    tj = read_training_json()
    v2 = read_pylox_v2_db()
    frig = read_frigate_db()
    clips = read_clip_files()
    readiness_assessment(tj, v2, frig, clips)

    print()
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
