"""Extract training frames from Frigate recordings and train anomaly models for all cameras.

Pulls frames from daytime recordings (8am-6pm) across multiple days to build
a robust "normal" baseline per camera. Then trains PatchCore models.

Usage: python train_all.py
"""

import os
import sys
import subprocess
import random
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("train_all")

RECORDINGS_DIR = Path("/home/acme-corpai/security-monitor/frigate-storage/recordings")
TRAINING_DIR = Path("/home/acme-corpai/pylox-v2/data/training_frames")
CAMERAS = ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6", "cam7", "cam8", "cam9"]
FRAMES_PER_CAMERA = 300
DAYTIME_HOURS = range(8, 18)  # 8am to 6pm


def extract_frames():
    """Extract training frames from recordings."""
    # Get available dates
    dates = sorted([d.name for d in RECORDINGS_DIR.iterdir() if d.is_dir()])
    logger.info(f"Found {len(dates)} days of recordings: {dates[0]} to {dates[-1]}")

    for camera in CAMERAS:
        cam_dir = TRAINING_DIR / camera
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already have enough frames
        existing = len(list(cam_dir.glob("*.jpg")))
        if existing >= FRAMES_PER_CAMERA:
            logger.info(f"{camera}: Already have {existing} frames, skipping extraction")
            continue

        logger.info(f"{camera}: Extracting {FRAMES_PER_CAMERA} frames...")

        # Collect all daytime video files for this camera
        video_files = []
        for date in dates:
            for hour in DAYTIME_HOURS:
                hour_dir = RECORDINGS_DIR / date / f"{hour:02d}" / camera
                if hour_dir.exists():
                    for mp4 in hour_dir.glob("*.mp4"):
                        video_files.append(mp4)

        if not video_files:
            logger.warning(f"{camera}: No daytime recordings found, trying all hours")
            for date in dates:
                for hour_dir in (RECORDINGS_DIR / date).iterdir():
                    cam_path = hour_dir / camera
                    if cam_path.exists():
                        for mp4 in cam_path.glob("*.mp4"):
                            video_files.append(mp4)

        if not video_files:
            logger.error(f"{camera}: No recordings found at all, skipping")
            continue

        logger.info(f"{camera}: Found {len(video_files)} video segments")

        # Sample random segments and extract 1 frame from each
        random.shuffle(video_files)
        selected = video_files[:FRAMES_PER_CAMERA]

        extracted = 0
        for i, video_path in enumerate(selected):
            output_path = cam_dir / f"frame_{i:04d}.jpg"
            try:
                # Extract middle frame from each segment using ffmpeg
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-sseof", "-2",  # 2 seconds before end (middle-ish)
                        "-i", str(video_path),
                        "-vframes", "1",
                        "-q:v", "2",
                        "-vf", "scale=480:-1",  # resize to 480px width
                        str(output_path),
                    ],
                    timeout=10,
                    capture_output=True,
                )
                if output_path.exists() and output_path.stat().st_size > 1000:
                    extracted += 1
                else:
                    output_path.unlink(missing_ok=True)
            except Exception as e:
                pass

            if extracted % 50 == 0 and extracted > 0:
                logger.info(f"  {camera}: {extracted}/{FRAMES_PER_CAMERA} frames extracted")

        logger.info(f"{camera}: Extracted {extracted} training frames")


def train_models():
    """Train anomaly models for all cameras with enough frames."""
    # Add engine to path
    sys.path.insert(0, str(Path(__file__).parent))

    for camera in CAMERAS:
        cam_dir = TRAINING_DIR / camera
        frame_count = len(list(cam_dir.glob("*.jpg"))) if cam_dir.exists() else 0

        if frame_count < 50:
            logger.warning(f"{camera}: Only {frame_count} frames, need 50+. Skipping training.")
            continue

        logger.info(f"{camera}: Training PatchCore model on {frame_count} frames...")

        try:
            from engine.anomaly.detector import AnomalyDetector
            model_path = AnomalyDetector.train_model(camera)
            logger.info(f"{camera}: Model saved to {model_path}")
        except Exception as e:
            logger.error(f"{camera}: Training failed: {e}")


def set_retention():
    """Configure Frigate to keep only 7 days of recordings."""
    config_path = Path("/home/acme-corpai/security-monitor/frigate-config.yml")
    logger.info("Setting Frigate retention to 7 days...")
    logger.info("NOTE: Edit frigate-config.yml manually to add:")
    logger.info("  record:")
    logger.info("    retain:")
    logger.info("      days: 7")
    logger.info("Then restart Frigate container.")


if __name__ == "__main__":
    logger.info("=== PYLOX V2 ANOMALY TRAINING PIPELINE ===")
    logger.info("")

    logger.info("Step 1: Extracting training frames from recordings...")
    extract_frames()

    logger.info("")
    logger.info("Step 2: Training anomaly models...")
    train_models()

    logger.info("")
    logger.info("Step 3: Storage retention recommendation")
    set_retention()

    logger.info("")
    logger.info("Done! Reload models with: curl -X POST localhost:3450/api/v2/anomaly/reload")
