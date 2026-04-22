"""Frame Scorer — grabs snapshots from Frigate and feeds them to Anomalib.

Runs as a background task, pulling snapshots for active cameras at a
configurable interval. Uses the GPU queue for inference priority.
"""

import asyncio
import time
import logging
import io
from typing import Optional

import numpy as np
from PIL import Image

from engine.config import FRIGATE_API, CAMERAS
from engine.anomaly.detector import AnomalyDetector
from engine.events.engine import EventEngine
from engine.gpu_queue import GPUQueue, Priority
from engine import database as db

logger = logging.getLogger("pylox-v2.anomaly.scorer")


class FrameScorer:
    """Periodically scores camera frames for anomalies."""

    def __init__(self, detector: AnomalyDetector, event_engine: EventEngine,
                 gpu_queue: GPUQueue, interval: float = 5.0):
        self.detector = detector
        self.event_engine = event_engine
        self.gpu_queue = gpu_queue
        self.interval = interval  # seconds between scoring rounds
        self._running = False
        self._task = None
        self._stats = {
            "rounds": 0,
            "frames_fetched": 0,
            "frames_scored": 0,
            "errors": 0,
        }

    async def start(self):
        """Start the periodic frame scoring loop."""
        if not self.detector.models:
            logger.info("No anomaly models loaded — frame scoring disabled")
            return

        self._running = True
        self._task = asyncio.create_task(self._scoring_loop())
        logger.info(f"Frame scorer started (interval: {self.interval}s, "
                    f"cameras: {list(self.detector.models.keys())})")

    async def stop(self):
        """Stop the frame scoring loop."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("Frame scorer stopped")

    async def _scoring_loop(self):
        """Main loop — score each camera with a model."""
        while self._running:
            try:
                self._stats["rounds"] += 1

                cameras = list(self.detector.models.keys())
                if self._stats["rounds"] <= 3:
                    logger.info(f"Scoring round {self._stats['rounds']}, cameras: {len(cameras)}")

                for camera in cameras:
                    if not self._running:
                        break

                    frame = await self._fetch_snapshot(camera)
                    if frame is None:
                        continue

                    # Submit to GPU queue
                    try:
                        score = await self.gpu_queue.submit(
                            f"anomaly-{camera}",
                            self.detector.score_frame,
                            camera, frame,
                            priority=Priority.NORMAL,
                        )

                        if score is not None:
                            self._stats["frames_scored"] += 1

                            # Store score
                            db.insert_anomaly_score(camera, score)

                            # Fire event if anomalous
                            if score > self.detector.thresholds.get(camera, 0.5):
                                self.event_engine.process_anomaly(camera, score)
                                logger.warning(
                                    f"Anomaly detected on {camera}: score={score:.3f}"
                                )

                    except Exception as e:
                        self._stats["errors"] += 1
                        logger.error(f"Anomaly scoring error on {camera}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Frame scoring loop error: {e}")

            await asyncio.sleep(self.interval)

    async def _fetch_snapshot(self, camera: str) -> Optional[np.ndarray]:
        """Fetch latest snapshot from Frigate."""
        import urllib.request

        try:
            url = f"{FRIGATE_API}/api/{camera}/latest.jpg?h=480"
            req = urllib.request.urlopen(url, timeout=5)
            data = req.read()
            if len(data) > 1000:
                img = Image.open(io.BytesIO(data)).convert("RGB")
                self._stats["frames_fetched"] += 1
                return np.array(img)
            return None
        except Exception as e:
            logger.error(f"Snapshot fetch failed for {camera}: {e}")
            return None

    def get_stats(self) -> dict:
        return {**self._stats}

    async def score_single(self, camera: str) -> Optional[float]:
        """Score a single frame on demand (for API calls)."""
        frame = await self._fetch_snapshot(camera)
        if frame is None:
            return None

        return await self.gpu_queue.submit(
            f"anomaly-{camera}-manual",
            self.detector.score_frame,
            camera, frame,
            priority=Priority.HIGH,
        )
