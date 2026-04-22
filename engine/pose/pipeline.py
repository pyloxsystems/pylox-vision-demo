"""Pose Pipeline — orchestrates pose estimation for tracked persons.

Fetches person crops from Frigate snapshots, runs MediaPipe pose estimation,
builds body meshes, and broadcasts to connected WebSocket clients for
3D twin rendering.

Runs as a background task, processing persons detected in the last N seconds.
"""

import io
import time
import asyncio
import logging
from typing import Optional

import numpy as np
from PIL import Image

from engine.config import FRIGATE_API
from engine.pose.estimator import PoseEstimator
from engine.pose.body_mesh import build_body_mesh
from engine.gpu_queue import GPUQueue, Priority
from engine import database as db

logger = logging.getLogger("pylox-v2.pose.pipeline")


class PosePipeline:
    """Processes active tracks through pose estimation."""

    def __init__(self, gpu_queue: GPUQueue, on_pose_update=None):
        self.gpu_queue = gpu_queue
        self.on_pose_update = on_pose_update
        self.estimator = PoseEstimator()
        self._running = False
        self._task = None
        self._pose_cache = {}  # track_id -> {pose_data, timestamp}
        self._cache_ttl = 2.0  # seconds before re-estimating
        self._stats = {
            "poses_estimated": 0,
            "meshes_built": 0,
            "cache_hits": 0,
            "errors": 0,
        }

    async def start(self):
        """Start the pose pipeline background loop."""
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("Pose pipeline started")

    async def stop(self):
        """Stop the pose pipeline."""
        self._running = False
        if self._task:
            self._task.cancel()
        self.estimator.close()
        logger.info("Pose pipeline stopped")

    async def _process_loop(self):
        """Main loop — estimate pose for all active tracks."""
        while self._running:
            try:
                # Get active tracks
                active_tracks = db.get_active_tracks()

                for track in active_tracks:
                    if not self._running:
                        break

                    track_id = track["id"]
                    camera = track["camera"]
                    label = track["label"]

                    # Only process person tracks
                    if label != "person":
                        continue

                    # Check cache
                    cached = self._pose_cache.get(track_id)
                    if cached and (time.time() - cached["timestamp"]) < self._cache_ttl:
                        self._stats["cache_hits"] += 1
                        continue

                    # Fetch snapshot and estimate pose
                    try:
                        pose_data = await self._estimate_for_track(track_id, camera, track)
                        if pose_data:
                            self._stats["poses_estimated"] += 1

                            # Build mesh
                            positions = track.get("positions", [])
                            if isinstance(positions, str):
                                import json
                                positions = json.loads(positions)

                            world_pos = None
                            if positions:
                                last = positions[-1]
                                # Convert pixel position to normalized 3D world position
                                world_pos = {
                                    "x": last.get("x", 0) / 1920 * 10 - 5,  # map to -5..5
                                    "y": 0,  # ground plane
                                    "z": last.get("y", 0) / 1080 * 10 - 5,
                                }

                            mesh = build_body_mesh(
                                pose_data,
                                scale=1.7,  # ~1.7m human height
                                position=world_pos,
                            )

                            if mesh:
                                self._stats["meshes_built"] += 1

                                # Cache it
                                self._pose_cache[track_id] = {
                                    "pose": pose_data,
                                    "mesh": mesh,
                                    "timestamp": time.time(),
                                }

                                # Broadcast for 3D twin
                                if self.on_pose_update:
                                    self.on_pose_update({
                                        "source": "pose",
                                        "track_id": track_id,
                                        "camera": camera,
                                        "mesh": mesh,
                                    })

                    except Exception as e:
                        self._stats["errors"] += 1
                        logger.error(f"Pose estimation error for {track_id}: {e}")

                # Cleanup stale cache entries
                now = time.time()
                stale = [k for k, v in self._pose_cache.items()
                         if now - v["timestamp"] > 30]
                for k in stale:
                    del self._pose_cache[k]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Pose pipeline error: {e}")

            await asyncio.sleep(1.0)  # Process every second

    async def _estimate_for_track(self, track_id: str, camera: str,
                                   track: dict) -> Optional[dict]:
        """Fetch snapshot and run pose estimation for a track."""
        import aiohttp

        # Get the latest snapshot from Frigate for this person
        try:
            url = f"{FRIGATE_API}/api/events/{track_id}/snapshot.jpg"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status != 200:
                        # Fall back to camera latest
                        url = f"{FRIGATE_API}/api/{camera}/latest.jpg?h=720"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp2:
                            if resp2.status != 200:
                                return None
                            data = await resp2.read()
                    else:
                        data = await resp.read()

            img = Image.open(io.BytesIO(data)).convert("RGB")
            frame = np.array(img)

            # Run pose estimation via GPU queue
            pose_data = await self.gpu_queue.submit(
                f"pose-{track_id[:8]}",
                self.estimator.estimate,
                frame,
                priority=Priority.HIGH,
            )

            return pose_data

        except Exception as e:
            logger.debug(f"Snapshot fetch failed for {track_id}: {e}")
            return None

    def get_cached_pose(self, track_id: str) -> Optional[dict]:
        """Get cached pose for a track (for API)."""
        cached = self._pose_cache.get(track_id)
        if cached:
            return cached["mesh"]
        return None

    def get_all_poses(self) -> dict:
        """Get all cached poses (for 3D twin initial load)."""
        return {
            track_id: data["mesh"]
            for track_id, data in self._pose_cache.items()
            if data.get("mesh")
        }

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "cached_poses": len(self._pose_cache),
            "estimator": self.estimator.get_stats(),
        }
