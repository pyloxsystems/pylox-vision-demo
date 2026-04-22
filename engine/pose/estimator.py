"""Pose Estimator — extracts 3D body pose from person crops using MediaPipe.

Pipeline:
  1. Frigate detects person → bounding box crop
  2. MediaPipe Pose extracts 33 3D landmarks
  3. Clothing color extracted from upper/lower body regions
  4. Body data packaged for 3D twin rendering

Output per person:
  {
    "track_id": "...",
    "landmarks": [{x, y, z, visibility}, ...],  # 33 points
    "clothing": {"top": "#hex", "bottom": "#hex"},
    "pose_type": "standing" | "walking" | "sitting" | "running",
    "confidence": 0.85
  }
"""

import logging
import numpy as np
from typing import Optional
from PIL import Image
import mediapipe as mp

logger = logging.getLogger("pylox-v2.pose")

# MediaPipe landmark indices for body regions
UPPER_BODY = [11, 12, 13, 14, 23, 24]  # shoulders, elbows, hips
LOWER_BODY = [23, 24, 25, 26, 27, 28]  # hips, knees, ankles
HEAD = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Skeleton connections for rendering (pairs of landmark indices)
SKELETON_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left arm
    (11, 13), (13, 15),
    # Right arm
    (12, 14), (14, 16),
    # Left leg
    (23, 25), (25, 27),
    # Right leg
    (24, 26), (26, 28),
    # Head connections
    (0, 11), (0, 12),
]


class PoseEstimator:
    """Extracts 3D pose from person detection crops."""

    def __init__(self):
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._stats = {
            "frames_processed": 0,
            "poses_detected": 0,
            "failed": 0,
        }

    def estimate(self, frame: np.ndarray, bbox: dict = None) -> Optional[dict]:
        """Extract 3D pose from a frame (or crop if bbox provided).

        Args:
            frame: RGB image as numpy array
            bbox: Optional {x, y, w, h} bounding box to crop person

        Returns:
            Pose data dict or None if no pose detected
        """
        self._stats["frames_processed"] += 1

        try:
            # Crop to bounding box if provided
            if bbox:
                x, y, w, h = int(bbox["x"]), int(bbox["y"]), int(bbox["w"]), int(bbox["h"])
                # Add padding
                pad = int(max(w, h) * 0.1)
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad)
                y2 = min(frame.shape[0], y + h + pad)
                crop = frame[y1:y2, x1:x2]
            else:
                crop = frame
                x1, y1 = 0, 0

            if crop.size == 0:
                return None

            # Run MediaPipe Pose
            results = self.pose.process(crop)

            if not results.pose_world_landmarks:
                self._stats["failed"] += 1
                return None

            self._stats["poses_detected"] += 1

            # Extract 3D landmarks
            landmarks = []
            for lm in results.pose_world_landmarks.landmark:
                landmarks.append({
                    "x": round(lm.x, 4),
                    "y": round(lm.y, 4),
                    "z": round(lm.z, 4),
                    "visibility": round(lm.visibility, 3),
                })

            # Extract clothing colors
            clothing = self._extract_clothing_colors(crop, results.pose_landmarks)

            # Classify pose type
            pose_type = self._classify_pose(landmarks)

            # Average visibility as confidence
            confidence = np.mean([lm["visibility"] for lm in landmarks])

            return {
                "landmarks": landmarks,
                "skeleton": SKELETON_CONNECTIONS,
                "clothing": clothing,
                "pose_type": pose_type,
                "confidence": round(float(confidence), 3),
                "crop_offset": {"x": x1, "y": y1} if bbox else None,
            }

        except Exception as e:
            self._stats["failed"] += 1
            logger.error(f"Pose estimation error: {e}")
            return None

    def _extract_clothing_colors(self, crop: np.ndarray,
                                  pose_landmarks) -> dict:
        """Extract dominant colors from upper and lower body regions."""
        if not pose_landmarks:
            return {"top": "#666666", "bottom": "#444444"}

        h, w = crop.shape[:2]
        landmarks = pose_landmarks.landmark

        try:
            # Upper body region (between shoulders and hips)
            upper_y_min = int(min(landmarks[11].y, landmarks[12].y) * h)
            upper_y_max = int(min(landmarks[23].y, landmarks[24].y) * h)
            upper_x_min = int(min(landmarks[11].x, landmarks[12].x) * w)
            upper_x_max = int(max(landmarks[11].x, landmarks[12].x) * w)

            if upper_y_max > upper_y_min and upper_x_max > upper_x_min:
                upper_region = crop[
                    max(0, upper_y_min):min(h, upper_y_max),
                    max(0, upper_x_min):min(w, upper_x_max)
                ]
                top_color = self._dominant_color(upper_region)
            else:
                top_color = "#666666"

            # Lower body region (between hips and ankles)
            lower_y_min = int(min(landmarks[23].y, landmarks[24].y) * h)
            lower_y_max = int(max(landmarks[27].y, landmarks[28].y) * h)
            lower_x_min = int(min(landmarks[23].x, landmarks[24].x) * w)
            lower_x_max = int(max(landmarks[23].x, landmarks[24].x) * w)

            # Widen the lower body region a bit
            lower_pad = int((lower_x_max - lower_x_min) * 0.3)
            lower_x_min = max(0, lower_x_min - lower_pad)
            lower_x_max = min(w, lower_x_max + lower_pad)

            if lower_y_max > lower_y_min and lower_x_max > lower_x_min:
                lower_region = crop[
                    max(0, lower_y_min):min(h, lower_y_max),
                    max(0, lower_x_min):min(w, lower_x_max)
                ]
                bottom_color = self._dominant_color(lower_region)
            else:
                bottom_color = "#444444"

            return {"top": top_color, "bottom": bottom_color}

        except Exception:
            return {"top": "#666666", "bottom": "#444444"}

    def _dominant_color(self, region: np.ndarray) -> str:
        """Get dominant color from an image region using k-means-like approach."""
        if region.size == 0 or region.shape[0] < 5 or region.shape[1] < 5:
            return "#666666"

        # Resize to small for speed
        small = np.array(Image.fromarray(region).resize((8, 8)))
        # Average color
        avg = small.reshape(-1, 3).mean(axis=0).astype(int)
        return f"#{avg[0]:02x}{avg[1]:02x}{avg[2]:02x}"

    def _classify_pose(self, landmarks: list) -> str:
        """Classify the detected pose type."""
        if len(landmarks) < 33:
            return "unknown"

        # Get key landmark positions
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        left_knee = landmarks[25]
        right_knee = landmarks[26]
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Hip-knee angle to detect sitting
        hip_y = (left_hip["y"] + right_hip["y"]) / 2
        knee_y = (left_knee["y"] + right_knee["y"]) / 2
        ankle_y = (left_ankle["y"] + right_ankle["y"]) / 2
        shoulder_y = (left_shoulder["y"] + right_shoulder["y"]) / 2

        # Sitting: knees bent significantly (hip-knee-ankle angle)
        hip_knee_diff = abs(hip_y - knee_y)
        knee_ankle_diff = abs(knee_y - ankle_y)

        if hip_knee_diff < 0.05 and knee_ankle_diff > 0.1:
            return "sitting"

        # Running: large stride (ankle separation)
        ankle_sep = abs(left_ankle["z"] - right_ankle["z"])
        knee_sep = abs(left_knee["z"] - right_knee["z"])
        if ankle_sep > 0.15 and knee_sep > 0.1:
            return "running"

        # Walking: moderate stride
        if ankle_sep > 0.05:
            return "walking"

        return "standing"

    def get_stats(self) -> dict:
        return {**self._stats}

    def close(self):
        """Release MediaPipe resources."""
        self.pose.close()
