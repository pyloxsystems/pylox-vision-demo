"""Camera-to-3D Coordinate Mapping — maps 2D pixel positions to 3D world coordinates.

Each camera has a known position and orientation in the 3D space.
Given a 2D detection (bounding box) from a camera, we project it to
the 3D ground plane to place the person in the twin.

For a proper calibration, we need:
  - Camera intrinsics (focal length, principal point)
  - Camera extrinsics (position, rotation in world space)
  - Ground plane equation (typically y=0)

For MVP, we use a simplified homography-based mapping:
  - Define 4+ reference points (2D pixel -> 3D world)
  - Compute homography matrix
  - Transform any 2D point to 3D ground plane
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("pylox-v2.spatial.mapping")


class CameraMapping:
    """Maps 2D pixel coordinates from a camera to 3D world coordinates."""

    def __init__(self):
        # Per-camera calibration data
        # camera_id -> { homography: 3x3 matrix, position: [x,y,z], fov: degrees }
        self.calibrations = {}

    def add_calibration(self, camera_id: str, reference_points: list,
                        camera_position: list, camera_fov: float = 90):
        """Add calibration for a camera using reference point pairs.

        Args:
            camera_id: Camera identifier
            reference_points: List of {"pixel": [x, y], "world": [x, z]} pairs
                              (at least 4 points for homography)
            camera_position: [x, y, z] position of camera in world space
            camera_fov: Horizontal field of view in degrees
        """
        if len(reference_points) < 4:
            raise ValueError("Need at least 4 reference points for homography")

        # Extract point pairs
        src_points = np.float32([p["pixel"] for p in reference_points])
        dst_points = np.float32([p["world"] for p in reference_points])

        # Compute homography (2D pixel -> 2D ground plane)
        H, status = self._compute_homography(src_points, dst_points)

        if H is None:
            raise ValueError("Failed to compute homography")

        self.calibrations[camera_id] = {
            "homography": H,
            "position": np.array(camera_position, dtype=np.float32),
            "fov": camera_fov,
            "ref_points": reference_points,
        }

        logger.info(f"Calibration added for {camera_id} with {len(reference_points)} points")

    def add_simple_calibration(self, camera_id: str, camera_position: list,
                                look_at: list, frame_width: int = 1920,
                                frame_height: int = 1080, coverage_meters: float = 10):
        """Add simplified calibration without reference points.

        Assumes camera looks at a rectangular ground area.
        Good enough for initial deployment.
        """
        # Simple linear mapping: pixel space -> ground plane
        # The camera covers an area of coverage_meters x coverage_meters
        cam_pos = np.array(camera_position, dtype=np.float32)
        look = np.array(look_at, dtype=np.float32)

        # Direction from camera to look-at point
        direction = look - cam_pos
        direction[1] = 0  # project to ground plane
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        # Create a simple affine mapping
        # pixel (0,0) -> world top-left, pixel (W,H) -> world bottom-right
        half = coverage_meters / 2

        # 4 corner reference points
        right = np.cross(direction, np.array([0, 1, 0]))
        if np.linalg.norm(right) > 0:
            right = right / np.linalg.norm(right)
        else:
            right = np.array([1, 0, 0])

        center_ground = look.copy()
        center_ground[1] = 0

        corners_world = [
            center_ground - right * half - direction * half,  # top-left
            center_ground + right * half - direction * half,  # top-right
            center_ground + right * half + direction * half,  # bottom-right
            center_ground - right * half + direction * half,  # bottom-left
        ]

        corners_pixel = [
            [0, 0],
            [frame_width, 0],
            [frame_width, frame_height],
            [0, frame_height],
        ]

        ref_points = [
            {"pixel": corners_pixel[i], "world": [corners_world[i][0], corners_world[i][2]]}
            for i in range(4)
        ]

        self.add_calibration(camera_id, ref_points, camera_position)

    def pixel_to_world(self, camera_id: str, pixel_x: float,
                        pixel_y: float) -> Optional[dict]:
        """Map a 2D pixel coordinate to 3D world position on the ground plane.

        Returns {"x": float, "y": 0, "z": float} or None if not calibrated.
        """
        cal = self.calibrations.get(camera_id)
        if not cal:
            return None

        H = cal["homography"]

        # Apply homography
        point = np.array([pixel_x, pixel_y, 1.0])
        result = H @ point
        result = result / result[2]  # normalize

        return {
            "x": round(float(result[0]), 3),
            "y": 0,  # ground plane
            "z": round(float(result[1]), 3),
        }

    def bbox_to_world(self, camera_id: str, x: float, y: float,
                       w: float, h: float) -> Optional[dict]:
        """Map a bounding box to world position.

        Uses the bottom-center of the bbox (feet position) for ground projection.
        """
        # Bottom-center of bounding box = feet position
        foot_x = x + w / 2
        foot_y = y + h

        world_pos = self.pixel_to_world(camera_id, foot_x, foot_y)
        if not world_pos:
            return None

        # Estimate height from bbox size and camera distance
        # This is approximate — proper calibration would be more accurate
        world_pos["estimated_height"] = round(h / 200.0, 2)  # rough estimate

        return world_pos

    def get_calibration(self, camera_id: str) -> Optional[dict]:
        """Get calibration data for a camera."""
        cal = self.calibrations.get(camera_id)
        if not cal:
            return None
        return {
            "camera_id": camera_id,
            "position": cal["position"].tolist(),
            "fov": cal["fov"],
            "ref_points": cal["ref_points"],
            "calibrated": True,
        }

    def get_all_calibrations(self) -> dict:
        """Get all camera calibrations."""
        return {
            cam_id: self.get_calibration(cam_id)
            for cam_id in self.calibrations
        }

    def _compute_homography(self, src_points: np.ndarray,
                             dst_points: np.ndarray):
        """Compute homography matrix using DLT (Direct Linear Transform)."""
        if len(src_points) < 4:
            return None, None

        n = len(src_points)
        A = np.zeros((2 * n, 9))

        for i in range(n):
            x, y = src_points[i]
            u, v = dst_points[i]
            A[2*i] = [-x, -y, -1, 0, 0, 0, x*u, y*u, u]
            A[2*i+1] = [0, 0, 0, -x, -y, -1, x*v, y*v, v]

        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H = H / H[2, 2]

        return H, np.ones(n)
