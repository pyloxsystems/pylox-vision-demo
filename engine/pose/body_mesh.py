"""Body Mesh Builder — converts MediaPipe landmarks to Three.js-renderable body data.

Instead of full SMPL (which requires licensed model files), we build a
simplified body mesh from the 33 MediaPipe landmarks:
  - Spheres at joints
  - Cylinders connecting joints (skeleton)
  - Color-coded by clothing color
  - Head as a sphere with face blurred/anonymized

The output is a JSON-serializable structure that Three.js can render directly.
"""

import math
import logging

logger = logging.getLogger("pylox-v2.pose.mesh")

# Joint radii for rendering (proportional to body)
JOINT_RADII = {
    0: 0.12,   # nose (head center)
    11: 0.06,  # left shoulder
    12: 0.06,  # right shoulder
    13: 0.04,  # left elbow
    14: 0.04,  # right elbow
    15: 0.03,  # left wrist
    16: 0.03,  # right wrist
    23: 0.06,  # left hip
    24: 0.06,  # right hip
    25: 0.05,  # left knee
    26: 0.05,  # right knee
    27: 0.04,  # left ankle
    28: 0.04,  # right ankle
}

# Limb thickness for cylinder rendering
LIMB_THICKNESS = {
    (11, 12): 0.07,  # shoulders (torso top)
    (23, 24): 0.07,  # hips (torso bottom)
    (11, 23): 0.06,  # left torso
    (12, 24): 0.06,  # right torso
    (11, 13): 0.04,  # left upper arm
    (13, 15): 0.03,  # left forearm
    (12, 14): 0.04,  # right upper arm
    (14, 16): 0.03,  # right forearm
    (23, 25): 0.05,  # left thigh
    (25, 27): 0.04,  # left shin
    (24, 26): 0.05,  # right thigh
    (26, 28): 0.04,  # right shin
    (0, 11): 0.04,   # neck left
    (0, 12): 0.04,   # neck right
}

# Which limbs belong to upper vs lower body (for clothing colors)
UPPER_LIMBS = {(11, 12), (23, 24), (11, 23), (12, 24), (11, 13), (13, 15), (12, 14), (14, 16), (0, 11), (0, 12)}
LOWER_LIMBS = {(23, 25), (25, 27), (24, 26), (26, 28)}


def build_body_mesh(pose_data: dict, scale: float = 1.0,
                     position: dict = None) -> dict:
    """Build a renderable body mesh from pose estimation data.

    Args:
        pose_data: Output from PoseEstimator.estimate()
        scale: Scale factor for the body
        position: {x, y, z} world position offset

    Returns:
        Three.js-renderable mesh data
    """
    landmarks = pose_data.get("landmarks", [])
    clothing = pose_data.get("clothing", {"top": "#666666", "bottom": "#444444"})
    pose_type = pose_data.get("pose_type", "standing")

    if len(landmarks) < 33:
        return None

    pos = position or {"x": 0, "y": 0, "z": 0}

    # Build joints (spheres)
    joints = []
    for idx, radius in JOINT_RADII.items():
        lm = landmarks[idx]
        if lm["visibility"] < 0.3:
            continue

        # Determine color based on body part
        if idx == 0:
            color = "#ddccbb"  # skin tone for head
        elif idx in [11, 12, 13, 14, 15, 16]:
            color = clothing["top"]
        else:
            color = clothing["bottom"]

        joints.append({
            "index": idx,
            "position": {
                "x": round((lm["x"] * scale) + pos["x"], 4),
                "y": round((-lm["y"] * scale) + pos["y"], 4),  # flip Y for 3D
                "z": round((lm["z"] * scale) + pos["z"], 4),
            },
            "radius": round(radius * scale, 4),
            "color": color,
        })

    # Build limbs (cylinders)
    limbs = []
    for (i, j), thickness in LIMB_THICKNESS.items():
        lm_a = landmarks[i]
        lm_b = landmarks[j]

        if lm_a["visibility"] < 0.3 or lm_b["visibility"] < 0.3:
            continue

        # Color based on body region
        connection = (i, j)
        if connection in UPPER_LIMBS:
            color = clothing["top"]
        elif connection in LOWER_LIMBS:
            color = clothing["bottom"]
        else:
            color = "#666666"

        limbs.append({
            "from_index": i,
            "to_index": j,
            "start": {
                "x": round((lm_a["x"] * scale) + pos["x"], 4),
                "y": round((-lm_a["y"] * scale) + pos["y"], 4),
                "z": round((lm_a["z"] * scale) + pos["z"], 4),
            },
            "end": {
                "x": round((lm_b["x"] * scale) + pos["x"], 4),
                "y": round((-lm_b["y"] * scale) + pos["y"], 4),
                "z": round((lm_b["z"] * scale) + pos["z"], 4),
            },
            "thickness": round(thickness * scale, 4),
            "color": color,
        })

    # Compute bounding box
    visible_positions = [j["position"] for j in joints]
    if visible_positions:
        bbox = {
            "min": {
                "x": min(p["x"] for p in visible_positions),
                "y": min(p["y"] for p in visible_positions),
                "z": min(p["z"] for p in visible_positions),
            },
            "max": {
                "x": max(p["x"] for p in visible_positions),
                "y": max(p["y"] for p in visible_positions),
                "z": max(p["z"] for p in visible_positions),
            },
        }
        height = bbox["max"]["y"] - bbox["min"]["y"]
    else:
        bbox = None
        height = 0

    return {
        "type": "body_mesh",
        "joints": joints,
        "limbs": limbs,
        "clothing": clothing,
        "pose_type": pose_type,
        "confidence": pose_data.get("confidence", 0),
        "height": round(height, 3),
        "bounding_box": bbox,
        "world_position": pos,
    }
