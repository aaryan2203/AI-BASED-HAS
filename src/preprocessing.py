"""
BAS-HAR Pose Preprocessing & Normalization

Normalizes human pose landmarks to be invariant to:
- Camera distance (scale normalization)
- Person location in frame (pelvis-centered translation)
- Body size differences
- Minor camera movements

Also computes geometric features including joint angles, relative distances,
and keypoint velocities.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


# COCO 17 Keypoint Indices
KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

KEYPOINT_INDEX = {v: k for k, v in KEYPOINT_NAMES.items()}


def calculate_angle(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, min_conf: float = 0.1
) -> float:
    """
    Calculate the 2D planar angle at vertex b formed by vectors ba and bc in degrees [0, 180].
    Points a, b, c are expected as (x, y) or (x, y, conf).
    Returns 0.0 if any point is invalid or confidence is too low.
    """
    if len(a) >= 3 and len(b) >= 3 and len(c) >= 3:
        if a[2] < min_conf or b[2] < min_conf or c[2] < min_conf:
            return 0.0

    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def normalize_pose_landmarks(
    keypoints: np.ndarray,
    min_conf: float = 0.2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize 17 COCO pose keypoints (shape: (17, 2) or (17, 3)):
    1. Translation centering: places the pelvis/mid-hip at (0, 0).
    2. Scale normalization: divides by the torso length (mid-shoulder to mid-hip).
       Falls back to bounding box scale if torso keypoints have low confidence.

    Returns:
        normalized_keypoints: shape (17, 3) where each row is [norm_x, norm_y, conf]
        meta: dict containing translation center and scale factor
    """
    kpts = np.array(keypoints, dtype=np.float32)
    if kpts.ndim != 2 or kpts.shape[0] < 17:
        raise ValueError(f"Expected keypoints shape (17, 2) or (17, 3), got {kpts.shape}")

    has_conf = kpts.shape[1] >= 3
    coords = kpts[:, :2].copy()
    confs = kpts[:, 2].copy() if has_conf else np.ones((kpts.shape[0],), dtype=np.float32)

    # 1. Determine origin (mid-hip or fallback to mid-shoulder or centroid)
    l_hip = coords[KEYPOINT_INDEX["left_hip"]]
    r_hip = coords[KEYPOINT_INDEX["right_hip"]]
    l_hip_conf = confs[KEYPOINT_INDEX["left_hip"]]
    r_hip_conf = confs[KEYPOINT_INDEX["right_hip"]]

    l_sh = coords[KEYPOINT_INDEX["left_shoulder"]]
    r_sh = coords[KEYPOINT_INDEX["right_shoulder"]]
    l_sh_conf = confs[KEYPOINT_INDEX["left_shoulder"]]
    r_sh_conf = confs[KEYPOINT_INDEX["right_shoulder"]]

    if l_hip_conf > min_conf and r_hip_conf > min_conf:
        center = (l_hip + r_hip) / 2.0
    elif l_hip_conf > min_conf:
        center = l_hip.copy()
    elif r_hip_conf > min_conf:
        center = r_hip.copy()
    elif l_sh_conf > min_conf and r_sh_conf > min_conf:
        center = (l_sh + r_sh) / 2.0
    else:
        # Fallback to mean of visible coordinates
        visible_mask = confs > min_conf
        if np.any(visible_mask):
            center = np.mean(coords[visible_mask], axis=0)
        else:
            center = np.mean(coords, axis=0)

    # 2. Translate coordinates so center is (0, 0)
    centered_coords = coords - center

    # 3. Determine scale (torso length or bbox diagonal)
    scale = 1.0
    mid_shoulder = (l_sh + r_sh) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0
    shoulder_conf = min(l_sh_conf, r_sh_conf)
    hip_conf = min(l_hip_conf, r_hip_conf)

    if shoulder_conf > min_conf and hip_conf > min_conf:
        torso_len = np.linalg.norm(mid_shoulder - mid_hip)
        if torso_len > 1e-4:
            scale = float(torso_len)
    
    if scale <= 1e-4 or scale == 1.0:
        # Use bounding box extent of visible keypoints
        visible_mask = confs > min_conf
        if np.any(visible_mask):
            vis_coords = coords[visible_mask]
            min_xy = np.min(vis_coords, axis=0)
            max_xy = np.max(vis_coords, axis=0)
            bbox_diag = np.linalg.norm(max_xy - min_xy)
            if bbox_diag > 1e-4:
                scale = float(bbox_diag) / 2.0
        else:
            scale = 100.0

    normalized_coords = centered_coords / scale

    # Reassemble with confidence
    normalized_kpts = np.zeros((17, 3), dtype=np.float32)
    normalized_kpts[:, :2] = normalized_coords
    normalized_kpts[:, 2] = confs

    meta = {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "scale": float(scale),
    }

    return normalized_kpts, meta


def extract_pose_geometric_features(normalized_kpts: np.ndarray) -> np.ndarray:
    """
    Extracts geometric features from normalized (17, 3) keypoints:
    - 7 joint angles (in radians, normalized to [0, 1]):
      * Left Elbow, Right Elbow
      * Left Knee, Right Knee
      * Left Shoulder, Right Shoulder
      * Torso inclination angle
    - 8 relative normalized distances:
      * Left wrist to right wrist
      * Left wrist to left shoulder
      * Right wrist to right shoulder
      * Left wrist to left hip
      * Right wrist to right hip
      * Left ankle to right ankle
      * Left wrist to nose
      * Right wrist to nose
    
    Returns 1D array of shape (15,)
    """
    coords = normalized_kpts[:, :2]
    confs = normalized_kpts[:, 2]

    # Angles (in degrees, converted to [0, 1])
    angles = [
        calculate_angle(coords[KEYPOINT_INDEX["left_shoulder"]], coords[KEYPOINT_INDEX["left_elbow"]], coords[KEYPOINT_INDEX["left_wrist"]]),
        calculate_angle(coords[KEYPOINT_INDEX["right_shoulder"]], coords[KEYPOINT_INDEX["right_elbow"]], coords[KEYPOINT_INDEX["right_wrist"]]),
        calculate_angle(coords[KEYPOINT_INDEX["left_hip"]], coords[KEYPOINT_INDEX["left_knee"]], coords[KEYPOINT_INDEX["left_ankle"]]),
        calculate_angle(coords[KEYPOINT_INDEX["right_hip"]], coords[KEYPOINT_INDEX["right_knee"]], coords[KEYPOINT_INDEX["right_ankle"]]),
        calculate_angle(coords[KEYPOINT_INDEX["left_hip"]], coords[KEYPOINT_INDEX["left_shoulder"]], coords[KEYPOINT_INDEX["left_elbow"]]),
        calculate_angle(coords[KEYPOINT_INDEX["right_hip"]], coords[KEYPOINT_INDEX["right_shoulder"]], coords[KEYPOINT_INDEX["right_elbow"]]),
    ]
    # Torso tilt angle relative to vertical [0, -1]
    mid_sh = (coords[KEYPOINT_INDEX["left_shoulder"]] + coords[KEYPOINT_INDEX["right_shoulder"]]) / 2.0
    mid_hip = (coords[KEYPOINT_INDEX["left_hip"]] + coords[KEYPOINT_INDEX["right_hip"]]) / 2.0
    torso_vec = mid_sh - mid_hip
    norm_torso = np.linalg.norm(torso_vec)
    if norm_torso > 1e-6:
        # vertical vector is (0, -1) in image coordinates (y goes down)
        vert_vec = np.array([0.0, -1.0])
        cos_tilt = np.clip(np.dot(torso_vec, vert_vec) / norm_torso, -1.0, 1.0)
        torso_tilt = np.degrees(np.arccos(cos_tilt))
    else:
        torso_tilt = 0.0
    angles.append(torso_tilt)

    # Normalize angles [0, 180] -> [0, 1]
    normalized_angles = np.array(angles, dtype=np.float32) / 180.0

    # Distances
    def dist(idx1: str, idx2: str) -> float:
        return float(np.linalg.norm(coords[KEYPOINT_INDEX[idx1]] - coords[KEYPOINT_INDEX[idx2]]))

    distances = np.array([
        dist("left_wrist", "right_wrist"),
        dist("left_wrist", "left_shoulder"),
        dist("right_wrist", "right_shoulder"),
        dist("left_wrist", "left_hip"),
        dist("right_wrist", "right_hip"),
        dist("left_ankle", "right_ankle"),
        dist("left_wrist", "nose"),
        dist("right_wrist", "nose"),
    ], dtype=np.float32)

    return np.concatenate([normalized_angles, distances])
