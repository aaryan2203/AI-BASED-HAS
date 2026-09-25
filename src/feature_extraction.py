"""
BAS-HAR Feature Extraction Module

Combines normalized pose keypoints, geometric joint features, temporal keypoint velocities,
and optional object interaction features into a unified 1D feature vector for HAR models.
"""

from typing import List, Optional, Tuple
import numpy as np

from src.preprocessing import extract_pose_geometric_features
from src.pose_estimation import PoseResult
from src.object_detection import DetectedObject


class FeatureExtractor:
    """
    Extracts a deterministic 1D feature vector per frame.
    
    Feature Layout (Pose-Only, 100 dims):
    - [0:51]   : 17 normalized keypoints (x, y, confidence)
    - [51:66]  : 15 geometric features (7 joint angles + 8 relative distances)
    - [66:100] : 34 temporal velocities (dx, dy for 17 keypoints relative to previous frame)

    Optional Object Features (Appended if enabled, +16 dims -> 116 dims):
    - [100:116]: 16 object interaction features (wrist distances, counts, bbox coords)
    """

    def __init__(self, include_objects: bool = False, expected_dim: int = 100):
        self.include_objects = include_objects
        self.expected_dim = expected_dim
        self.prev_norm_coords: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Reset internal temporal velocity state (e.g. at start of a new video)."""
        self.prev_norm_coords = None

    def extract(
        self,
        pose: Optional[PoseResult],
        objects: Optional[List[DetectedObject]] = None,
        object_interaction_feats: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract fixed-size 1D feature vector.
        
        Args:
            pose: PoseResult from PoseEstimator (or None if no person detected)
            objects: Optional list of detected objects
            object_interaction_feats: Optional precomputed 16-dim interaction features

        Returns:
            1D numpy array of shape (expected_dim,)
        """
        if pose is None:
            self.reset()
            return np.zeros(self.expected_dim, dtype=np.float32)

        norm_kpts = pose.normalized_keypoints  # (17, 3)
        curr_coords = norm_kpts[:, :2]         # (17, 2)

        # 1. 17 keypoints flattened (51 dims)
        kpt_flat = norm_kpts.flatten().astype(np.float32)

        # 2. Geometric features (15 dims)
        geom_feats = extract_pose_geometric_features(norm_kpts)

        # 3. Velocities (34 dims)
        if self.prev_norm_coords is not None:
            velocity = (curr_coords - self.prev_norm_coords).flatten().astype(np.float32)
            # Clip extreme velocity spikes caused by keypoint jumps
            velocity = np.clip(velocity, -5.0, 5.0)
        else:
            velocity = np.zeros(34, dtype=np.float32)

        self.prev_norm_coords = curr_coords.copy()

        # Assemble pose feature vector (100 dims)
        pose_vector = np.concatenate([kpt_flat, geom_feats, velocity])

        if self.include_objects:
            if object_interaction_feats is not None:
                obj_vec = object_interaction_feats[:16]
            else:
                obj_vec = np.zeros(16, dtype=np.float32)
            full_vector = np.concatenate([pose_vector, obj_vec])
        else:
            full_vector = pose_vector

        # Ensure exact expected dimension
        if len(full_vector) < self.expected_dim:
            padded = np.zeros(self.expected_dim, dtype=np.float32)
            padded[: len(full_vector)] = full_vector
            return padded
        elif len(full_vector) > self.expected_dim:
            return full_vector[: self.expected_dim]

        return full_vector.astype(np.float32)
