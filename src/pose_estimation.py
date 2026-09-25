"""
BAS-HAR Pose Estimation Module

Performs offline 2D human pose estimation using YOLO-Pose (Ultralytics).
Extracts 17 COCO landmarks and provides visualization utilities.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from src.preprocessing import normalize_pose_landmarks

logger = logging.getLogger(__name__)

# COCO 17 Keypoint Skeleton Connections (pair of keypoint indices)
COCO_SKELETON_PAIRS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),  # Legs & Hips
    (5, 11), (6, 12), (5, 6),                           # Torso
    (5, 7), (7, 9), (6, 8), (8, 10),                    # Arms
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),             # Head / Face
    (3, 5), (4, 6),                                     # Ears to Shoulders
]


@dataclass
class PoseResult:
    """Encapsulates pose estimation outputs for a single primary person."""
    keypoints: np.ndarray             # Raw (17, 3): x, y, confidence
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float                 # Person detection confidence
    normalized_keypoints: np.ndarray  # Normalized (17, 3): norm_x, norm_y, conf
    norm_meta: dict                   # Dict with center_x, center_y, scale


class PoseEstimator:
    """Offline Human Pose Estimator powered by YOLO-Pose."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11n-pose.pt",
        conf_threshold: float = 0.40,
        kpt_threshold: float = 0.30,
        device: str = "cpu",
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.kpt_threshold = kpt_threshold
        self.device = device
        self.model = None
        self._is_ready = False

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load YOLO-pose model offline."""
        try:
            from ultralytics import YOLO
            logger.info("Loading pose model from %s on device=%s", self.model_path, self.device)
            self.model = YOLO(self.model_path)
            self._is_ready = True
            logger.info("Pose model loaded successfully.")
        except Exception as e:
            logger.warning(
                "Could not initialize YOLO pose model (%s): %s. "
                "Pose estimator will operate in mock/fallback mode until a valid weight file is supplied.",
                self.model_path, e,
            )
            self.model = None
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def estimate(self, frame: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose landmarks for the most prominent person in the frame.
        
        Args:
            frame: BGR image array (H, W, 3)

        Returns:
            PoseResult if a person is detected, else None.
        """
        if frame is None or frame.size == 0:
            return None

        if not self._is_ready or self.model is None:
            return None

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
            if not results or len(results) == 0:
                return None

            result = results[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                return None

            # Find person with highest box confidence
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                best_idx = 0
                person_conf = 0.5
                xyxy = (0, 0, frame.shape[1], frame.shape[0])
            else:
                confs = boxes.conf.cpu().numpy()
                best_idx = int(np.argmax(confs))
                person_conf = float(confs[best_idx])
                box = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                xyxy = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))

            # Keypoints shape: (N, 17, 2) or (N, 17, 3)
            raw_kpts = result.keypoints.data[best_idx].cpu().numpy()
            if raw_kpts.shape[0] < 17:
                return None

            if raw_kpts.shape[1] == 2:
                # Append confidences as 1.0 if not provided
                confs_col = np.ones((17, 1), dtype=np.float32)
                raw_kpts = np.hstack([raw_kpts, confs_col])

            # Normalize pose landmarks
            norm_kpts, meta = normalize_pose_landmarks(
                raw_kpts, min_conf=self.kpt_threshold
            )

            return PoseResult(
                keypoints=raw_kpts,
                bbox=xyxy,
                confidence=person_conf,
                normalized_keypoints=norm_kpts,
                norm_meta=meta,
            )

        except Exception as e:
            logger.error("Pose estimation error: %s", e)
            return None

    def draw_skeleton(
        self,
        frame: np.ndarray,
        pose: PoseResult,
        color_kpt: Tuple[int, int, int] = (0, 255, 0),
        color_bone: Tuple[int, int, int] = (255, 180, 0),
        draw_bbox: bool = True,
    ) -> np.ndarray:
        """Draw pose skeleton and optional bounding box on a copy of the frame."""
        vis = frame.copy()
        kpts = pose.keypoints

        # Draw Bounding Box
        if draw_bbox:
            x1, y1, x2, y2 = pose.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(
                vis,
                f"Person: {pose.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 255),
                2,
            )

        # Draw bones
        for idx1, idx2 in COCO_SKELETON_PAIRS:
            if idx1 < len(kpts) and idx2 < len(kpts):
                pt1 = kpts[idx1]
                pt2 = kpts[idx2]
                if pt1[2] >= self.kpt_threshold and pt2[2] >= self.kpt_threshold:
                    p1 = (int(pt1[0]), int(pt1[1]))
                    p2 = (int(pt2[0]), int(pt2[1]))
                    cv2.line(vis, p1, p2, color_bone, 2, cv2.LINE_AA)

        # Draw keypoint dots
        for i in range(min(17, len(kpts))):
            pt = kpts[i]
            if pt[2] >= self.kpt_threshold:
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 4, color_kpt, -1, cv2.LINE_AA)

        return vis
