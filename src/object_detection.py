"""
BAS-HAR Modular Object Detection & Human-Object Interaction

Detects experiment-relevant objects (containers, tools, devices) using YOLO.
Computes spatial relationships such as hand/wrist-to-object distances.
Designed to be completely modular and optional.
"""

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Represents a detected object in the frame."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]      # cx, cy


class ObjectDetector:
    """Modular offline object detector for experiment items."""

    def __init__(
        self,
        model_path: Union[str, Path] = "yolo11n.pt",
        conf_threshold: float = 0.45,
        relevant_classes: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.relevant_classes = set(relevant_classes) if relevant_classes else None
        self.device = device
        self.model = None
        self._is_ready = False

        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load YOLO object detector offline."""
        try:
            from ultralytics import YOLO
            logger.info("Loading object detector from %s on device=%s", self.model_path, self.device)
            self.model = YOLO(self.model_path)
            self._is_ready = True
            logger.info("Object detector loaded successfully.")
        except Exception as e:
            logger.warning(
                "Could not initialize YOLO object detector (%s): %s. "
                "Object detection will be inactive or bypassed.",
                self.model_path, e,
            )
            self.model = None
            self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def detect(self, frame: np.ndarray) -> List[DetectedObject]:
        """
        Detect relevant objects in the frame.

        Args:
            frame: BGR frame array

        Returns:
            List of DetectedObject items
        """
        if frame is None or frame.size == 0 or not self._is_ready or self.model is None:
            return []

        try:
            results = self.model(
                frame,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False,
            )
            if not results or len(results) == 0:
                return []

            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                return []

            detected: List[DetectedObject] = []
            names = result.names or {}

            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cls_id = cls_arr[i]
                cls_name = names.get(cls_id, f"obj_{cls_id}")

                # Exclude person from object detections
                if cls_name == "person":
                    continue

                if self.relevant_classes and cls_name not in self.relevant_classes:
                    continue

                x1, y1, x2, y2 = xyxy_arr[i].astype(int)
                conf = float(conf_arr[i])
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                detected.append(
                    DetectedObject(
                        class_name=cls_name,
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        center=(cx, cy),
                    )
                )

            return detected

        except Exception as e:
            logger.error("Object detection error: %s", e)
            return []

    def compute_interaction_features(
        self,
        objects: List[DetectedObject],
        left_wrist: Optional[np.ndarray] = None,
        right_wrist: Optional[np.ndarray] = None,
        scale: float = 1.0,
    ) -> np.ndarray:
        """
        Compute human-object spatial relationships:
        - min distance from left wrist to any object (normalized by scale)
        - min distance from right wrist to any object (normalized by scale)
        - number of objects in immediate reach (within distance < 1.0)
        - total number of detected objects
        
        Returns 1D feature array of shape (16,)
        """
        feats = np.zeros(16, dtype=np.float32)
        if not objects:
            return feats

        feats[0] = float(len(objects))
        scale = max(1e-4, scale)

        min_dist_l = 999.0
        min_dist_r = 999.0
        reach_count = 0

        for obj in objects:
            obj_xy = np.array(obj.center, dtype=np.float32)

            if left_wrist is not None and len(left_wrist) >= 2:
                dl = np.linalg.norm(left_wrist[:2] - obj_xy) / scale
                if dl < min_dist_l:
                    min_dist_l = dl
                if dl < 1.0:
                    reach_count += 1

            if right_wrist is not None and len(right_wrist) >= 2:
                dr = np.linalg.norm(right_wrist[:2] - obj_xy) / scale
                if dr < min_dist_r:
                    min_dist_r = dr
                if dr < 1.0:
                    reach_count += 1

        feats[1] = float(min(10.0, min_dist_l)) if min_dist_l < 999.0 else 10.0
        feats[2] = float(min(10.0, min_dist_r)) if min_dist_r < 999.0 else 10.0
        feats[3] = float(reach_count)

        # Store closest object relative positions (up to 2 objects: 6 dims each = 12 dims)
        for i, obj in enumerate(objects[:2]):
            base_idx = 4 + (i * 6)
            feats[base_idx] = obj.confidence
            feats[base_idx + 1] = obj.center[0] / 1280.0
            feats[base_idx + 2] = obj.center[1] / 720.0
            feats[base_idx + 3] = (obj.bbox[2] - obj.bbox[0]) / 1280.0
            feats[base_idx + 4] = (obj.bbox[3] - obj.bbox[1]) / 720.0
            feats[base_idx + 5] = 1.0  # present flag

        return feats

    def draw_objects(
        self,
        frame: np.ndarray,
        objects: List[DetectedObject],
        color: Tuple[int, int, int] = (255, 100, 0),
    ) -> np.ndarray:
        """Draw bounding boxes and class labels for detected objects."""
        vis = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.class_name} ({obj.confidence:.2f})"
            cv2.putText(
                vis,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        return vis
