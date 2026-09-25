import logging
import time
from enum import Enum, auto
from dataclasses import dataclass
import numpy as np

from src.pose import PoseKeypoints
from src.detector import DetectionResult
from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    NONE = auto()
    APPROACHING = auto()
    NEAR = auto()
    TOUCHING = auto()
    GRASPING = auto()
    RELEASING = auto()

@dataclass
class InteractionEvent:
    hand: str
    object_name: str
    object_bbox: tuple[int, int, int, int]
    interaction_type: InteractionType
    confidence: float
    distance: float
    timestamp: float
    wrist_pos: tuple[float, float] = (0.0, 0.0)  # Added to draw lines from wrist to object

class HandObjectInteraction:
    def __init__(self, proximity_threshold: float = 50.0, touch_threshold: float = 25.0):
        self.proximity_threshold = proximity_threshold
        self.touch_threshold = touch_threshold
        
    def _get_distance_to_bbox(self, point: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        x, y = point
        x1, y1, x2, y2 = bbox
        
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return float(np.sqrt(dx**2 + dy**2))

    def analyze(self, poses: list[PoseKeypoints], detections: list[DetectionResult]) -> list[InteractionEvent]:
        events = []
        now = time.time()
        
        for pose in poses:
            left_wrist = pose.keypoints[9]
            right_wrist = pose.keypoints[10]
            left_conf = pose.confidences[9]
            right_conf = pose.confidences[10]
            
            for hand_name, wrist_pt, conf in [('left', left_wrist, left_conf), ('right', right_wrist, right_conf)]:
                if conf < 0.3:
                    continue
                    
                for det in detections:
                    dist = self._get_distance_to_bbox(wrist_pt, det.bbox)
                    
                    interaction_type = InteractionType.NONE
                    if dist <= self.touch_threshold:
                        interaction_type = InteractionType.TOUCHING
                    elif dist <= self.proximity_threshold:
                        interaction_type = InteractionType.NEAR
                    elif dist <= self.proximity_threshold * 2:
                        interaction_type = InteractionType.APPROACHING
                        
                    if interaction_type != InteractionType.NONE:
                        events.append(InteractionEvent(
                            hand=hand_name,
                            object_name=det.class_name,
                            object_bbox=det.bbox,
                            interaction_type=interaction_type,
                            confidence=min(conf, det.confidence),
                            distance=dist,
                            timestamp=now,
                            wrist_pos=(float(wrist_pt[0]), float(wrist_pt[1]))
                        ))
        return events

    def get_activity_from_interactions(self, interactions: list[InteractionEvent], experiment_step_hints: dict | None = None) -> tuple[str, float]:
        if not interactions:
            return ('approach_rack', 0.5)
            
        for event in interactions:
            if event.interaction_type in (InteractionType.NEAR, InteractionType.TOUCHING, InteractionType.GRASPING):
                obj = event.object_name.lower()
                if 'sample_container' in obj or 'container' in obj:
                    return ('pick_container', event.confidence)  # Alternatively open/close
                elif 'button' in obj:
                    return ('press_button', event.confidence)
                elif 'experiment_device' in obj or 'device' in obj:
                    return ('place_sample', event.confidence)
                elif 'sample' in obj:
                    return ('remove_sample', event.confidence)
                    
        return ('return_tool', 0.3)

    def draw_interactions(self, frame: np.ndarray, interactions: list[InteractionEvent]) -> np.ndarray:
        import cv2
        out_frame = frame.copy()
        
        color_map = {
            InteractionType.APPROACHING: (0, 255, 255),  # Yellow
            InteractionType.NEAR: (0, 165, 255),      # Orange
            InteractionType.TOUCHING: (0, 0, 255),       # Red
            InteractionType.GRASPING: (255, 0, 0),       # Blue
            InteractionType.RELEASING: (0, 255, 0)       # Green
        }
        
        for event in interactions:
            color = color_map.get(event.interaction_type, (255, 255, 255))
            x1, y1, x2, y2 = event.object_bbox
            
            # Draw bounding box and label
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{event.hand} {event.interaction_type.name} {event.object_name}"
            cv2.putText(out_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw line from wrist to object center
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            wrist_x, wrist_y = map(int, event.wrist_pos)
            cv2.line(out_frame, (wrist_x, wrist_y), (obj_center_x, obj_center_y), color, 2)
            
        return out_frame
