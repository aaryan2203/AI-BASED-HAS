import logging
from dataclasses import dataclass
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    class_name: str
    class_id: int
    confidence: float

class ObjectDetector:
    def __init__(self, config: Config, model_path: str = 'yolo11n.pt'):
        self.config = config
        self.threshold = getattr(config, 'detection_threshold', 0.5)
        self.object_mapping = getattr(config, 'object_mapping', {})
        self.device = config.resolve_device() if hasattr(config, 'resolve_device') else 'cpu'
        
        self.model = None
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        if YOLO is None:
            logger.error("ultralytics package not found. Cannot load YOLO model.")
            return
            
        try:
            logger.info(f"Loading YOLO model from {model_path} on device {self.device}...")
            self.model = YOLO(model_path)
            # Send to device
            self.model.to(self.device)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.exception(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        if self.model is None:
            return []
            
        results = self.model(frame, verbose=False, device=self.device)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.threshold:
                    continue
                    
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detections.append(DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf
                ))
                
        return detections

    def detect_persons(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detects only persons (usually class 'person' in COCO)."""
        detections = self.detect(frame)
        return [d for d in detections if d.class_name == 'person']

    def detect_objects(self, frame: np.ndarray) -> list[DetectionResult]:
        """Detects experiment objects based on config object_mapping."""
        detections = self.detect(frame)
        mapped_detections = []
        for d in detections:
            if d.class_name in self.object_mapping:
                mapped_name = self.object_mapping[d.class_name]
                mapped_detections.append(DetectionResult(
                    bbox=d.bbox,
                    class_name=mapped_name,
                    class_id=d.class_id,
                    confidence=d.confidence
                ))
            elif getattr(self.config, 'keep_unmapped_objects', False) and d.class_name != 'person':
                mapped_detections.append(d)
        return mapped_detections

    def draw_detections(self, frame: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
        """Draws bounding boxes and labels on the frame."""
        out_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Colors: Green for astronaut (person), Blue for experiment objects
            is_person = det.class_name == 'person'
            color = (0, 255, 0) if is_person else (255, 0, 0)
            
            cv2.rectangle(out_frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            cv2.rectangle(out_frame, (x1, y1 - text_height - 5), 
                          (x1 + text_width, y1), color, -1)
            cv2.putText(out_frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
        return out_frame
