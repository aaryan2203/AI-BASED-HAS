import logging
from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO
from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

COCO_KEYPOINT_NAMES = {
    0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
}

SKELETON_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8),
    (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

@dataclass
class PoseKeypoints:
    keypoints: np.ndarray  # (17, 2)
    confidences: np.ndarray  # (17,)
    bbox: tuple[int, int, int, int]
    overall_confidence: float


class PoseEstimator:
    def __init__(self, model_name: str = 'yolo11n-pose.pt'):
        self.device = getattr(Config, 'DEVICE', 'cpu')
        try:
            self.model = YOLO(model_name)
            self.model.to(self.device)
            logger.info(f"Loaded pose model {model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load pose model {model_name}: {e}")
            self.model = None

    def estimate(self, frame: np.ndarray) -> list[PoseKeypoints]:
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            poses = []
            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue
                
                for i in range(len(result.keypoints)):
                    kp_data = result.keypoints[i].data[0].cpu().numpy()
                    if kp_data.shape[1] == 3:
                        keypoints = kp_data[:, :2]
                        confidences = kp_data[:, 2]
                    else:
                        keypoints = kp_data
                        confidences = np.ones(17)
                    
                    box = result.boxes[i].xyxy[0].cpu().numpy().astype(int)
                    conf = float(result.boxes[i].conf[0].cpu().numpy())
                    poses.append(PoseKeypoints(keypoints, confidences, tuple(box), conf))
            return poses
        except Exception as e:
            logger.error(f"Error estimating pose: {e}")
            return []

    def draw_skeleton(self, frame: np.ndarray, poses: list[PoseKeypoints], conf_threshold: float = 0.5) -> np.ndarray:
        import cv2
        out_frame = frame.copy()
        
        for pose in poses:
            for pt_idx, (x, y) in enumerate(pose.keypoints):
                if pose.confidences[pt_idx] > conf_threshold:
                    cv2.circle(out_frame, (int(x), int(y)), 4, (0, 255, 255), -1)
            
            for pair_idx, (idx1, idx2) in enumerate(SKELETON_PAIRS):
                if pose.confidences[idx1] > conf_threshold and pose.confidences[idx2] > conf_threshold:
                    pt1 = (int(pose.keypoints[idx1][0]), int(pose.keypoints[idx1][1]))
                    pt2 = (int(pose.keypoints[idx2][0]), int(pose.keypoints[idx2][1]))
                    
                    ratio = pair_idx / len(SKELETON_PAIRS)
                    color = (int(255 * (1 - ratio)), int(255 * (1 - ratio)), 255)
                    cv2.line(out_frame, pt1, pt2, color, 2)
                    
        return out_frame


class PoseBuffer:
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.buffer: list[PoseKeypoints] = []

    def add(self, pose: PoseKeypoints):
        self.buffer.append(pose)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def get_sequence(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros((0, 34))
        seq = []
        for pose in self.buffer:
            seq.append(pose.keypoints.flatten())
        return np.array(seq)

    def get_motion_features(self) -> dict:
        if len(self.buffer) < 2:
            return {
                "wrist_velocity": 0.0,
                "hand_displacement": 0.0,
                "body_center": (0.0, 0.0),
                "is_reaching": False
            }
        
        first_pose = self.buffer[0]
        last_pose = self.buffer[-1]
        
        first_left_wrist = first_pose.keypoints[9]
        first_right_wrist = first_pose.keypoints[10]
        last_left_wrist = last_pose.keypoints[9]
        last_right_wrist = last_pose.keypoints[10]
        
        displacement_left = float(np.linalg.norm(last_left_wrist - first_left_wrist))
        displacement_right = float(np.linalg.norm(last_right_wrist - first_right_wrist))
        
        hand_displacement = displacement_left + displacement_right
        wrist_velocity = hand_displacement / len(self.buffer)
        
        left_hip = last_pose.keypoints[11]
        right_hip = last_pose.keypoints[12]
        body_center = (float(left_hip[0] + right_hip[0]) / 2.0, float(left_hip[1] + right_hip[1]) / 2.0)
        
        left_shoulder = last_pose.keypoints[5]
        right_shoulder = last_pose.keypoints[6]
        
        dist_left_arm = float(np.linalg.norm(last_left_wrist - left_shoulder))
        dist_right_arm = float(np.linalg.norm(last_right_wrist - right_shoulder))
        
        is_reaching = (dist_left_arm > 100.0) or (dist_right_arm > 100.0)
        
        return {
            "wrist_velocity": wrist_velocity,
            "hand_displacement": hand_displacement,
            "body_center": body_center,
            "is_reaching": is_reaching
        }
        
    def is_full(self) -> bool:
        return len(self.buffer) == self.max_size
        
    def clear(self):
        self.buffer.clear()
