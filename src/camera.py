import cv2
import time
import logging
import platform
import numpy as np
from datetime import datetime
from collections import deque
from typing import Iterator

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class CameraCapture:
    def __init__(self, config: Config, source_override=None):
        self.config = config
        self.source = source_override if source_override is not None else config.camera_source
        self.width = getattr(config, 'camera_width', None)
        self.height = getattr(config, 'camera_height', None)
        self.fps = getattr(config, 'camera_fps', None)
        self.backend = getattr(config, 'camera_backend', "ANY")
        self.retry_delay = getattr(config, 'camera_retry_delay', 1.0)
        self.max_retries = getattr(config, 'camera_max_retries', 3)
        
        self.cap = None
        self._fps_queue = deque(maxlen=30)
        self._last_frame_time = time.time()
        
        self._initialize_camera()

    def _initialize_camera(self) -> bool:
        """Initializes the VideoCapture object with appropriate backend."""
        try:
            if isinstance(self.source, int) and platform.system() == "Windows":
                # Use DirectShow backend for webcams on Windows
                backend = cv2.CAP_DSHOW if self.backend.upper() == "DSHOW" else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.source, backend)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera source: {self.source}")
                return False

            # Set resolution if provided
            if self.width and self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set FPS if provided
            if self.fps:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
            # Read back actual settings
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True
        except Exception as e:
            logger.exception(f"Exception during camera initialization: {e}")
            return False

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Reads a frame with retry logic on failure."""
        retries = 0
        while retries <= self.max_retries:
            if self.cap is None or not self.cap.isOpened():
                logger.warning("Camera not opened, attempting to reconnect...")
                if not self._initialize_camera():
                    time.sleep(self.retry_delay)
                    retries += 1
                    continue

            ret, frame = self.cap.read()
            if ret and frame is not None:
                self._update_fps()
                # Optional: Add timestamp overlay if configured
                if getattr(self.config, 'camera_overlay_timestamp', True):
                    self._add_timestamp(frame)
                return True, frame
            
            logger.warning(f"Failed to read frame. Retry {retries}/{self.max_retries}")
            self.release()
            time.sleep(self.retry_delay)
            retries += 1
            
        logger.error("Max retries reached. Camera disconnected.")
        return False, None
        
    def _update_fps(self):
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps_queue.append(1.0 / dt)
            
    def get_current_fps(self) -> float:
        if not self._fps_queue:
            return 0.0
        return sum(self._fps_queue) / len(self._fps_queue)

    def _add_timestamp(self, frame: np.ndarray):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        fps_text = f"FPS: {self.get_current_fps():.1f}"
        
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def frames(self) -> Iterator[np.ndarray]:
        """Generator yielding frames."""
        while True:
            success, frame = self.read_frame()
            if not success:
                break
            yield frame
            
    def release(self):
        """Releases the camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Camera released.")
