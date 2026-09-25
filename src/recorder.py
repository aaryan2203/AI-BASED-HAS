import cv2
import logging
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import numpy as np

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class VideoRecorder:
    """
    Video Recording Module.
    Writes video frames to an MP4 file in a background thread.
    """
    def __init__(self, config: Config):
        self.config = config
        self.recordings_dir = PROJECT_ROOT / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        
        self._is_recording = False
        self._queue = queue.Queue(maxsize=300) # prevent memory blowup
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._output_path = None
        self._writer = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self, width: int, height: int, fps: float):
        if self._is_recording:
            logger.warning("Already recording, ignoring start request.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = self.recordings_dir / f"BAS_{timestamp}.mp4"
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._writer = cv2.VideoWriter(str(self._output_path), fourcc, fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter. Check codecs.")
                
            self._is_recording = True
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker, daemon=True)
            self._worker_thread.start()
            logger.info(f"Started recording to {self._output_path}")
            
        except Exception as e:
            logger.error(f"Failed to start video recording: {e}")
            self._is_recording = False
            self._writer = None

    def _worker(self):
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                frame = self._queue.get(timeout=0.1)
                if self._writer and frame is not None:
                    self._writer.write(frame)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error writing frame: {e}")

    def write_frame(self, frame: np.ndarray):
        if not self._is_recording:
            return
            
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            logger.warning("Video recording queue full, dropping frame.")

    def stop(self) -> str:
        if not self._is_recording:
            return ""
            
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
            
        if self._writer:
            self._writer.release()
            self._writer = None
            
        self._is_recording = False
        logger.info(f"Stopped recording. File saved at {self._output_path}")
        return str(self._output_path) if self._output_path else ""
