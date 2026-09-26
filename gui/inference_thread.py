"""
BAS-HAR Background Inference Thread

Runs video capture, pose estimation, feature extraction, temporal buffering,
activity recognition, and segmentation asynchronously to keep Tkinter GUI responsive.
"""

import logging
from queue import Queue, Full
import threading
import time
from typing import Optional, Union
import cv2
import numpy as np

from src.config import Config
from src.pose_estimation import PoseEstimator
from src.object_detection import ObjectDetector
from src.feature_extraction import FeatureExtractor
from src.sequence_builder import RealtimeSequenceBuffer
from src.activity_recognition import ActivityRecognizer, ActivityPrediction
from src.activity_segmentation import ActivitySegmenter, ActivityEvent

logger = logging.getLogger("BAS-HAR.InferenceThread")


class InferenceWorker(threading.Thread):
    """
    Dedicated worker thread that consumes frames from camera or video,
    runs the HAR pipeline, and pushes visualization data into a thread-safe Queue.
    """

    def __init__(
        self,
        config: Config,
        frame_queue: Queue,
        event_queue: Queue,
        source: Union[int, str] = 0,
    ):
        super().__init__(daemon=True, name="HARInferenceWorker")
        self.config = config
        self.frame_queue = frame_queue
        self.event_queue = event_queue
        self.source = source

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Instantiate pipeline components
        self.pose_estimator = PoseEstimator(
            model_path=self.config.pose_model_path,
            conf_threshold=self.config.pose_confidence_threshold,
            kpt_threshold=self.config.keypoint_threshold,
        )

        self.object_detector = (
            ObjectDetector(
                model_path=self.config.object_model_path,
                conf_threshold=self.config.object_confidence_threshold,
            )
            if self.config.object_detection_enabled
            else None
        )

        self.feature_extractor = FeatureExtractor(
            include_objects=self.config.object_detection_enabled,
            expected_dim=self.config.input_size,
        )

        self.sequence_buffer = RealtimeSequenceBuffer(
            sequence_length=self.config.sequence_length,
            feature_dim=self.config.input_size,
        )

        self.activity_recognizer = ActivityRecognizer(config=self.config)
        self.segmenter = ActivitySegmenter(
            csv_path=self.config.activity_timeline_csv,
            min_duration_sec=0.5,
        )

        self.cap: Optional[cv2.VideoCapture] = None

    def stop(self) -> None:
        """Signal worker to terminate."""
        self._stop_event.set()

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def set_source(self, source: Union[int, str]) -> None:
        """Switch video source."""
        self.source = source

    def reload_model(self) -> bool:
        """Reload HAR model weights and labels."""
        return self.activity_recognizer.load_model_and_labels()

    def run(self) -> None:
        """Worker main loop."""
        logger.info("Opening video source: %s", self.source)
        try:
            # If source is integer string (e.g. "0"), convert to int
            src_val = int(self.source) if str(self.source).isdigit() else self.source
            self.cap = cv2.VideoCapture(src_val)
        except Exception as e:
            logger.error("Failed to open source %s: %s", self.source, e)
            return

        if not self.cap or not self.cap.isOpened():
            logger.error("VideoCapture could not be opened for source: %s", self.source)
            return

        self.feature_extractor.reset()
        self.sequence_buffer.clear()

        last_fps_calc = time.time()
        frames_counted = 0
        fps = 0.0

        while not self._stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Video ended or camera disconnected
                logger.info("Video stream ended or frame empty.")
                break

            frames_counted += 1
            now = time.time()
            if now - last_fps_calc >= 1.0:
                fps = frames_counted / (now - last_fps_calc)
                frames_counted = 0
                last_fps_calc = now

            # 1. Pose Estimation
            pose_result = self.pose_estimator.estimate(frame)

            # 2. Optional Object Detection
            objects = None
            obj_interaction = None
            if self.object_detector and self.object_detector.is_ready:
                objects = self.object_detector.detect(frame)
                lw = pose_result.keypoints[9] if pose_result else None
                rw = pose_result.keypoints[10] if pose_result else None
                scale = pose_result.norm_meta.get("scale", 1.0) if pose_result else 1.0
                obj_interaction = self.object_detector.compute_interaction_features(objects, lw, rw, scale)

            # 3. Feature Extraction
            features = self.feature_extractor.extract(pose_result, objects, obj_interaction)

            # 4. Sequence Buffer
            sequence = self.sequence_buffer.push(features)

            # 5. Activity Recognition
            pred = self.activity_recognizer.predict(sequence)
            pred.fps = fps if fps > 0 else pred.fps

            # 6. Activity Segmentation & Timeline Update
            if pred.is_known:
                concluded_event = self.segmenter.update(pred.activity, pred.confidence, timestamp=now)
                if concluded_event:
                    try:
                        self.event_queue.put_nowait(concluded_event)
                    except Full:
                        pass

            # 7. Visual Overlays
            vis_frame = frame.copy()
            if pose_result:
                vis_frame = self.pose_estimator.draw_skeleton(vis_frame, pose_result)

            if objects and self.object_detector:
                vis_frame = self.object_detector.draw_objects(vis_frame, objects)

            # Overlay Prediction Badge on frame
            badge_color = (0, 200, 50) if pred.is_known else (100, 100, 100)
            text = f"Activity: {pred.activity.upper()} ({pred.confidence * 100.0:.1f}%)"
            cv2.rectangle(vis_frame, (10, 10), (450, 55), (20, 20, 20), -1)
            cv2.rectangle(vis_frame, (10, 10), (450, 55), badge_color, 2)
            cv2.putText(
                vis_frame,
                text,
                (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Push visual frame and prediction data to UI queue
            payload = {
                "frame": vis_frame,
                "prediction": pred,
                "model_ready": self.activity_recognizer.is_ready,
                "error_msg": self.activity_recognizer.error_message,
            }

            try:
                # Keep queue fresh by discarding older unrendered frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put_nowait(payload)
            except Full:
                pass

            # Sleep slightly to match target FPS if reading video file
            time.sleep(0.005)

        # Conclude and flush ongoing activity segment
        final_event = self.segmenter.flush()
        if final_event:
            try:
                self.event_queue.put_nowait(final_event)
            except:
                pass

        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Inference worker finished.")
