"""
BAS-HAR: Offline Human Activity Recognition System
Entry Point for GUI, CLI, Training, Dataset Preparation, and Evaluation.

Usage:
    python main.py                   # Run GUI (default)
    python main.py --cli             # Run CLI / Headless inference
    python main.py --source video.mp4# Run CLI with video file
    python main.py --prepare-data    # Prepare dataset from data/raw/
    python main.py --train           # Train HAR model
    python main.py --evaluate        # Evaluate trained HAR model on test set
"""

import argparse
import logging
from pathlib import Path
import sys
import time
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, setup_logging


def run_cli_inference(source=0, config=None):
    """Run CLI/Headless inference and print predictions to console."""
    
    from src.pose_estimation import PoseEstimator
    from src.object_detection import ObjectDetector
    from src.feature_extraction import FeatureExtractor
    from src.sequence_builder import RealtimeSequenceBuffer
    from src.activity_recognition import ActivityRecognizer
    from src.activity_segmentation import ActivitySegmenter

    cfg = config or Config()
    logger = logging.getLogger("BAS-HAR.CLI")

    logger.info("Initializing components for CLI inference...")
    pose_est = PoseEstimator(
        model_path=cfg.pose_model_path,
        conf_threshold=cfg.pose_confidence_threshold,
        kpt_threshold=cfg.keypoint_threshold,
    )
    obj_det = (
        ObjectDetector(
            model_path=cfg.object_model_path,
            conf_threshold=cfg.object_confidence_threshold,
        )
        if cfg.object_detection_enabled
        else None
    )
    feat_ext = FeatureExtractor(
        include_objects=cfg.object_detection_enabled,
        expected_dim=cfg.input_size,
    )
    seq_buf = RealtimeSequenceBuffer(
        sequence_length=cfg.sequence_length,
        feature_dim=cfg.input_size,
    )
    recognizer = ActivityRecognizer(config=cfg)
    segmenter = ActivitySegmenter(
        csv_path=cfg.activity_timeline_csv,
        min_duration_sec=0.5,
    )

    src_val = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src_val)
    if not cap.isOpened():
        logger.error("Could not open video source: %s", source)
        return

    logger.info("Starting inference on source '%s'. Press Ctrl+C to stop.", source)
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = time.time()
            pose_res = pose_est.estimate(frame)
            objects = None
            obj_inter = None
            if obj_det and obj_det.is_ready:
                objects = obj_det.detect(frame)
                lw = pose_res.keypoints[9] if pose_res else None
                rw = pose_res.keypoints[10] if pose_res else None
                scale = pose_res.norm_meta.get("scale", 1.0) if pose_res else 1.0
                obj_inter = obj_det.compute_interaction_features(objects, lw, rw, scale)

            features = feat_ext.extract(pose_res, objects, obj_inter)
            sequence = seq_buf.push(features)
            pred = recognizer.predict(sequence)

            if pred.is_known:
                event = segmenter.update(pred.activity, pred.confidence, timestamp=now)
                if event:
                    print(f"[{event.start_str}] Concluded: {event.activity} (Duration: {event.duration_sec}s, Conf: {event.confidence * 100:.1f}%)")

            print(
                f"\rActivity: {pred.activity.upper():<16} | Conf: {pred.confidence*100:5.1f}% | FPS: {pred.fps:4.1f} | Latency: {pred.processing_time_ms:4.1f}ms",
                end="",
                flush=True,
            )

    except KeyboardInterrupt:
        print("\nInference stopped by user.")
    finally:
        final_ev = segmenter.flush()
        if final_ev:
            print(f"[{final_ev.start_str}] Concluded: {final_ev.activity} (Duration: {final_ev.duration_sec}s, Conf: {final_ev.confidence * 100:.1f}%)")
        cap.release()
        print("\nSession finished. Activity timeline logged to:", cfg.activity_timeline_csv)


def main():
    parser = argparse.ArgumentParser(description="BAS-HAR: Offline Human Activity Recognition System")
    parser.add_argument("--cli", action="store_true", help="Run inference in terminal CLI mode without GUI")
    parser.add_argument("--source", type=str, default=None, help="Video file or camera index (default: from config.json)")
    parser.add_argument("--prepare-data", action="store_true", help="Run dataset preparation pipeline")
    parser.add_argument("--train", action="store_true", help="Train HAR recurrent model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model on test dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config.json")
    args = parser.parse_args()

    config = Config(args.config)
    setup_logging(config)

    if args.prepare_data:
        from training.prepare_dataset import prepare_dataset
        prepare_dataset(config)
        return

    if args.train:
        from training.train import train_model
        train_model(config)
        return

    if args.evaluate:
        from training.evaluate import evaluate_model
        evaluate_model(config)
        return

    source = args.source if args.source is not None else config.camera_source

    if args.cli:
        run_cli_inference(source=source, config=config)
    else:
        from gui.main import run_gui
        run_gui()


if __name__ == "__main__":
    main()
