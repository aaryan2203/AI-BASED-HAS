"""
BAS-HAR Dataset Preparation Pipeline

Processes raw video directories organized by activity class:
data/raw/
├── walking/
│   ├── video1.mp4
│   └── video2.mp4
├── sitting/
│   ├── video3.mp4
└── ...

Pipeline:
1. Validates dataset structure and video integrity.
2. Dynamically discovers classes and writes labels.json.
3. Extracts frame poses and optional object features using PoseEstimator and FeatureExtractor.
4. Generates temporal sliding window sequences.
5. Performs stratified Train/Val/Test split.
6. Exports .npz sequences and dataset_summary.json.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.pose_estimation import PoseEstimator
from src.object_detection import ObjectDetector
from src.feature_extraction import FeatureExtractor
from src.sequence_builder import create_sliding_windows

logger = logging.getLogger("BAS-HAR.Dataset")


class DatasetValidator:
    """Validates raw dataset integrity before processing."""

    def __init__(self, raw_dir: Path, min_samples_per_class: int = 1):
        self.raw_dir = Path(raw_dir)
        self.min_samples_per_class = min_samples_per_class
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self) -> Tuple[bool, List[Path]]:
        """
        Validates raw dataset directory.
        Returns:
            (is_valid, list of valid class directories)
        """
        self.errors.clear()
        self.warnings.clear()

        if not self.raw_dir.exists():
            self.errors.append(f"Raw dataset directory does not exist: {self.raw_dir}")
            return False, []

        class_dirs = [d for d in self.raw_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        if not class_dirs:
            self.errors.append(f"No activity class folders found inside: {self.raw_dir}")
            return False, []

        valid_class_dirs = []
        valid_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        for c_dir in sorted(class_dirs):
            video_files = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
            if not video_files:
                self.warnings.append(f"Class folder '{c_dir.name}' has no supported video files.")
                continue

            if len(video_files) < self.min_samples_per_class:
                self.warnings.append(
                    f"Class '{c_dir.name}' has only {len(video_files)} video(s), "
                    f"less than recommended min {self.min_samples_per_class}."
                )

            valid_class_dirs.append(c_dir)

        if not valid_class_dirs:
            self.errors.append("No class folders contain valid video files.")
            return False, []

        return True, valid_class_dirs


def process_video_file(
    video_path: Path,
    pose_estimator: PoseEstimator,
    feature_extractor: FeatureExtractor,
    object_detector: Optional[ObjectDetector] = None,
    max_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Reads a single video file, extracts features per frame, and returns (T, feature_dim).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Corrupted or unreadable video: %s", video_path)
        return None

    feature_extractor.reset()
    frame_features = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # Pose Estimation
        pose_res = pose_estimator.estimate(frame)

        # Optional Object Detection
        objects = None
        obj_interaction = None
        if object_detector and object_detector.is_ready:
            objects = object_detector.detect(frame)
            lw = pose_res.keypoints[9] if pose_res else None
            rw = pose_res.keypoints[10] if pose_res else None
            scale = pose_res.norm_meta.get("scale", 1.0) if pose_res else 1.0
            obj_interaction = object_detector.compute_interaction_features(objects, lw, rw, scale)

        # Feature Extraction
        feat = feature_extractor.extract(pose_res, objects, obj_interaction)
        frame_features.append(feat)

        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break

    cap.release()

    if not frame_features:
        logger.warning("No frames extracted from video: %s", video_path)
        return None

    return np.array(frame_features, dtype=np.float32)


def prepare_dataset(
    config: Optional[Config] = None,
    raw_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> bool:
    """
    Full dataset preparation pipeline.
    """
    cfg = config or Config()
    raw_path = Path(raw_dir) if raw_dir else cfg.raw_dataset_dir
    seq_dir = Path(output_dir) if output_dir else cfg.sequences_dir
    processed_dir = cfg.processed_dataset_dir

    seq_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Validating dataset at %s...", raw_path)
    validator = DatasetValidator(raw_path, min_samples_per_class=1)
    is_valid, valid_classes = validator.validate()

    for w in validator.warnings:
        logger.warning(w)

    if not is_valid:
        for err in validator.errors:
            logger.error(err)
        return False

    # Dynamic Class Label Discovery
    class_names = sorted([d.name for d in valid_classes])
    label_map = {idx: name for idx, name in enumerate(class_names)}
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    # Save dynamic labels.json
    labels_file = cfg.labels_path
    with open(labels_file, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)
    logger.info("Saved %d dynamic classes to %s", len(label_map), labels_file)

    # Initialize Feature Extractors
    pose_est = PoseEstimator(
        model_path=cfg.pose_model_path,
        conf_threshold=cfg.pose_confidence_threshold,
        kpt_threshold=cfg.keypoint_threshold,
    )
    obj_det = None
    if cfg.object_detection_enabled:
        obj_det = ObjectDetector(
            model_path=cfg.object_model_path,
            conf_threshold=cfg.object_confidence_threshold,
        )

    feat_ext = FeatureExtractor(
        include_objects=cfg.object_detection_enabled,
        expected_dim=cfg.input_size,
    )

    sequence_length = cfg.sequence_length
    stride = cfg.sequence_stride

    all_sequences: List[np.ndarray] = []
    all_labels: List[int] = []
    class_sample_counts: Dict[str, int] = {name: 0 for name in class_names}

    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    for c_dir in valid_classes:
        c_name = c_dir.name
        c_idx = name_to_idx[c_name]
        video_files = [f for f in c_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]

        logger.info("Processing class '%s' (%d videos)...", c_name, len(video_files))
        for v_file in video_files:
            features = process_video_file(v_file, pose_est, feat_ext, obj_det)
            if features is None or len(features) == 0:
                continue

            windows = create_sliding_windows(features, sequence_length=sequence_length, stride=stride)
            for w in windows:
                all_sequences.append(w)
                all_labels.append(c_idx)
                class_sample_counts[c_name] += 1

    if not all_sequences:
        logger.error("No valid sequences generated from dataset.")
        return False

    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    logger.info("Extracted %d total sequences of shape (%d, %d)", len(X), sequence_length, cfg.input_size)
    for c_name, count in class_sample_counts.items():
        logger.info("  - Class '%s': %d sequences", c_name, count)

    # Stratified Split (Train 70%, Val 15%, Test 15%)
    from sklearn.model_selection import train_test_split

    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
        )
    except ValueError:
        # Fallback without stratification if a class has too few samples
        logger.warning("Stratified split failed due to low class sample counts. Using unstratified split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )

    # Save splits as compressed NPZ
    np.savez_compressed(seq_dir / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(seq_dir / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(seq_dir / "test.npz", X=X_test, y=y_test)

    # Save summary metadata
    summary = {
        "num_classes": len(class_names),
        "classes": label_map,
        "sequence_length": sequence_length,
        "feature_dim": cfg.input_size,
        "total_samples": len(X),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "class_distribution": class_sample_counts,
    }
    with open(processed_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    logger.info("Dataset preparation complete! Splits saved in %s", seq_dir)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    prepare_dataset()
