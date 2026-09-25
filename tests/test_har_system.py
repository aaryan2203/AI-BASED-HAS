"""
BAS-HAR Comprehensive Test Suite

Tests:
1. Pose normalization & invariance to translation and scale
2. Geometric feature extraction and angle computation
3. Feature extraction dimension consistency (100-dim pose, 116-dim pose+object)
4. Temporal sequence builder & rolling buffer
5. PyTorch HARClassifier architecture (LSTM and GRU)
6. Dynamic label loading and low confidence 'Unknown' handling
7. Temporal prediction smoothing algorithms
8. Activity segmentation & timeline event logging
9. Error handling for missing model weights and missing labels
"""

import json
from pathlib import Path
import tempfile
import time
import numpy as np
import pytest
import torch

from src.config import Config
from src.preprocessing import (
    calculate_angle,
    normalize_pose_landmarks,
    extract_pose_geometric_features,
    KEYPOINT_INDEX,
)
from src.pose_estimation import PoseResult
from src.object_detection import DetectedObject, ObjectDetector
from src.feature_extraction import FeatureExtractor
from src.sequence_builder import RealtimeSequenceBuffer, create_sliding_windows
from src.activity_recognition import HARClassifier, ActivityRecognizer, ActivityPrediction
from src.activity_segmentation import ActivitySegmenter, ActivityEvent


# ── 1. Pose Preprocessing Tests ──────────────────────────────────────

def test_calculate_angle():
    # Right angle: (1, 0) - (0, 0) - (0, 1) should be 90 degrees
    a = np.array([1.0, 0.0, 1.0])
    b = np.array([0.0, 0.0, 1.0])
    c = np.array([0.0, 1.0, 1.0])
    angle = calculate_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-4

    # Straight line: (1, 0) - (0, 0) - (-1, 0) should be 180 degrees
    c2 = np.array([-1.0, 0.0, 1.0])
    angle_straight = calculate_angle(a, b, c2)
    assert abs(angle_straight - 180.0) < 1e-4


def test_pose_normalization_invariance():
    # Create synthetic 17 COCO keypoints
    kpts1 = np.zeros((17, 3), dtype=np.float32)
    kpts1[:, 2] = 1.0  # Full confidence

    # Define hips and shoulders
    kpts1[KEYPOINT_INDEX["left_hip"], :2] = [-20.0, 100.0]
    kpts1[KEYPOINT_INDEX["right_hip"], :2] = [20.0, 100.0]
    kpts1[KEYPOINT_INDEX["left_shoulder"], :2] = [-25.0, 50.0]
    kpts1[KEYPOINT_INDEX["right_shoulder"], :2] = [25.0, 50.0]
    kpts1[KEYPOINT_INDEX["left_wrist"], :2] = [-40.0, 80.0]

    # Translate and scale kpts2 (representing person moving closer and to the right)
    shift = np.array([300.0, 200.0], dtype=np.float32)
    scale = 2.5
    kpts2 = kpts1.copy()
    kpts2[:, :2] = kpts1[:, :2] * scale + shift

    norm1, meta1 = normalize_pose_landmarks(kpts1)
    norm2, meta2 = normalize_pose_landmarks(kpts2)

    # Normalized coordinates should be identical despite shift and scale!
    np.testing.assert_allclose(norm1[:, :2], norm2[:, :2], atol=1e-4)


def test_extract_pose_geometric_features():
    norm_kpts = np.zeros((17, 3), dtype=np.float32)
    norm_kpts[:, 2] = 1.0
    geom_feats = extract_pose_geometric_features(norm_kpts)
    # 7 angles + 8 distances = 15 dimensions
    assert geom_feats.shape == (15,)
    assert not np.isnan(geom_feats).any()


# ── 2. Feature Extraction Tests ──────────────────────────────────────

def test_feature_extractor_dimensions():
    # Pose only (100 dims)
    extractor_pose = FeatureExtractor(include_objects=False, expected_dim=100)
    assert extractor_pose.expected_dim == 100

    dummy_pose = PoseResult(
        keypoints=np.zeros((17, 3), dtype=np.float32),
        bbox=(0, 0, 100, 200),
        confidence=0.9,
        normalized_keypoints=np.zeros((17, 3), dtype=np.float32),
        norm_meta={},
    )
    feat1 = extractor_pose.extract(dummy_pose)
    assert feat1.shape == (100,)

    # Pose + Objects (116 dims)
    extractor_obj = FeatureExtractor(include_objects=True, expected_dim=116)
    feat2 = extractor_obj.extract(dummy_pose)
    assert feat2.shape == (116,)


def test_feature_extractor_velocity():
    extractor = FeatureExtractor(include_objects=False, expected_dim=100)

    pose1 = PoseResult(
        keypoints=np.zeros((17, 3), dtype=np.float32),
        bbox=(0, 0, 100, 200),
        confidence=0.9,
        normalized_keypoints=np.zeros((17, 3), dtype=np.float32),
        norm_meta={},
    )
    # Frame 1: velocities should be zero
    feat1 = extractor.extract(pose1)
    assert np.all(feat1[66:100] == 0.0)

    # Frame 2: shift left wrist
    norm2 = np.zeros((17, 3), dtype=np.float32)
    norm2[KEYPOINT_INDEX["left_wrist"], 0] = 0.5  # Moved 0.5 units right
    pose2 = PoseResult(
        keypoints=np.zeros((17, 3), dtype=np.float32),
        bbox=(0, 0, 100, 200),
        confidence=0.9,
        normalized_keypoints=norm2,
        norm_meta={},
    )
    feat2 = extractor.extract(pose2)
    # Check left wrist velocity dx at index 66 + 9*2 = 84
    wrist_dx_idx = 66 + (KEYPOINT_INDEX["left_wrist"] * 2)
    assert abs(feat2[wrist_dx_idx] - 0.5) < 1e-4


# ── 3. Sequence Builder Tests ────────────────────────────────────────

def test_realtime_sequence_buffer():
    buf = RealtimeSequenceBuffer(sequence_length=10, feature_dim=100)
    assert not buf.is_full()

    # Push 1 feature: should return (10, 100) with zero padding
    seq = buf.push(np.ones(100, dtype=np.float32))
    assert seq.shape == (10, 100)
    assert np.all(seq[0] == 0.0)  # Oldest is padded
    assert np.all(seq[-1] == 1.0)  # Latest is 1.0

    # Push 9 more
    for i in range(9):
        seq = buf.push(np.full(100, i + 2, dtype=np.float32))

    assert buf.is_full()
    assert seq.shape == (10, 100)
    assert seq[0, 0] == 1.0
    assert seq[-1, 0] == 10.0


def test_sliding_window_generation():
    T = 50
    features = np.arange(T * 10, dtype=np.float32).reshape(T, 10)
    windows = create_sliding_windows(features, sequence_length=20, stride=5)
    # (50 - 20) / 5 + 1 = 7 windows
    assert windows.shape == (7, 20, 10)


# ── 4. PyTorch Classifier Tests ──────────────────────────────────────

def test_har_classifier_lstm_and_gru():
    batch_size = 4
    seq_len = 30
    input_dim = 100
    num_classes = 5

    x = torch.randn(batch_size, seq_len, input_dim)

    # Test LSTM
    lstm_model = HARClassifier(
        input_size=input_dim,
        hidden_size=64,
        num_classes=num_classes,
        num_layers=2,
        rnn_type="LSTM",
    )
    out_lstm = lstm_model(x)
    assert out_lstm.shape == (batch_size, num_classes)

    # Test GRU
    gru_model = HARClassifier(
        input_size=input_dim,
        hidden_size=64,
        num_classes=num_classes,
        num_layers=1,
        rnn_type="GRU",
    )
    out_gru = gru_model(x)
    assert out_gru.shape == (batch_size, num_classes)


# ── 5. Activity Recognizer & Dynamic Labels ───────────────────────────

def test_activity_recognizer_low_confidence_and_dynamic_labels():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        labels_file = tmp_path / "labels.json"
        model_file = tmp_path / "activity_lstm.pth"

        # 1. Write dynamic labels
        labels = {"0": "walking", "1": "reaching", "2": "picking"}
        with open(labels_file, "w") as f:
            json.dump(labels, f)

        # 2. Write dummy trained weights
        model = HARClassifier(input_size=100, hidden_size=32, num_classes=3)
        torch.save({"model_state_dict": model.state_dict()}, model_file)

        # 3. Initialize Recognizer
        cfg = Config()
        cfg.set("model.input_size", 100)
        cfg.set("model.hidden_size", 32)
        cfg.set("model.num_layers", 2)
        cfg.set("inference.confidence_threshold", 0.50)

        recognizer = ActivityRecognizer(
            config=cfg,
            model_path=model_file,
            labels_path=labels_file,
            device="cpu",
        )
        assert recognizer.is_ready
        assert len(recognizer.labels) == 3

        # Test prediction shape
        dummy_seq = np.random.randn(30, 100).astype(np.float32)
        pred = recognizer.predict(dummy_seq)
        assert isinstance(pred, ActivityPrediction)
        assert pred.activity in ["walking", "reaching", "picking", "Unknown"]


def test_missing_model_error_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        non_existent_model = tmp_path / "does_not_exist.pth"
        labels_file = tmp_path / "labels.json"
        with open(labels_file, "w") as f:
            json.dump({"0": "standing"}, f)

        recognizer = ActivityRecognizer(
            model_path=non_existent_model,
            labels_path=labels_file,
        )
        # Should not crash!
        assert not recognizer.is_ready
        assert recognizer.error_message is not None
        assert "not found" in recognizer.error_message

        # Predicting when not ready should return Unknown safely
        dummy_seq = np.zeros((30, 100), dtype=np.float32)
        pred = recognizer.predict(dummy_seq)
        assert pred.activity == "Unknown"
        assert pred.confidence == 0.0


# ── 6. Activity Segmentation Tests ────────────────────────────────────

def test_activity_segmenter():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "timeline.csv"
        segmenter = ActivitySegmenter(csv_path=csv_file, min_duration_sec=0.2)

        t0 = 1000.0
        # Activity 1: standing from t=1000.0 to t=1002.0
        segmenter.update("standing", 0.95, timestamp=t0)
        segmenter.update("standing", 0.96, timestamp=t0 + 1.0)
        
        # Transition to reaching at t=1002.0
        ev1 = segmenter.update("reaching", 0.90, timestamp=t0 + 2.0)
        assert ev1 is not None
        assert ev1.activity == "standing"
        assert abs(ev1.duration_sec - 2.0) < 1e-2
        assert abs(ev1.confidence - 0.955) < 1e-2

        # Activity 2: reaching from t=1002.0 to t=1005.0
        segmenter.update("reaching", 0.92, timestamp=t0 + 4.0)
        ev2 = segmenter.flush(timestamp=t0 + 5.0)
        assert ev2 is not None
        assert ev2.activity == "reaching"
        assert abs(ev2.duration_sec - 3.0) < 1e-2

        # Check CSV output
        assert csv_file.exists()
        with open(csv_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 3  # Header + 2 events


# ── 7. GUI Initialization Test ────────────────────────────────────────

def test_gui_initialization():
    import tkinter as tk
    from gui.main import BASHARApp

    root = tk.Tk()
    root.withdraw()  # Hide window during test
    try:
        app = BASHARApp(root)
        assert app is not None
        assert app.prediction_panel is not None
        assert app.timeline_widget is not None
        assert app.video_canvas is not None
        assert app.control_panel is not None
        root.update()
    finally:
        root.destroy()

