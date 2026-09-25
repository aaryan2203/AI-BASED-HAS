"""
BAS-HAR Configuration Loader

Loads and provides typed access to all application settings from config.json.
Handles missing keys with sensible defaults.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default configuration path
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"


class Config:
    """Centralized configuration manager for BAS-HAR."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load configuration from JSON file."""
        if not self._config_path.exists():
            logger.warning(
                "Config file not found at %s, using defaults", self._config_path
            )
            self._data = {}
            return

        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            logger.info("Configuration loaded from %s", self._config_path)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse config file: %s", e)
            self._data = {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a nested config value using dot-separated key path.
        Example: config.get("camera.width", 1280)
        """
        keys = key_path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set a nested config value using dot-separated key path."""
        keys = key_path.split(".")
        current = self._data
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    def save(self, target_path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration back to JSON file."""
        path = Path(target_path) if target_path else self._config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4)
        logger.info("Configuration saved to %s", path)

    # ── Model Accessors ──────────────────────────────────────────────
    @property
    def model_architecture(self) -> str:
        return self.get("model.architecture", "LSTM")

    @property
    def sequence_length(self) -> int:
        return int(self.get("model.sequence_length", 30))

    @property
    def input_size(self) -> int:
        return int(self.get("model.input_size", 100))

    @property
    def hidden_size(self) -> int:
        return int(self.get("model.hidden_size", 128))

    @property
    def num_layers(self) -> int:
        return int(self.get("model.num_layers", 2))

    @property
    def dropout(self) -> float:
        return float(self.get("model.dropout", 0.3))

    @property
    def bidirectional(self) -> bool:
        return bool(self.get("model.bidirectional", False))

    @property
    def model_path(self) -> Path:
        return PROJECT_ROOT / self.get("model.model_path", "models/activity_model/activity_lstm.pth")

    @property
    def labels_path(self) -> Path:
        return PROJECT_ROOT / self.get("model.labels_path", "models/activity_model/labels.json")

    @property
    def metadata_path(self) -> Path:
        return PROJECT_ROOT / self.get("model.metadata_path", "models/activity_model/metadata.json")

    # ── Inference Accessors ──────────────────────────────────────────
    @property
    def confidence_threshold(self) -> float:
        return float(self.get("inference.confidence_threshold", 0.50))

    @property
    def prediction_smoothing(self) -> str:
        return str(self.get("inference.prediction_smoothing", "majority_vote"))

    @property
    def smoothing_window(self) -> int:
        return int(self.get("inference.smoothing_window", 7))

    @property
    def unknown_label(self) -> str:
        return str(self.get("inference.unknown_label", "Unknown"))

    # ── Pose Accessors ───────────────────────────────────────────────
    @property
    def pose_enabled(self) -> bool:
        return bool(self.get("pose.enabled", True))

    @property
    def pose_model_path(self) -> Union[str, Path]:
        p = self.get("pose.model_path", "yolo11n-pose.pt")
        # Check if local relative to root exists
        full_p = PROJECT_ROOT / p
        return full_p if full_p.exists() else p

    @property
    def pose_confidence_threshold(self) -> float:
        return float(self.get("pose.confidence_threshold", 0.40))

    @property
    def keypoint_threshold(self) -> float:
        return float(self.get("pose.keypoint_threshold", 0.30))

    # ── Object Detection Accessors ───────────────────────────────────
    @property
    def object_detection_enabled(self) -> bool:
        return bool(self.get("object_detection.enabled", False))

    @property
    def object_model_path(self) -> Union[str, Path]:
        p = self.get("object_detection.model_path", "yolo11n.pt")
        full_p = PROJECT_ROOT / p
        return full_p if full_p.exists() else p

    @property
    def object_confidence_threshold(self) -> float:
        return float(self.get("object_detection.confidence_threshold", 0.45))

    # ── Dataset & Training Accessors ─────────────────────────────────
    @property
    def raw_dataset_dir(self) -> Path:
        return PROJECT_ROOT / self.get("dataset.raw_dir", "data/raw")

    @property
    def processed_dataset_dir(self) -> Path:
        return PROJECT_ROOT / self.get("dataset.processed_dir", "data/processed")

    @property
    def sequences_dir(self) -> Path:
        return PROJECT_ROOT / self.get("dataset.sequences_dir", "data/sequences")

    @property
    def sequence_stride(self) -> int:
        return int(self.get("dataset.sequence_stride", 10))

    @property
    def batch_size(self) -> int:
        return int(self.get("training.batch_size", 32))

    @property
    def epochs(self) -> int:
        return int(self.get("training.epochs", 50))

    @property
    def learning_rate(self) -> float:
        return float(self.get("training.learning_rate", 0.001))

    # ── Camera Accessors ─────────────────────────────────────────────
    @property
    def camera_source(self) -> Union[int, str]:
        src = self.get("camera.source", 0)
        try:
            return int(src)
        except (ValueError, TypeError):
            return str(src)

    @property
    def camera_width(self) -> int:
        return int(self.get("camera.width", 1280))

    @property
    def camera_height(self) -> int:
        return int(self.get("camera.height", 720))

    @property
    def camera_fps(self) -> int:
        return int(self.get("camera.fps", 30))

    # ── Logging Accessors ────────────────────────────────────────────
    @property
    def log_dir(self) -> Path:
        return PROJECT_ROOT / self.get("logging.log_dir", "logs")

    @property
    def activity_timeline_csv(self) -> Path:
        return PROJECT_ROOT / self.get("logging.activity_timeline_csv", "logs/activity_timeline.csv")

    @property
    def inference_log(self) -> Path:
        return PROJECT_ROOT / self.get("logging.inference_log", "logs/inference.log")

    @property
    def training_log(self) -> Path:
        return PROJECT_ROOT / self.get("logging.training_log", "logs/training.log")


def setup_logging(config: Optional[Config] = None) -> None:
    """Configure system-wide logging based on config."""
    cfg = config or Config()
    log_dir = cfg.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level_name = cfg.get("logging.log_level", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)

    file_handler = logging.FileHandler(cfg.inference_log, encoding="utf-8")
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [file_handler, stream_handler]
