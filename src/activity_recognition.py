"""
BAS-HAR Temporal Activity Recognition Engine

Provides modular PyTorch LSTM/GRU classifier, dynamic label loading from labels.json,
temporal prediction smoothing, low-confidence 'Unknown' thresholding, and latency tracking.
"""

from collections import deque
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.config import Config

logger = logging.getLogger(__name__)


class HARClassifier(nn.Module):
    """
    Modular recurrent temporal classifier for human activity recognition.
    Supports LSTM and GRU backbones.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional

        rnn_dropout = dropout if num_layers > 1 else 0.0

        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=bidirectional,
            )

        mult = 2 if bidirectional else 1
        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * mult, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # RNN forward
        out, _ = self.rnn(x)  # out shape: (batch_size, sequence_length, hidden_size * mult)
        
        # Take last time step
        last_step = out[:, -1, :]
        logits = self.fc_layer(last_step)
        return logits


@dataclass
class ActivityPrediction:
    """Encapsulates output of a single activity recognition inference step."""
    activity: str
    confidence: float
    is_known: bool
    raw_probabilities: Dict[str, float] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    fps: float = 0.0
    timestamp: float = field(default_factory=time.time)


class ActivityRecognizer:
    """
    Inference manager for Human Activity Recognition.
    Handles dynamic label mapping, temporal smoothing, and low-confidence filtering.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[Union[str, Path]] = None,
        labels_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.config = config or Config()
        self.model_path = Path(model_path) if model_path else self.config.model_path
        self.labels_path = Path(labels_path) if labels_path else self.config.labels_path

        # Determine compute device
        if device:
            self.device = torch.device(device)
        else:
            configured_dev = self.config.get("training.device", "auto")
            if configured_dev == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

        self.conf_threshold = self.config.confidence_threshold
        self.smoothing_mode = self.config.prediction_smoothing
        self.smoothing_window = self.config.smoothing_window
        self.unknown_label = self.config.unknown_label

        # State buffers
        self.labels: Dict[int, str] = {}
        self.model: Optional[HARClassifier] = None
        self._is_ready = False
        self.error_message: Optional[str] = None

        # Smoothing history
        self.prediction_history: deque = deque(maxlen=self.smoothing_window)
        self.prob_history: deque = deque(maxlen=self.smoothing_window)

        # FPS calculation
        self._last_inference_time = time.time()
        self._fps_history: deque = deque(maxlen=10)

        # Initialize
        self.load_model_and_labels()

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def load_model_and_labels(self) -> bool:
        """Loads labels.json and activity_lstm.pth dynamically."""
        self._is_ready = False
        self.error_message = None

        # 1. Load labels.json
        if not self.labels_path.exists():
            self.error_message = f"HAR labels not found at {self.labels_path}. Please train a model."
            logger.warning(self.error_message)
            return False

        try:
            with open(self.labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.labels = {int(k): str(v) for k, v in data.items()}
            logger.info("Loaded %d activity labels from %s", len(self.labels), self.labels_path)
        except Exception as e:
            self.error_message = f"Failed to parse labels.json: {e}"
            logger.error(self.error_message)
            return False

        if not self.labels:
            self.error_message = "No labels defined in labels.json"
            return False

        # 2. Check model weights
        if not self.model_path.exists():
            self.error_message = (
                f"HAR model weights not found at {self.model_path}. "
                f"Please train a model or load a valid model."
            )
            logger.warning(self.error_message)
            return False

        # 3. Instantiate model architecture and load weights
        try:
            num_classes = len(self.labels)
            input_size = self.config.input_size
            hidden_size = self.config.hidden_size
            num_layers = self.config.num_layers
            rnn_type = self.config.model_architecture
            dropout = self.config.dropout
            bidirectional = self.config.bidirectional

            self.model = HARClassifier(
                input_size=input_size,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                rnn_type=rnn_type,
                dropout=dropout,
                bidirectional=bidirectional,
            ).to(self.device)

            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)

            self.model.eval()
            self._is_ready = True
            logger.info("Loaded HAR model successfully from %s", self.model_path)
            return True

        except Exception as e:
            self.error_message = f"Failed to load HAR model: {e}"
            logger.error(self.error_message)
            self.model = None
            self._is_ready = False
            return False

    def predict(self, sequence: np.ndarray) -> ActivityPrediction:
        """
        Classifies an activity from a temporal sequence tensor.

        Args:
            sequence: 2D array of shape (sequence_length, feature_dim)

        Returns:
            ActivityPrediction with label, confidence, latency, and FPS.
        """
        start_time = time.perf_counter()

        if not self._is_ready or self.model is None:
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ActivityPrediction(
                activity=self.unknown_label,
                confidence=0.0,
                is_known=False,
                processing_time_ms=elapsed,
                fps=0.0,
            )

        try:
            seq_tensor = torch.tensor(
                sequence, dtype=torch.float32, device=self.device
            ).unsqueeze(0)  # Shape: (1, seq_len, feature_dim)

            with torch.no_grad():
                logits = self.model(seq_tensor)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            # Record probability history for smoothing
            self.prob_history.append(probs)

            # Map raw probabilities
            raw_prob_dict = {
                self.labels[idx]: float(prob)
                for idx, prob in enumerate(probs)
                if idx in self.labels
            }

            # Apply Smoothing
            smoothed_activity, smoothed_conf = self._apply_smoothing(probs)

            # Low confidence threshold check
            is_known = True
            if smoothed_conf < self.conf_threshold:
                final_activity = self.unknown_label
                is_known = False
            else:
                final_activity = smoothed_activity

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Calculate FPS
            now = time.time()
            dt = now - self._last_inference_time
            self._last_inference_time = now
            if dt > 0:
                self._fps_history.append(1.0 / dt)
            fps = float(np.mean(self._fps_history)) if self._fps_history else 0.0

            return ActivityPrediction(
                activity=final_activity,
                confidence=float(smoothed_conf),
                is_known=is_known,
                raw_probabilities=raw_prob_dict,
                processing_time_ms=elapsed_ms,
                fps=fps,
            )

        except Exception as e:
            logger.error("Inference execution error: %s", e)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ActivityPrediction(
                activity=self.unknown_label,
                confidence=0.0,
                is_known=False,
                processing_time_ms=elapsed_ms,
                fps=0.0,
            )

    def _apply_smoothing(self, current_probs: np.ndarray) -> Tuple[str, float]:
        """Apply configured prediction smoothing algorithm."""
        best_idx = int(np.argmax(current_probs))
        best_conf = float(current_probs[best_idx])
        instant_activity = self.labels.get(best_idx, self.unknown_label)

        self.prediction_history.append((instant_activity, best_conf))

        if self.smoothing_mode == "moving_average" and len(self.prob_history) > 1:
            mean_probs = np.mean(self.prob_history, axis=0)
            s_idx = int(np.argmax(mean_probs))
            return self.labels.get(s_idx, self.unknown_label), float(mean_probs[s_idx])

        elif self.smoothing_mode == "exponential" and len(self.prob_history) > 1:
            alpha = 0.4
            weighted = self.prob_history[0]
            for p in list(self.prob_history)[1:]:
                weighted = alpha * p + (1 - alpha) * weighted
            s_idx = int(np.argmax(weighted))
            return self.labels.get(s_idx, self.unknown_label), float(weighted[s_idx])

        else:
            # Default: majority voting over prediction window
            counts: Dict[str, int] = {}
            conf_accum: Dict[str, List[float]] = {}
            for act, c in self.prediction_history:
                counts[act] = counts.get(act, 0) + 1
                conf_accum.setdefault(act, []).append(c)

            majority_act = max(counts, key=counts.get)
            avg_conf = float(np.mean(conf_accum[majority_act]))
            return majority_act, avg_conf
