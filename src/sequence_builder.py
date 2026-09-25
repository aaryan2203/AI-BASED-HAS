"""
BAS-HAR Sequence Builder Module

Manages temporal window buffers for online inference and sliding-window
sequence generation for offline dataset preparation.
"""

from collections import deque
from typing import List, Optional, Tuple, Union
import numpy as np


class RealtimeSequenceBuffer:
    """
    Rolling temporal sequence buffer for live video stream inference.
    Maintains a FIFO queue of length `sequence_length`.
    """

    def __init__(self, sequence_length: int = 30, feature_dim: int = 100):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=sequence_length)
        self.total_pushed = 0

    def clear(self) -> None:
        """Reset the buffer."""
        self.buffer.clear()
        self.total_pushed = 0

    def is_full(self) -> bool:
        """Returns True if the buffer contains at least sequence_length samples."""
        return len(self.buffer) >= self.sequence_length

    def push(self, feature: np.ndarray) -> np.ndarray:
        """
        Add a new feature vector to the buffer and return a full sequence tensor
        of shape (sequence_length, feature_dim).
        
        If fewer than sequence_length features have been pushed, left-pads with zeros.
        """
        feat = np.array(feature, dtype=np.float32)
        if feat.ndim != 1 or len(feat) != self.feature_dim:
            if len(feat) < self.feature_dim:
                pad = np.zeros(self.feature_dim, dtype=np.float32)
                pad[: len(feat)] = feat
                feat = pad
            else:
                feat = feat[: self.feature_dim]

        self.buffer.append(feat)
        self.total_pushed += 1

        # Build output array
        arr = np.array(self.buffer, dtype=np.float32)
        if len(arr) < self.sequence_length:
            pad_count = self.sequence_length - len(arr)
            padding = np.zeros((pad_count, self.feature_dim), dtype=np.float32)
            arr = np.vstack([padding, arr])

        return arr  # Shape: (sequence_length, feature_dim)

    def get_sequence(self) -> np.ndarray:
        """Return the current sequence without modifying the buffer."""
        if not self.buffer:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        arr = np.array(self.buffer, dtype=np.float32)
        if len(arr) < self.sequence_length:
            pad_count = self.sequence_length - len(arr)
            padding = np.zeros((pad_count, self.feature_dim), dtype=np.float32)
            arr = np.vstack([padding, arr])
        return arr


def create_sliding_windows(
    features: np.ndarray,
    sequence_length: int = 30,
    stride: int = 10,
) -> np.ndarray:
    """
    Generate overlapping temporal sequences from a continuous feature matrix.
    
    Args:
        features: 2D array of shape (T, feature_dim)
        sequence_length: Number of time steps per window
        stride: Step size between consecutive windows

    Returns:
        3D array of shape (num_windows, sequence_length, feature_dim)
    """
    if features is None or len(features) == 0:
        return np.empty((0, sequence_length, 0), dtype=np.float32)

    T, D = features.shape

    if T < sequence_length:
        # Pad with initial frame replication or zeros to reach sequence_length
        pad_count = sequence_length - T
        padding = np.repeat(features[0:1], pad_count, axis=0) if T > 0 else np.zeros((pad_count, D), dtype=np.float32)
        seq = np.vstack([padding, features])
        return seq[np.newaxis, :, :]

    windows = []
    for start in range(0, T - sequence_length + 1, stride):
        window = features[start : start + sequence_length]
        windows.append(window)

    if not windows:
        # Guarantee at least the tail sequence
        windows.append(features[-sequence_length:])

    return np.array(windows, dtype=np.float32)
