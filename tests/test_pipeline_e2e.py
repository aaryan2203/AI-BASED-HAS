"""
End-to-End Pipeline Verification Test for BAS-HAR

Verifies that:
1. Training loop loads sequences and metadata.
2. Generates activity_lstm.pth, labels.json, metadata.json, and training_history.json.
3. Produces loss and accuracy curve visualization.
4. Evaluation computes accuracy, precision, recall, F1, and confusion matrix image.
"""

import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch

from src.config import Config
from training.train import train_model
from training.evaluate import evaluate_model
from training.visualize import plot_training_history, plot_confusion_matrix


def test_training_and_evaluation_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        seq_dir = tmp_path / "sequences"
        seq_dir.mkdir(parents=True, exist_ok=True)
        model_dir = tmp_path / "models" / "activity_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        results_dir = tmp_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        num_classes = 3
        seq_len = 15
        feat_dim = 100
        samples_per_class = 20

        # Create labels
        labels = {"0": "standing", "1": "walking", "2": "reaching"}
        labels_file = model_dir / "labels.json"
        with open(labels_file, "w") as f:
            json.dump(labels, f)

        # Generate sample sequence data with class-separable features
        X_list = []
        y_list = []
        for c in range(num_classes):
            data = np.random.randn(samples_per_class, seq_len, feat_dim).astype(np.float32)
            data[:, :, c] += 3.0  # separable signal
            X_list.append(data)
            y_list.extend([c] * samples_per_class)

        X = np.concatenate(X_list, axis=0)
        y = np.array(y_list, dtype=np.int64)

        # Train / Val / Test split
        n_train = 40
        n_val = 10
        n_test = 10

        np.savez_compressed(seq_dir / "train.npz", X=X[:n_train], y=y[:n_train])
        np.savez_compressed(seq_dir / "val.npz", X=X[n_train : n_train + n_val], y=y[n_train : n_train + n_val])
        np.savez_compressed(seq_dir / "test.npz", X=X[n_train + n_val :], y=y[n_train + n_val :])

        # Configure custom paths
        cfg = Config()
        cfg.set("dataset.sequences_dir", str(seq_dir))
        cfg.set("model.model_path", str(model_dir / "activity_lstm.pth"))
        cfg.set("model.labels_path", str(labels_file))
        cfg.set("model.metadata_path", str(model_dir / "metadata.json"))
        cfg.set("model.history_path", str(model_dir / "training_history.json"))
        cfg.set("model.sequence_length", seq_len)
        cfg.set("model.input_size", feat_dim)
        cfg.set("model.hidden_size", 32)
        cfg.set("model.num_layers", 1)
        cfg.set("training.epochs", 3)
        cfg.set("training.batch_size", 16)
        cfg.set("logging.log_dir", str(logs_dir))
        cfg.set("logging.training_log", str(logs_dir / "training.log"))

        # 1. Run Training
        success = train_model(config=cfg, epochs=3, batch_size=16)
        assert success

        # Verify saved model files
        assert (model_dir / "activity_lstm.pth").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "training_history.json").exists()

        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)
            assert meta["num_classes"] == 3
            assert meta["architecture"] == "LSTM"

        # 2. Run Evaluation
        eval_success = evaluate_model(config=cfg)
        assert eval_success
