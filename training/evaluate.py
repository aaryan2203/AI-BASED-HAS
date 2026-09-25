"""
BAS-HAR Model Evaluation Pipeline

Evaluates trained HAR model on the held-out test sequence dataset.
Calculates Accuracy, Precision, Recall, F1-Score, and per-class performance metrics.
Generates:
- results/classification_report.txt
- results/confusion_matrix.png
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.activity_recognition import HARClassifier
from training.visualize import plot_confusion_matrix

logger = logging.getLogger("BAS-HAR.Evaluate")


def evaluate_model(config: Optional[Config] = None) -> bool:
    """Evaluate model on the test dataset."""
    cfg = config or Config()
    results_dir = Path(cfg.get("results.dir", PROJECT_ROOT / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    test_file = cfg.sequences_dir / "test.npz"
    if not test_file.exists():
        logger.error("Test sequence file not found at %s. Please run prepare_dataset.py.", test_file)
        return False

    if not cfg.labels_path.exists():
        logger.error("labels.json missing at %s", cfg.labels_path)
        return False

    if not cfg.model_path.exists():
        logger.error("HAR model checkpoint not found at %s. Please train a model.", cfg.model_path)
        return False

    # 1. Load Labels & Data
    with open(cfg.labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)
    class_names = [labels_dict[str(i)] for i in range(len(labels_dict))]

    test_data = np.load(test_file)
    X_test, y_test = test_data["X"], test_data["y"]

    logger.info("Loaded %d test sequences across %d classes.", len(X_test), len(class_names))

    # 2. Setup Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("training.device") == "cuda" else "cpu")
    input_size = X_test.shape[2]
    num_classes = len(class_names)

    model = HARClassifier(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_classes=num_classes,
        num_layers=cfg.num_layers,
        rnn_type=cfg.model_architecture,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)

    checkpoint = torch.load(cfg.model_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 3. Model Inference on Test Set
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        logits = model(test_tensor)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    # 4. Compute Metrics
    acc = accuracy_score(y_test, preds)
    report = classification_report(
        y_test,
        preds,
        target_names=class_names,
        labels=list(range(num_classes)),
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(num_classes)))

    # Console Output
    print("\n" + "=" * 60)
    print("BAS-HAR Model Evaluation Results")
    print("=" * 60)
    print(f"Overall Test Accuracy: {acc * 100.0:.2f}%\n")
    print(report)

    # Save Classification Report
    report_file = results_dir / "classification_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("BAS-HAR Model Evaluation Report\n")
        f.write(f"Model: {cfg.model_path}\n")
        f.write(f"Total Test Sequences: {len(X_test)}\n")
        f.write(f"Overall Accuracy: {acc * 100.0:.2f}%\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    logger.info("Saved classification report to %s", report_file)

    # Plot Confusion Matrix
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=class_names, save_path=cm_path)
    logger.info("Saved confusion matrix image to %s", cm_path)

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate_model()
