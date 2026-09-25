"""
BAS-HAR Visualization Utilities

Generates plots for:
- Training and Validation Loss / Accuracy curves
- Confusion Matrix with class labels
- Activity Timeline and Distribution charts
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / file saving
import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Union[str, Path] = "results/training_curves.png",
) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss Plot
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r--", label="Val Loss", linewidth=2)
    ax1.set_title("Loss over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy Plot
    train_acc = [a * 100.0 for a in history["train_acc"]]
    val_acc = [a * 100.0 for a in history["val_acc"]]
    ax2.plot(epochs, train_acc, "b-", label="Train Accuracy", linewidth=2)
    ax2.plot(epochs, val_acc, "r--", label="Val Accuracy", linewidth=2)
    ax2.set_title("Accuracy over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path] = "results/confusion_matrix.png",
    normalize: bool = True,
) -> None:
    """Plot and save confusion matrix heatmap."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1e-6)
        display_data = cm_norm
        fmt = ".2f"
    else:
        display_data = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(display_data, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Activity Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = display_data.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = f"{display_data[i, j]:{fmt}}"
            ax.text(
                j,
                i,
                val,
                ha="center",
                va="center",
                color="white" if display_data[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
