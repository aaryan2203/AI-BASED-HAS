"""
BAS-HAR Model Training Pipeline

Trains recurrent temporal classifier (LSTM/GRU) on prepared sequences.
Saves:
- models/activity_model/activity_lstm.pth
- models/activity_model/labels.json
- models/activity_model/metadata.json
- models/activity_model/training_history.json
- logs/training.log
- results/training_curves.png (via visualize.py)
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from src.activity_recognition import HARClassifier
from training.visualize import plot_training_history

logger = logging.getLogger("BAS-HAR.Train")


def train_model(
    config: Optional[Config] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> bool:
    """Execute complete model training."""
    cfg = config or Config()
    
    # Setup training log file
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(cfg.training_log, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    num_epochs = epochs or cfg.epochs
    b_size = batch_size or cfg.batch_size
    lr = learning_rate or cfg.learning_rate

    try:
        logger.info("=" * 60)
        logger.info("Starting BAS-HAR Training Pipeline")
        logger.info("=" * 60)

        # 1. Load sequence splits
        train_file = cfg.sequences_dir / "train.npz"
        val_file = cfg.sequences_dir / "val.npz"

        if not train_file.exists() or not val_file.exists():
            logger.error(
                "Prepared sequence files not found in %s. Please run prepare_dataset.py first.",
                cfg.sequences_dir,
            )
            return False

        train_data = np.load(train_file)
        val_data = np.load(val_file)

        X_train, y_train = train_data["X"], train_data["y"]
        X_val, y_val = val_data["X"], val_data["y"]

        logger.info("Loaded Train samples: %d, Validation samples: %d", len(X_train), len(X_val))

        # 2. Load dynamic labels
        if not cfg.labels_path.exists():
            logger.error("labels.json missing at %s", cfg.labels_path)
            return False

        with open(cfg.labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)

        num_classes = len(labels_dict)
        seq_len = X_train.shape[1]
        input_dim = X_train.shape[2]

        logger.info("Model configuration:")
        logger.info("  - Classes: %d (%s)", num_classes, list(labels_dict.values()))
        logger.info("  - Sequence Length: %d", seq_len)
        logger.info("  - Input Dimension: %d", input_dim)
        logger.info("  - Architecture: %s", cfg.model_architecture)

        # Compute class weights for loss balance
        class_counts = np.bincount(y_train, minlength=num_classes)
        total_samples = len(y_train)
        class_weights = []
        for count in class_counts:
            weight = total_samples / (num_classes * max(1, count))
            class_weights.append(weight)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        # 3. Setup Device & Dataloaders
        device_name = cfg.get("training.device", "auto")
        if device_name == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info("Training on device: %s", device)

        weights_tensor = weights_tensor.to(device)

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long),
        )

        train_loader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=b_size, shuffle=False)

        # 4. Initialize Model
        model = HARClassifier(
            input_size=input_dim,
            hidden_size=cfg.hidden_size,
            num_classes=num_classes,
            num_layers=cfg.num_layers,
            rnn_type=cfg.model_architecture,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        ).to(device)

        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=cfg.get("training.weight_decay", 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # 5. Training Loop
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = -1.0
        patience = cfg.get("training.patience", 10)
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            # Train Step
            model.train()
            train_loss = 0.0
            train_correct = 0
            total_train = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_y)
                preds = torch.argmax(outputs, dim=-1)
                train_correct += (preds == batch_y).sum().item()
                total_train += len(batch_y)

            epoch_train_loss = train_loss / max(1, total_train)
            epoch_train_acc = train_correct / max(1, total_train)

            # Validation Step
            model.eval()
            val_loss = 0.0
            val_correct = 0
            total_val = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item() * len(batch_y)
                    preds = torch.argmax(outputs, dim=-1)
                    val_correct += (preds == batch_y).sum().item()
                    total_val += len(batch_y)

            epoch_val_loss = val_loss / max(1, total_val)
            epoch_val_acc = val_correct / max(1, total_val)

            scheduler.step(epoch_val_acc)

            history["train_loss"].append(float(epoch_train_loss))
            history["train_acc"].append(float(epoch_train_acc))
            history["val_loss"].append(float(epoch_val_loss))
            history["val_acc"].append(float(epoch_val_acc))

            if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
                logger.info(
                    "Epoch [%3d/%3d] - Train Loss: %.4f, Train Acc: %.2f%% | Val Loss: %.4f, Val Acc: %.2f%%",
                    epoch,
                    num_epochs,
                    epoch_train_loss,
                    epoch_train_acc * 100.0,
                    epoch_val_loss,
                    epoch_val_acc * 100.0,
                )

            # Save best model
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                patience_counter = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "val_acc": best_val_acc,
                        "val_loss": epoch_val_loss,
                        "num_classes": num_classes,
                        "input_size": input_dim,
                    },
                    cfg.model_path,
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered at epoch %d", epoch)
                    break

        training_time = round(time.time() - start_time, 2)
        logger.info("Training completed in %.2f seconds. Best Validation Acc: %.2f%%", training_time, best_val_acc * 100.0)

        # 6. Save Metadata & History
        metadata = {
            "architecture": cfg.model_architecture,
            "input_size": input_dim,
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_layers,
            "sequence_length": seq_len,
            "num_classes": num_classes,
            "classes": labels_dict,
            "best_val_acc": round(best_val_acc, 4),
            "total_epochs_trained": len(history["train_loss"]),
            "training_time_sec": training_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(cfg.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        with open(cfg.get("model.history_path", "models/activity_model/training_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

        # 7. Generate Training Curves Plot
        plot_training_history(history, results_dir / "training_curves.png")

        logger.info("Model and training artifacts saved successfully.")
        return True

    finally:
        logger.removeHandler(file_handler)
        file_handler.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_model()
