"""
Train LSTM/GRU Activity Classifier for BAS-HAR Project.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ACTIVITY_CLASSES = [
    "approach_rack", "pick_container", "open_container", "remove_sample",
    "place_sample", "press_button", "close_container", "return_tool"
]
ACTIVITY_TO_IDX = {name: idx for idx, name in enumerate(ACTIVITY_CLASSES)}

class ActivityDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist.")
            return

        for json_file in self.data_dir.glob("*.json"):
            with open(json_file, 'r') as f:
                try:
                    data = json.load(f)
                    if 'activity' in data and 'keypoint_sequence' in data:
                        activity = data['activity']
                        if activity in ACTIVITY_TO_IDX:
                            self.samples.append((data['keypoint_sequence'], ACTIVITY_TO_IDX[activity]))
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def get_model(input_dim, hidden_dim, num_layers, num_classes, rnn_type='LSTM'):
    if rnn_type.upper() == 'LSTM':
        class LSTMClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, num_classes)
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        return LSTMClassifier()
    elif rnn_type.upper() == 'GRU':
        class GRUClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_dim, num_classes)
            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :])
        return GRUClassifier()
    else:
        raise ValueError(f"Unknown RNN type: {rnn_type}")

def main():
    parser = argparse.ArgumentParser(description="Train Activity Classifier")
    parser.add_argument("--data-dir", type=str, default=str(PROJECT_ROOT / "data" / "activity_sequences"), help="Training data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--rnn-type", type=str, default="LSTM", choices=["LSTM", "GRU"], help="RNN type")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    dataset = ActivityDataset(args.data_dir)
    if len(dataset) == 0:
        logger.error(f"No training data found in {args.data_dir}")
        sys.exit(1)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    input_dim = len(dataset[0][0][0]) if len(dataset[0][0]) > 0 else 34
    num_classes = len(ACTIVITY_CLASSES)
    
    model = get_model(input_dim, args.hidden_dim, args.num_layers, num_classes, args.rnn_type).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float('inf')
    model_dir = PROJECT_ROOT / "models" / "activity_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"activity_{args.rnn_type.lower()}.pth"
    log_path = model_dir / "training_log.json"
    
    training_log = []

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        training_log.append({"epoch": epoch+1, "loss": avg_loss, "accuracy": accuracy})
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved better model to {model_path}")

    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
        
    logger.info(f"Training complete. Log saved to {log_path}")

if __name__ == '__main__':
    main()
