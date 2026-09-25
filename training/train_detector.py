"""
Train YOLO Object Detector for BAS-HAR Project.
"""
import os
import sys
import argparse
import logging
import shutil
from pathlib import Path

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    from ultralytics import YOLO
except ImportError:
    logger.error("ultralytics is not installed. Please install it using 'pip install ultralytics'.")
    YOLO = None

def main():
    parser = argparse.ArgumentParser(description="Train YOLO Object Detector")
    parser.add_argument("--data", type=str, default=str(PROJECT_ROOT / "data" / "dataset.yaml"), help="Path to dataset.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device (cuda or cpu)")
    parser.add_argument("--project", type=str, default=str(PROJECT_ROOT / "models" / "object_detector"), help="Output directory")
    parser.add_argument("--name", type=str, default="run", help="Run name")
    args = parser.parse_args()

    if YOLO is None:
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset file {data_path} not found.")
        sys.exit(1)

    logger.info(f"Initializing YOLO model with {args.model}")
    model = YOLO(args.model)

    logger.info("Starting training...")
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=True
    )

    logger.info("Training complete.")
    best_pt = Path(args.project) / args.name / 'weights' / 'best.pt'
    logger.info(f"Best model saved to {best_pt}")

    # Copy best model to models/object_detector/best.pt
    target_pt = PROJECT_ROOT / "models" / "object_detector" / "best.pt"
    if best_pt.exists():
        target_pt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, target_pt)
        logger.info(f"Copied best model to {target_pt}")

if __name__ == '__main__':
    main()
