# BAS-HAR: Human Activity Recognition for On-Board BAS Experiments

AI-powered onboard assistant for monitoring astronaut activities during predefined scientific experiments on a space station. Processes camera data locally at the edge — no cloud APIs required.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 2. System Architecture

```text
                  ┌─────────────────────────┐
                  │ Camera / Video / File   │
                  └────────────┬────────────┘
                               │ BGR Frame
                               ▼
                  ┌─────────────────────────┐
                  │ Frame Preprocessing     │
                  └────────────┬────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       ┌───────────────┐               ┌───────────────┐
       │ Pose Model    │               │ Object Model  │
       │  (YOLO-Pose)  │               │    (YOLO)     │
       └───────┬───────┘               └───────┬───────┘
               │ 17 Keypoints                  │ Bboxes & Classes
               └───────────────┬───────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │ Feature Extraction      │
                  │ - Normalized Coords     │
                  │ - Joint Angles          │
                  │ - Relative Distances    │
                  │ - Keypoint Velocities   │
                  │ - Optional Obj Proximity│
                  └────────────┬────────────┘
                               │ 1D Feature Vector (100 / 116 dims)
                               ▼
                  ┌─────────────────────────┐
                  │ Temporal Sequence Buffer│ (Sliding window: 30 frames)
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │ HAR Deep Learning Model │
                  │      (LSTM / GRU)       │
                  └────────────┬────────────┘
                               ▼
                  ┌─────────────────────────┐
                  │ Prediction Smoothing    │
                  │ & Confidence Filter     │ (< threshold -> 'Unknown')
                  └────────────┬────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       ┌───────────────┐               ┌───────────────┐
       │ Modern Tkinter│               │ Activity Log  │
       │   GUI / CLI   │               │ & CSV Timeline│
       └───────────────┘               └───────────────┘
```

## Features

| Feature | Status | Description |
|---------|--------|-------------|
| Person Detection | ✅ Ready | YOLO-based astronaut detection |
| Object Detection | ✅ Ready | Configurable experiment objects (COCO pretrained + custom training) |
| Pose Estimation | ✅ Ready | 17-keypoint COCO skeleton tracking |
| Hand-Object Interaction | ✅ Ready | Spatial proximity + temporal tracking |
| Activity Recognition | ✅ Ready | LSTM/GRU model + rule-based demo mode |
| Sequence Validation | ✅ Ready | State-machine with temporal confirmation |
| Voice Alerts | ✅ Ready | Offline TTS (English + Hindi) |
| Experiment Logging | ✅ Ready | SQLite + JSON/CSV/TXT export |
| Video Recording | ✅ Ready | Timestamped MP4 recording |
| Network Streaming | ✅ Ready | MJPEG over HTTP |
| GUI Dashboard | ✅ Ready | PySide6 dark-themed dashboard |
| HMR/3D Pose | 🔬 Stub | Interface defined, see `models/hmr/README.md` |

## 3. Project Structure

```text
BAS-HAR/
│
├── data/
│   ├── raw/                       # Raw video files grouped by activity class
│   │   ├── walking/
│   │   ├── sitting/
│   │   ├── standing/
│   │   └── reaching/
│   ├── processed/                 # Metadata and dataset summaries
│   └── sequences/                 # Stratified Train/Val/Test .npz sequence files
│
├── models/
│   ├── activity_model/            # Trained weights, dynamic labels.json, and metadata
│   │   ├── activity_lstm.pth
│   │   ├── labels.json
│   │   ├── metadata.json
│   │   └── training_history.json
│   ├── pose/                      # Offline YOLO pose weights
│   └── object_detection/          # Offline YOLO object detection weights
│
├── src/
│   ├── config.py                  # Centralized configuration manager
│   ├── preprocessing.py           # Pose centering, scale normalization, joint angles
│   ├── pose_estimation.py         # Offline YOLO-pose inference and skeleton overlays
│   ├── object_detection.py        # Modular object detector & interaction features
│   ├── feature_extraction.py      # Feature fusion (100-dim pose, 116-dim pose+object)
│   ├── sequence_builder.py        # Real-time rolling buffer & offline sliding windows
│   ├── activity_recognition.py    # PyTorch LSTM/GRU classifier, smoothing, Unknown filter
│   └── activity_segmentation.py   # Transition detection, duration timing, CSV logging
│
├── training/
│   ├── prepare_dataset.py         # Dataset validation, sequence generation, split export
│   ├── train.py                   # PyTorch training loop, checkpointing, history
│   ├── evaluate.py                # Test set evaluation (Accuracy, Precision, Recall, F1)
│   └── visualize.py               # Curves (loss/acc) and confusion matrix plotting
│
├── gui/
│   ├── main.py                    # Modern Tkinter application
│   ├── inference_thread.py        # Background asynchronous CV/HAR worker thread
│   └── widgets.py                 # Video canvas, prediction card, timeline, controls
│
├── logs/                          # Runtime logs and activity timeline CSV
├── results/                       # Evaluation charts and classification reports
├── tests/                         # Pytest test suite
│
├── config.json                    # Central configuration file
├── requirements.txt               # Dependencies
├── main.py                        # Unified entry point
└── README.md
```

---

## 4. Installation & Setup

1. **Clone or navigate to the repository:**
   ```powershell
   cd C:\Users\Nikhil\.gemini\antigravity\scratch\BAS-HAR
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Verify installation by running test suite:**
   ```powershell
   python -m pytest tests/test_har_system.py -v
   ```

---

## 5. Usage Guide

### A. Launch Graphical User Interface (Default)
```powershell
python main.py
```
* **Video Display**: Real-time camera or video playback with bounding boxes and skeleton overlays.
* **Prediction Panel**: Displays Current Activity, Confidence (%), FPS, Latency (ms), and Engine Status.
* **Activity Timeline**: Interactive table tracking start times, activity names, durations, and confidence levels.
* **Controls**:
  - `Start Camera`: Connects to local webcam.
  - `Load Video`: Browse and run inference on any local video file.
  - `Stop Stream`: Safely pauses playback and inference.
  - `Load Model`: Load a custom `.pth` checkpoint.
  - `Export CSV`: Save the activity timeline table to a `.csv` file.

### B. Command-Line / Headless Inference
Run inference in the terminal without GUI:
```powershell
# Using default webcam
python main.py --cli

# Using recorded video file
python main.py --cli --source path/to/sample_video.mp4
```

### C. Dataset Preparation Pipeline
Place videos in subdirectories named after your activity classes:
```text
data/raw/
├── walking/
│   ├── walk1.mp4
│   └── walk2.mp4
├── reaching/
│   └── reach1.mp4
└── standing/
    └── stand1.mp4
```
Run dataset preparation:
```powershell
python main.py --prepare-data
```
The script validates videos, discovers classes dynamically, generates `labels.json`, extracts temporal features, creates 70/15/15 stratified splits, and saves `.npz` sequence files to `data/sequences/`.

### D. Model Training
Train the recurrent HAR neural network:
```powershell
python main.py --train
```
Artifacts generated:
* `models/activity_model/activity_lstm.pth` (best checkpoint)
* `models/activity_model/metadata.json`
* `models/activity_model/training_history.json`
* `results/training_curves.png` (Loss & Accuracy plots)

### E. Model Evaluation
Evaluate the model on held-out test data:
```powershell
python main.py --evaluate
```
Reports generated:
* Precision, Recall, and F1-Score breakdown per activity class.
* `results/classification_report.txt`
* `results/confusion_matrix.png` (normalized confusion matrix heatmap)

---

## 6. Configuration (`config.json`)

All operational settings are centralized in `config.json`:

```json
{
    "model": {
        "architecture": "LSTM",
        "sequence_length": 30,
        "input_size": 100,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3
    },
    "inference": {
        "confidence_threshold": 0.50,
        "prediction_smoothing": "majority_vote",
        "smoothing_window": 7,
        "unknown_label": "Unknown"
    },
    "pose": {
        "enabled": true,
        "model_path": "yolo11n-pose.pt",
        "confidence_threshold": 0.40
    },
    "object_detection": {
        "enabled": false,
        "model_path": "yolo11n.pt"
    }
}
```

---

## 7. Adding New Activities

To add a new activity (e.g. `pressing`):
1. Create folder `data/raw/pressing/`.
2. Add video clips of the pressing action.
3. Run `python main.py --prepare-data`.


## License

MIT License — Hackathon Prototype

## Authors

Built for BAS (Bharatiya Antariksh Station) hackathon.
4. Run `python main.py --train`.
5. Run `python main.py` to recognize the new activity immediately! No code changes required.
