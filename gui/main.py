"""
BAS-HAR Main Tkinter Graphical User Interface

Assembles the video canvas, prediction card, activity timeline, and controls.
Polls background inference worker asynchronously for smooth real-time rendering.
"""

import json
import logging
from pathlib import Path
from queue import Queue, Empty
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config
from gui.widgets import (
    VideoCanvasWidget,
    PredictionPanelWidget,
    TimelineWidget,
    ControlPanelWidget,
    BG_DARK,
    BG_CARD,
    TEXT_PRIMARY,
    TEXT_MUTED,
    ACCENT_GREEN,
    ACCENT_RED,
    ACCENT_YELLOW,
    BORDER_COLOR,
)
from gui.inference_thread import InferenceWorker

logger = logging.getLogger("BAS-HAR.GUI")


class BASHARApp:
    """Main Application Window for BAS-HAR System."""

    def __init__(self, root: tk.Tk, config: Optional[Config] = None):
        self.root = root
        self.config = config or Config()

        # Window Setup
        self.root.title(self.config.get("gui.title", "BAS-HAR: Offline Human Activity Recognition"))
        self.root.geometry(f"{self.config.get('gui.width', 1360)}x{self.config.get('gui.height', 820)}")
        self.root.minsize(1080, 700)
        self.root.configure(bg=BG_DARK)

        # Worker State
        self.frame_queue: Queue = Queue(maxsize=2)
        self.event_queue: Queue = Queue(maxsize=100)
        self.worker: Optional[InferenceWorker] = None
        self.current_source: Union[int, str] = self.config.camera_source

        # Setup Styles & UI Layout
        self._setup_styles()
        self._build_ui()

        # Check model status initially
        self._check_initial_model_status()

        # Start queue polling loop
        self.root.after(30, self._poll_queues)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_styles(self) -> None:
        """Configure ttk dark styles."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=BG_DARK)
        style.configure("Card.TFrame", background=BG_CARD)
        style.configure(
            "Treeview",
            background="#1e242c",
            foreground=TEXT_PRIMARY,
            fieldbackground="#1e242c",
            rowheight=26,
            font=("Segoe UI", 9),
            bordercolor=BORDER_COLOR,
        )
        style.configure(
            "Treeview.Heading",
            background="#2a333f",
            foreground=TEXT_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#3b82f6")])

    def _build_ui(self) -> None:
        """Construct application layout."""
        # Top Header Bar
        header = tk.Frame(self.root, bg=BG_CARD, padx=20, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        header.pack(fill=tk.X, side=tk.TOP)

        title_lbl = tk.Label(
            header,
            text="BAS-HAR",
            font=("Segoe UI", 16, "bold"),
            fg="#60A5FA",
            bg=BG_CARD,
        )
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = tk.Label(
            header,
            text="— Offline Human Activity Recognition Framework",
            font=("Segoe UI", 12),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        subtitle_lbl.pack(side=tk.LEFT, padx=8)

        self.lbl_global_status = tk.Label(
            header,
            text="Engine Ready",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_GREEN,
            bg=BG_CARD,
        )
        self.lbl_global_status.pack(side=tk.RIGHT)

        # Main Content Layout: Split into Left (Video & Controls) and Right (Prediction & Timeline)
        content = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=16)
        content.pack(fill=tk.BOTH, expand=True)

        left_col = tk.Frame(content, bg=BG_DARK)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        right_col = tk.Frame(content, bg=BG_DARK, width=440)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        right_col.pack_propagate(False)

        # ── Left Column Components ──────────────────────────────────
        self.video_canvas = VideoCanvasWidget(left_col, width=800, height=480)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        self.control_panel = ControlPanelWidget(
            left_col,
            on_start_camera=self.start_camera,
            on_stop_camera=self.stop_stream,
            on_load_video=self.load_video,
            on_load_model=self.load_model,
            on_export_csv=self.export_csv,
        )
        self.control_panel.pack(fill=tk.X, side=tk.BOTTOM)

        # ── Right Column Components ─────────────────────────────────
        self.prediction_panel = PredictionPanelWidget(right_col)
        self.prediction_panel.pack(fill=tk.X, pady=(0, 12))

        self.timeline_widget = TimelineWidget(right_col)
        self.timeline_widget.pack(fill=tk.BOTH, expand=True)

    def _check_initial_model_status(self) -> None:
        """Check if HAR model weights exist and update status badge."""
        if not self.config.model_path.exists():
            self.lbl_global_status.config(
                text="HAR model not found. Please train a model or load a valid model.",
                fg=ACCENT_YELLOW,
            )
            self.prediction_panel.update_prediction(
                activity="No Model Loaded",
                confidence=0.0,
                model_name=self.config.model_architecture,
                status="Missing Model",
                status_color=ACCENT_YELLOW,
            )

    def start_camera(self) -> None:
        """Start inference worker with live camera source."""
        self.stop_stream()
        self.current_source = self.config.camera_source
        self.lbl_global_status.config(text=f"Live Camera ({self.current_source})", fg=ACCENT_GREEN)
        self._start_worker(self.current_source)

    def load_video(self) -> None:
        """Prompt user for a video file and start inference."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        self.stop_stream()
        self.current_source = file_path
        self.lbl_global_status.config(text=f"Playing: {Path(file_path).name}", fg=ACCENT_GREEN)
        self._start_worker(self.current_source)

    def _start_worker(self, source: Union[int, str]) -> None:
        """Launch background worker thread."""
        self.worker = InferenceWorker(
            config=self.config,
            frame_queue=self.frame_queue,
            event_queue=self.event_queue,
            source=source,
        )
        self.worker.start()

    def stop_stream(self) -> None:
        """Terminate active worker and reset canvas."""
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=1.0)
            self.worker = None

        self.video_canvas.show_placeholder("Stream Stopped\nSelect 'Start Camera' or 'Load Video'")
        self.lbl_global_status.config(text="Stream Stopped", fg=TEXT_MUTED)
        self.prediction_panel.update_prediction(
            activity="Idle",
            confidence=0.0,
            fps=0.0,
            status="Stopped",
            status_color=TEXT_MUTED,
        )

    def load_model(self) -> None:
        """Prompt user to select an activity_lstm.pth model checkpoint."""
        file_path = filedialog.askopenfilename(
            title="Select HAR Model Checkpoint",
            filetypes=[("PyTorch Model", "*.pth *.pt"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        self.config.set("model.model_path", file_path)
        if self.worker and self.worker.is_alive():
            reloaded = self.worker.reload_model()
        else:
            reloaded = True

        if reloaded:
            messagebox.showinfo("Model Loaded", f"Model loaded successfully from:\n{file_path}")
            self.lbl_global_status.config(text=f"Model: {Path(file_path).name}", fg=ACCENT_GREEN)
        else:
            messagebox.showwarning(
                "Model Warning",
                "Model file loaded, but labels.json was missing or mismatched.",
            )

    def export_csv(self) -> None:
        """Export timeline events to user-selected CSV path."""
        target_path = filedialog.asksaveasfilename(
            title="Export Timeline CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="activity_timeline_export.csv",
        )
        if not target_path:
            return

        # Copy timeline CSV or dump current table items
        try:
            with open(target_path, "w", newline="", encoding="utf-8") as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(["time", "activity", "duration", "confidence"])
                for item_id in self.timeline_widget.tree.get_children():
                    vals = self.timeline_widget.tree.item(item_id, "values")
                    writer.writerow(vals)
            messagebox.showinfo("Export Successful", f"Activity timeline exported to:\n{target_path}")
        except Exception as e:
            messagebox.showerror("Export Failed", f"Could not write CSV: {e}")

    def _poll_queues(self) -> None:
        """Periodically consume frames and events from worker queue without blocking."""
        # Process Frame
        try:
            payload = self.frame_queue.get_nowait()
            frame = payload["frame"]
            pred = payload["prediction"]
            model_ready = payload["model_ready"]
            err_msg = payload.get("error_msg")

            self.video_canvas.update_frame(frame)

            if model_ready:
                status_color = ACCENT_GREEN if pred.is_known else ACCENT_YELLOW
                status_text = "Recognizing" if pred.is_known else "Low Confidence"
                self.prediction_panel.update_prediction(
                    activity=pred.activity,
                    confidence=pred.confidence,
                    fps=pred.fps,
                    latency_ms=pred.processing_time_ms,
                    model_name=self.config.model_architecture,
                    status=status_text,
                    status_color=status_color,
                )
            else:
                self.prediction_panel.update_prediction(
                    activity="No Model Loaded",
                    confidence=0.0,
                    fps=pred.fps,
                    latency_ms=pred.processing_time_ms,
                    model_name=self.config.model_architecture,
                    status="Missing Weights",
                    status_color=ACCENT_YELLOW,
                )
                if err_msg:
                    self.lbl_global_status.config(text=err_msg, fg=ACCENT_YELLOW)

        except Empty:
            pass

        # Process Timeline Events
        while True:
            try:
                event = self.event_queue.get_nowait()
                self.timeline_widget.add_event(
                    start_time_str=event.start_str,
                    activity=event.activity,
                    duration_sec=event.duration_sec,
                    confidence=event.confidence,
                )
            except Empty:
                break

        # Schedule next poll
        self.root.after(20, self._poll_queues)

    def on_close(self) -> None:
        """Clean shutdown when closing window."""
        self.stop_stream()
        self.root.destroy()


def run_gui() -> None:
    """Launch Tkinter application."""
    root = tk.Tk()
    app = BASHARApp(root)
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_gui()
