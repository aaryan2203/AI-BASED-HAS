"""
BAS-HAR GUI Modular Widgets

Provides styled Tkinter components:
- VideoCanvasWidget: Resizes and renders camera/video frames smoothly.
- PredictionPanelWidget: Displays current activity, confidence, FPS, model name, and engine status.
- TimelineWidget: Interactive Treeview table displaying segmented activity events.
- ControlPanelWidget: Buttons for camera, video files, model loading, and CSV export.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import cv2
import numpy as np
from PIL import Image, ImageTk

# Color Palette (Modern Dark Theme)
BG_DARK = "#121417"
BG_CARD = "#1a1f26"
BG_SURFACE = "#222831"
TEXT_PRIMARY = "#FFFFFF"
TEXT_MUTED = "#8E9AA8"
ACCENT_GREEN = "#10B981"
ACCENT_BLUE = "#3B82F6"
ACCENT_YELLOW = "#F59E0B"
ACCENT_RED = "#EF4444"
BORDER_COLOR = "#2D3748"


class VideoCanvasWidget(ttk.Frame):
    """Canvas widget for displaying video frames with aspect ratio preservation."""

    def __init__(self, parent, width: int = 800, height: int = 480):
        super().__init__(parent)
        self.target_width = width
        self.target_height = height

        self.canvas = tk.Canvas(
            self,
            width=self.target_width,
            height=self.target_height,
            bg="#0d1117",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._current_photo: Optional[ImageTk.PhotoImage] = None
        self.show_placeholder("No Video Stream Active\nSelect 'Start Camera' or 'Load Video'")

    def show_placeholder(self, message: str) -> None:
        """Display placeholder text when no stream is playing."""
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or self.target_width
        h = self.canvas.winfo_height() or self.target_height
        self.canvas.create_text(
            w // 2,
            h // 2,
            text=message,
            fill=TEXT_MUTED,
            font=("Segoe UI", 14, "bold"),
            justify=tk.CENTER,
            tags="placeholder",
        )

    def update_frame(self, bgr_frame: np.ndarray) -> None:
        """Render a new BGR video frame onto the canvas."""
        if bgr_frame is None or bgr_frame.size == 0:
            return

        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())

        fh, fw = bgr_frame.shape[:2]
        # Calculate aspect ratio scaling
        scale = min(cw / fw, ch / fh)
        nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))

        resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        self._current_photo = ImageTk.PhotoImage(image=img)

        self.canvas.delete("all")
        # Center the image on canvas
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._current_photo)


class PredictionPanelWidget(ttk.Frame):
    """Panel displaying real-time activity prediction, confidence, FPS, and status."""

    def __init__(self, parent):
        super().__init__(parent)
        self.configure(style="Card.TFrame")

        card = tk.Frame(self, bg=BG_CARD, padx=20, pady=16, highlightthickness=1, highlightbackground=BORDER_COLOR)
        card.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            card,
            text="CURRENT ACTIVITY",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        title.pack(anchor="w")

        # Large Activity Label
        self.lbl_activity = tk.Label(
            card,
            text="Unknown",
            font=("Segoe UI", 26, "bold"),
            fg=ACCENT_BLUE,
            bg=BG_CARD,
        )
        self.lbl_activity.pack(anchor="w", pady=(4, 12))

        # Confidence Bar and Text
        conf_row = tk.Frame(card, bg=BG_CARD)
        conf_row.pack(fill=tk.X, pady=(0, 12))

        tk.Label(
            conf_row,
            text="Confidence:",
            font=("Segoe UI", 10, "bold"),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        ).pack(side=tk.LEFT)

        self.lbl_confidence = tk.Label(
            conf_row,
            text="0.0%",
            font=("Segoe UI", 12, "bold"),
            fg=TEXT_PRIMARY,
            bg=BG_CARD,
        )
        self.lbl_confidence.pack(side=tk.RIGHT)

        # Meta Details Grid
        grid_frame = tk.Frame(card, bg=BG_CARD)
        grid_frame.pack(fill=tk.X, pady=(4, 0))

        # Row 1: FPS
        tk.Label(grid_frame, text="FPS:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.lbl_fps = tk.Label(grid_frame, text="0.0", font=("Segoe UI", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.lbl_fps.grid(row=0, column=1, sticky="w", padx=10, pady=2)

        # Row 2: Latency
        tk.Label(grid_frame, text="Latency:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.lbl_latency = tk.Label(grid_frame, text="0 ms", font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.lbl_latency.grid(row=1, column=1, sticky="w", padx=10, pady=2)

        # Row 3: Model
        tk.Label(grid_frame, text="Model:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.lbl_model = tk.Label(grid_frame, text="HAR-LSTM", font=("Segoe UI", 9), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.lbl_model.grid(row=2, column=1, sticky="w", padx=10, pady=2)

        # Row 4: Status
        tk.Label(grid_frame, text="Status:", font=("Segoe UI", 9), fg=TEXT_MUTED, bg=BG_CARD).grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.lbl_status = tk.Label(
            grid_frame, text="Ready", font=("Segoe UI", 9, "bold"), fg=ACCENT_GREEN, bg=BG_CARD
        )
        self.lbl_status.grid(row=3, column=1, sticky="w", padx=10, pady=2)

    def update_prediction(
        self,
        activity: str,
        confidence: float,
        fps: float = 0.0,
        latency_ms: float = 0.0,
        model_name: str = "HAR-LSTM",
        status: str = "Running",
        status_color: str = ACCENT_GREEN,
    ) -> None:
        """Update prediction card values."""
        act_display = activity.replace("_", " ").title()
        self.lbl_activity.config(text=act_display)

        conf_pct = confidence * 100.0
        self.lbl_confidence.config(text=f"{conf_pct:.1f}%")

        # Color-code based on confidence
        if conf_pct >= 75.0:
            self.lbl_activity.config(fg=ACCENT_GREEN)
        elif conf_pct >= 50.0:
            self.lbl_activity.config(fg=ACCENT_YELLOW)
        else:
            self.lbl_activity.config(fg=TEXT_MUTED)

        self.lbl_fps.config(text=f"{fps:.1f}")
        self.lbl_latency.config(text=f"{latency_ms:.1f} ms")
        self.lbl_model.config(text=model_name)
        self.lbl_status.config(text=status, fg=status_color)


class TimelineWidget(ttk.Frame):
    """Activity timeline event table displaying chronological activity segments."""

    def __init__(self, parent):
        super().__init__(parent)

        container = tk.Frame(self, bg=BG_CARD, padx=14, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        container.pack(fill=tk.BOTH, expand=True)

        header = tk.Label(
            container,
            text="ACTIVITY TIMELINE",
            font=("Segoe UI", 11, "bold"),
            fg=TEXT_MUTED,
            bg=BG_CARD,
        )
        header.pack(anchor="w", pady=(0, 8))

        columns = ("time", "activity", "duration", "confidence")
        self.tree = ttk.Treeview(container, columns=columns, show="headings", height=8)

        self.tree.heading("time", text="Start Time")
        self.tree.heading("activity", text="Activity")
        self.tree.heading("duration", text="Duration (s)")
        self.tree.heading("confidence", text="Confidence")

        self.tree.column("time", width=85, anchor="center")
        self.tree.column("activity", width=140, anchor="w")
        self.tree.column("duration", width=95, anchor="center")
        self.tree.column("confidence", width=95, anchor="center")

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def add_event(self, start_time_str: str, activity: str, duration_sec: float, confidence: float) -> None:
        """Append an activity event row to the timeline table."""
        conf_str = f"{confidence * 100.0:.1f}%"
        dur_str = f"{duration_sec:.1f}s"
        item_id = self.tree.insert(
            "",
            0,  # Insert at top (most recent first)
            values=(start_time_str, activity.replace("_", " ").title(), dur_str, conf_str),
        )

    def clear(self) -> None:
        """Clear all rows."""
        for item in self.tree.get_children():
            self.tree.delete(item)


class ControlPanelWidget(ttk.Frame):
    """Button bar providing video controls, model reloading, and export actions."""

    def __init__(
        self,
        parent,
        on_start_camera: Callable,
        on_stop_camera: Callable,
        on_load_video: Callable,
        on_load_model: Callable,
        on_export_csv: Callable,
    ):
        super().__init__(parent)
        self.configure(style="Card.TFrame")

        card = tk.Frame(self, bg=BG_CARD, padx=14, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        card.pack(fill=tk.X, expand=True)

        btn_style = {"bg": BG_SURFACE, "fg": TEXT_PRIMARY, "font": ("Segoe UI", 9, "bold"), "relief": tk.FLAT, "padx": 12, "pady": 6}

        self.btn_camera = tk.Button(card, text="Start Camera", command=on_start_camera, **btn_style)
        self.btn_camera.pack(side=tk.LEFT, padx=4)

        self.btn_stop = tk.Button(card, text="Stop Stream", command=on_stop_camera, **btn_style)
        self.btn_stop.pack(side=tk.LEFT, padx=4)

        self.btn_video = tk.Button(card, text="Load Video", command=on_load_video, **btn_style)
        self.btn_video.pack(side=tk.LEFT, padx=4)

        self.btn_model = tk.Button(card, text="Load Model", command=on_load_model, **btn_style)
        self.btn_model.pack(side=tk.LEFT, padx=4)

        self.btn_export = tk.Button(card, text="Export CSV", command=on_export_csv, **btn_style)
        self.btn_export.pack(side=tk.RIGHT, padx=4)
