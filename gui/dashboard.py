import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QSplitter, QMenuBar, QMenu, QStatusBar,
                               QToolBar)
from PySide6.QtGui import QAction, QIcon
import logging
import numpy as np

from src.config import Config, PROJECT_ROOT
from gui.widgets.video_widget import VideoWidget
from gui.widgets.progress_widget import ProgressWidget
from gui.widgets.activity_widget import ActivityWidget
from gui.widgets.status_widget import StatusWidget
from gui.widgets.alert_widget import AlertWidget
from gui.inference_thread import InferenceWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alias for backward compatibility
BASHARDashboard = None  # Will be set at end of module

class DashboardWindow(QMainWindow):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.setWindowTitle(config.window_title if config else "BAS-HAR: Human Activity Recognition")
        self.resize(config.window_width if config else 1400, config.window_height if config else 850)
        self.setMinimumSize(1280, 720)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QMenuBar { background-color: #0f172a; color: #e0e0e0; }
            QMenuBar::item:selected { background-color: #0f3460; }
            QMenu { background-color: #0f172a; color: #e0e0e0; border: 1px solid #333; }
            QMenu::item:selected { background-color: #0f3460; }
            QStatusBar { background-color: #0f172a; color: #e0e0e0; }
            QSplitter::handle { background-color: #0f3460; }
        """)
        
        self._init_ui()
        self._init_menus()
        
        self.worker = InferenceWorker(self.config)
        self._connect_signals()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left Panel (Video)
        self.video_widget = VideoWidget()
        splitter.addWidget(self.video_widget)
        
        # Right Panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        
        self.activity_widget = ActivityWidget()
        self.progress_widget = ProgressWidget()
        self.status_widget = StatusWidget()
        self.alert_widget = AlertWidget()
        
        right_layout.addWidget(self.activity_widget, 1)
        right_layout.addWidget(self.progress_widget, 2)
        right_layout.addWidget(self.status_widget, 1)
        right_layout.addWidget(self.alert_widget, 2)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 300]) # Roughly 70% / 30%
        
        # Status Bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("System Ready")

    def _init_menus(self):
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")
        
        start_act = QAction("Start Experiment", self)
        start_act.triggered.connect(self.start_experiment)
        file_menu.addAction(start_act)
        
        stop_act = QAction("Stop Experiment", self)
        stop_act.triggered.connect(self.stop_experiment)
        file_menu.addAction(stop_act)
        
        file_menu.addSeparator()
        
        exit_act = QAction("Exit", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)
        
        # Settings Menu
        settings_menu = menubar.addMenu("Settings")
        rec_act = QAction("Toggle Recording", self, checkable=True)
        settings_menu.addAction(rec_act)
        
        # Help Menu
        help_menu = menubar.addMenu("Help")
        about_act = QAction("About", self)
        help_menu.addAction(about_act)

    def _connect_signals(self):
        self.worker.frame_ready.connect(self.video_widget.update_frame)
        self.worker.fps_updated.connect(self.video_widget.set_fps)
        self.worker.activity_detected.connect(self.activity_widget.update_activity)
        self.worker.alert_generated.connect(self.alert_widget.add_alert)
        self.worker.status_updated.connect(self.status_widget.update_status)

    def start_experiment(self):
        self.statusBar.showMessage("Starting inference pipeline...")
        self.worker.start()
        self.status_widget.update_status({"Camera": "ONLINE", "AI Model": "ACTIVE"})
        self.alert_widget.add_alert("INFO", "Experiment started")

    def stop_experiment(self):
        self.statusBar.showMessage("Stopping inference pipeline...")
        self.worker.stop()
        self.status_widget.update_status({"Camera": "OFFLINE", "AI Model": "INACTIVE"})
        self.alert_widget.add_alert("INFO", "Experiment stopped")

    def closeEvent(self, event):
        self.stop_experiment()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Mock config for standalone testing
    config = Config()
    
    window = DashboardWindow(config)
    window.show()
    sys.exit(app.exec())

# Alias so main.py can import as BASHARDashboard
BASHARDashboard = DashboardWindow
