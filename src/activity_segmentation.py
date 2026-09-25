"""
BAS-HAR Activity Segmentation & Timeline Analytics

Detects activity transitions, segments continuous activity durations,
logs events to CSV, and generates experiment analytics.
"""

import csv
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ActivityEvent:
    """Represents a bounded activity segment."""
    activity: str
    start_time: float               # Epoch timestamp
    end_time: float                 # Epoch timestamp
    duration_sec: float
    confidence: float               # Average confidence across the segment
    sample_count: int = 1
    start_str: str = ""
    end_str: str = ""

    def __post_init__(self):
        if not self.start_str:
            self.start_str = datetime.fromtimestamp(self.start_time).strftime("%H:%M:%S")
        if not self.end_str:
            self.end_str = datetime.fromtimestamp(self.end_time).strftime("%H:%M:%S")


class ActivitySegmenter:
    """
    Monitors incoming activity predictions, detects transitions,
    and constructs a chronological timeline of activities.
    """

    def __init__(
        self,
        csv_path: Optional[Union[str, Path]] = None,
        min_duration_sec: float = 0.5,
    ):
        self.csv_path = Path(csv_path) if csv_path else None
        self.min_duration_sec = min_duration_sec

        self.current_activity: Optional[str] = None
        self.current_start_time: float = 0.0
        self.current_confs: List[float] = []

        self.events: List[ActivityEvent] = []
        self.transitions: Dict[Tuple[str, str], int] = {}

        if self.csv_path:
            self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV file with headers if it does not already exist."""
        try:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.csv_path.exists():
                with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "activity", "confidence", "duration"])
        except Exception as e:
            logger.error("Could not initialize timeline CSV: %s", e)

    def update(
        self,
        activity: str,
        confidence: float,
        timestamp: Optional[float] = None,
    ) -> Optional[ActivityEvent]:
        """
        Feed a new activity prediction frame.
        
        Returns:
            ActivityEvent if an activity segment concluded on this frame, else None.
        """
        now = timestamp if timestamp is not None else time.time()
        concluded_event: Optional[ActivityEvent] = None

        if self.current_activity is None:
            # First activity
            self.current_activity = activity
            self.current_start_time = now
            self.current_confs = [confidence]
            return None

        if activity == self.current_activity:
            # Ongoing activity continues
            self.current_confs.append(confidence)
            return None

        # Activity transition detected!
        duration = now - self.current_start_time
        if duration >= self.min_duration_sec:
            avg_conf = float(sum(self.current_confs) / max(1, len(self.current_confs)))
            concluded_event = ActivityEvent(
                activity=self.current_activity,
                start_time=self.current_start_time,
                end_time=now,
                duration_sec=round(duration, 2),
                confidence=round(avg_conf, 4),
                sample_count=len(self.current_confs),
            )
            self.events.append(concluded_event)

            # Record transition
            transition_key = (self.current_activity, activity)
            self.transitions[transition_key] = self.transitions.get(transition_key, 0) + 1

            # Log to CSV
            self._write_csv_event(concluded_event)

        # Switch to new activity
        self.current_activity = activity
        self.current_start_time = now
        self.current_confs = [confidence]

        return concluded_event

    def flush(self, timestamp: Optional[float] = None) -> Optional[ActivityEvent]:
        """Flush any currently ongoing activity segment (e.g. at end of stream)."""
        if self.current_activity is None:
            return None

        now = timestamp if timestamp is not None else time.time()
        duration = now - self.current_start_time
        concluded_event: Optional[ActivityEvent] = None

        if duration >= self.min_duration_sec:
            avg_conf = float(sum(self.current_confs) / max(1, len(self.current_confs)))
            concluded_event = ActivityEvent(
                activity=self.current_activity,
                start_time=self.current_start_time,
                end_time=now,
                duration_sec=round(duration, 2),
                confidence=round(avg_conf, 4),
                sample_count=len(self.current_confs),
            )
            self.events.append(concluded_event)
            self._write_csv_event(concluded_event)

        self.current_activity = None
        self.current_confs = []
        return concluded_event

    def _write_csv_event(self, event: ActivityEvent) -> None:
        """Append an event row to the timeline CSV."""
        if not self.csv_path:
            return
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.start_str,
                    event.activity,
                    event.confidence,
                    event.duration_sec,
                ])
        except Exception as e:
            logger.error("Failed to append event to CSV: %s", e)

    def get_summary_statistics(self) -> Dict[str, dict]:
        """
        Compute experiment analytics:
        - Per-activity duration and frequency
        - Activity transitions
        """
        stats: Dict[str, dict] = {}
        for ev in self.events:
            if ev.activity not in stats:
                stats[ev.activity] = {
                    "count": 0,
                    "total_duration_sec": 0.0,
                    "avg_duration_sec": 0.0,
                    "avg_confidence": 0.0,
                }
            s = stats[ev.activity]
            s["count"] += 1
            s["total_duration_sec"] += ev.duration_sec

        for act, s in stats.items():
            s["avg_duration_sec"] = round(s["total_duration_sec"] / max(1, s["count"]), 2)
            s["total_duration_sec"] = round(s["total_duration_sec"], 2)

        return stats
