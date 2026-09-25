import json
import logging
import sqlite3
import threading
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """
    Experiment Logger with SQLite.
    Handles thread-safe logging of experiment steps, events, and results.
    """
    def __init__(self, config: Config):
        self.config = config
        self.db_path = getattr(config, 'db_path', PROJECT_ROOT / "logs" / "experiments.db")
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS experiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        start_time TEXT NOT NULL,
                        end_time TEXT,
                        status TEXT NOT NULL,
                        total_steps INTEGER DEFAULT 0,
                        completed_steps INTEGER DEFAULT 0,
                        error_count INTEGER DEFAULT 0
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id INTEGER NOT NULL,
                        step_index INTEGER NOT NULL,
                        step_name TEXT NOT NULL,
                        expected_activity TEXT,
                        detected_activity TEXT,
                        status TEXT,
                        confidence REAL,
                        timestamp TEXT NOT NULL,
                        error_type TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id INTEGER NOT NULL,
                        event_type TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        data_json TEXT,
                        FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                    )
                ''')
                conn.commit()

    def start_experiment(self, name: str) -> int:
        now_str = datetime.now().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO experiments (name, start_time, status) VALUES (?, ?, ?)",
                    (name, now_str, 'RUNNING')
                )
                experiment_id = cursor.lastrowid
                conn.commit()
                return experiment_id

    def log_step(self, experiment_id: int, step_index: int, step_name: str, 
                 expected: str, detected: str, status: str, confidence: float, error_type: str = ''):
        now_str = datetime.now().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO steps (experiment_id, step_index, step_name, expected_activity, detected_activity, status, confidence, timestamp, error_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (experiment_id, step_index, step_name, expected, detected, status, confidence, now_str, error_type))
                
                if status == 'COMPLETED':
                    cursor.execute("UPDATE experiments SET completed_steps = completed_steps + 1 WHERE id = ?", (experiment_id,))
                if error_type:
                    cursor.execute("UPDATE experiments SET error_count = error_count + 1 WHERE id = ?", (experiment_id,))
                
                conn.commit()

    def log_event(self, experiment_id: int, event_type: str, message: str, data: dict | None = None):
        now_str = datetime.now().isoformat()
        data_json = json.dumps(data) if data else None
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO events (experiment_id, event_type, message, timestamp, data_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (experiment_id, event_type, message, now_str, data_json))
                conn.commit()

    def complete_experiment(self, experiment_id: int, status: str = 'COMPLETED'):
        now_str = datetime.now().isoformat()
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE experiments SET status = ?, end_time = ? WHERE id = ?", (status, now_str, experiment_id))
                conn.commit()

    def get_experiment_log(self, experiment_id: int) -> dict:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
                exp_row = cursor.fetchone()
                if not exp_row:
                    return {}
                
                result = dict(exp_row)
                
                cursor.execute("SELECT * FROM steps WHERE experiment_id = ? ORDER BY timestamp", (experiment_id,))
                result['steps'] = [dict(row) for row in cursor.fetchall()]
                
                cursor.execute("SELECT * FROM events WHERE experiment_id = ? ORDER BY timestamp", (experiment_id,))
                result['events'] = [dict(row) for row in cursor.fetchall()]
                
                return result

    def export_json(self, experiment_id: int, output_path: str):
        data = self.get_experiment_log(experiment_id)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def export_csv(self, experiment_id: int, output_path: str):
        data = self.get_experiment_log(experiment_id)
        if not data:
            return
            
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Experiment ID', 'Name', 'Start Time', 'End Time', 'Status'])
            writer.writerow([data['id'], data['name'], data['start_time'], data.get('end_time', ''), data['status']])
            writer.writerow([])
            
            writer.writerow(['Step Index', 'Step Name', 'Expected', 'Detected', 'Status', 'Confidence', 'Timestamp', 'Error'])
            for step in data.get('steps', []):
                writer.writerow([
                    step['step_index'], step['step_name'], step['expected_activity'],
                    step['detected_activity'], step['status'], step['confidence'],
                    step['timestamp'], step['error_type']
                ])

    def export_txt(self, experiment_id: int, output_path: str):
        data = self.get_experiment_log(experiment_id)
        if not data:
            return
            
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("BAS EXPERIMENT LOG\n")
            f.write("============================\n")
            f.write(f"Experiment: {data.get('name', 'Unknown')}\n")
            
            all_entries = []
            for step in data.get('steps', []):
                dt = datetime.fromisoformat(step['timestamp'])
                time_str = dt.strftime("%H:%M:%S")
                msg = f"Step {step['step_index']}: {step['step_name']} - {step['status']}"
                if step.get('error_type'):
                    msg += f" ({step['error_type']})"
                all_entries.append((dt, f"{time_str} - {msg}"))
                
            for event in data.get('events', []):
                dt = datetime.fromisoformat(event['timestamp'])
                time_str = dt.strftime("%H:%M:%S")
                all_entries.append((dt, f"{time_str} - EVENT: {event['message']}"))
                
            all_entries.sort(key=lambda x: x[0])
            for _, line in all_entries:
                f.write(line + "\n")
                
            f.write(f"...\n")
            f.write(f"Experiment Status: {data.get('status', 'UNKNOWN')}\n")
            f.write(f"Errors: {data.get('error_count', 0)}\n")

    def get_all_experiments(self) -> list[dict]:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM experiments ORDER BY id DESC")
                return [dict(row) for row in cursor.fetchall()]
