"""
Tests for ExperimentLogger (SQLite + export).

Covers: experiment lifecycle, step logging, event logging, exports.
"""

import json
import os
import sys
import tempfile
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger import ExperimentLogger
from src.config import Config


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database path."""
    return str(tmp_path / "test_experiments.db")


@pytest.fixture
def logger_instance(temp_db):
    """Create an ExperimentLogger with a temp database."""
    config = Config()
    log = ExperimentLogger.__new__(ExperimentLogger)
    log.db_path = Path(temp_db)
    log.db_path.parent.mkdir(parents=True, exist_ok=True)
    log._lock = threading.Lock()
    log._init_db()
    return log


# ── Test: Database Initialization ───────────────────────────────────


def test_db_created(logger_instance, temp_db):
    """Database file should be created on initialization."""
    assert os.path.exists(temp_db)


# ── Test: Experiment Lifecycle ──────────────────────────────────────


def test_start_experiment(logger_instance):
    """Starting an experiment should return an ID."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    assert isinstance(exp_id, int)
    assert exp_id > 0


def test_complete_experiment(logger_instance):
    """Completing an experiment should update its status."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    logger_instance.complete_experiment(exp_id, status="COMPLETED")

    log = logger_instance.get_experiment_log(exp_id)
    assert log is not None
    assert log["status"] == "COMPLETED"


# ── Test: Step Logging ──────────────────────────────────────────────


def test_log_step(logger_instance):
    """Logging a step should persist to database."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    logger_instance.log_step(
        experiment_id=exp_id,
        step_index=0,
        step_name="Approach Rack",
        expected="approach_rack",
        detected="approach_rack",
        status="CORRECT",
        confidence=0.95,
    )

    log = logger_instance.get_experiment_log(exp_id)
    assert len(log["steps"]) == 1
    assert log["steps"][0]["status"] == "CORRECT"


def test_log_multiple_steps(logger_instance):
    """Multiple steps should all be recorded."""
    exp_id = logger_instance.start_experiment("Test Experiment")

    steps = [
        ("Approach Rack", "approach_rack", "CORRECT", 0.95),
        ("Pick Container", "pick_container", "CORRECT", 0.91),
        ("Open Container", "open_container", "SKIPPED", 0.88),
    ]

    for i, (name, activity, status, conf) in enumerate(steps):
        logger_instance.log_step(
            experiment_id=exp_id,
            step_index=i,
            step_name=name,
            expected=activity,
            detected=activity,
            status=status,
            confidence=conf,
        )

    log = logger_instance.get_experiment_log(exp_id)
    assert len(log["steps"]) == 3


# ── Test: Event Logging ─────────────────────────────────────────────


def test_log_event(logger_instance):
    """Events should be recorded with metadata."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    logger_instance.log_event(
        experiment_id=exp_id,
        event_type="ASTRONAUT_DETECTED",
        message="Astronaut entered frame",
        data={"confidence": 0.96},
    )

    log = logger_instance.get_experiment_log(exp_id)
    assert len(log["events"]) == 1
    assert log["events"][0]["event_type"] == "ASTRONAUT_DETECTED"


# ── Test: JSON Export ───────────────────────────────────────────────


def test_export_json(logger_instance, tmp_path):
    """JSON export should produce valid JSON file."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    logger_instance.log_step(
        experiment_id=exp_id,
        step_index=0,
        step_name="Approach Rack",
        expected="approach_rack",
        detected="approach_rack",
        status="CORRECT",
        confidence=0.95,
    )
    logger_instance.complete_experiment(exp_id)

    output_path = str(tmp_path / "export.json")
    logger_instance.export_json(exp_id, output_path)

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        data = json.load(f)
    assert data["name"] == "Test Experiment"


# ── Test: CSV Export ────────────────────────────────────────────────


def test_export_csv(logger_instance, tmp_path):
    """CSV export should produce a valid CSV file."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    logger_instance.log_step(
        experiment_id=exp_id,
        step_index=0,
        step_name="Approach Rack",
        expected="approach_rack",
        detected="approach_rack",
        status="CORRECT",
        confidence=0.95,
    )

    output_path = str(tmp_path / "export.csv")
    logger_instance.export_csv(exp_id, output_path)

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
    assert "Approach Rack" in content


# ── Test: TXT Export ────────────────────────────────────────────────


def test_export_txt(logger_instance, tmp_path):
    """TXT export should produce a human-readable report."""
    exp_id = logger_instance.start_experiment("Test Experiment")
    logger_instance.log_step(
        experiment_id=exp_id,
        step_index=0,
        step_name="Approach Rack",
        expected="approach_rack",
        detected="approach_rack",
        status="CORRECT",
        confidence=0.95,
    )
    logger_instance.complete_experiment(exp_id)

    output_path = str(tmp_path / "report.txt")
    logger_instance.export_txt(exp_id, output_path)

    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        content = f.read()
    assert "BAS EXPERIMENT LOG" in content or "EXPERIMENT LOG" in content
    assert "Approach Rack" in content


# ── Test: Get All Experiments ───────────────────────────────────────


def test_get_all_experiments(logger_instance):
    """Should list all experiments."""
    logger_instance.start_experiment("Experiment 1")
    logger_instance.start_experiment("Experiment 2")

    experiments = logger_instance.get_all_experiments()
    assert len(experiments) == 2
