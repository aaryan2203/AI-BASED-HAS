"""
Tests for ExperimentManager.

Covers: loading experiment config, step access, validation, error handling.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiment_manager import ExperimentManager


@pytest.fixture
def sample_experiment(tmp_path):
    """Create a temporary experiment JSON file."""
    experiment = {
        "experiment_name": "Test Experiment",
        "experiment_id": "test_v1",
        "description": "A test experiment",
        "version": "1.0",
        "steps": [
            {
                "id": "step_one",
                "name": "Step One",
                "display_name": "First Step",
                "description": "Do the first thing",
                "expected_objects": ["astronaut"],
                "expected_interaction": None,
                "detection_hints": {},
            },
            {
                "id": "step_two",
                "name": "Step Two",
                "display_name": "Second Step",
                "description": "Do the second thing",
                "expected_objects": ["astronaut", "tool"],
                "expected_interaction": "hand_near_tool",
                "detection_hints": {"hand_object_proximity": "tool"},
            },
            {
                "id": "step_three",
                "name": "Step Three",
                "display_name": "Third Step",
                "description": "Do the third thing",
                "expected_objects": ["astronaut"],
                "expected_interaction": None,
                "detection_hints": {},
            },
        ],
        "metadata": {
            "estimated_duration_minutes": 2,
            "difficulty": "easy",
        },
    }

    config_path = tmp_path / "test_experiment.json"
    with open(config_path, "w") as f:
        json.dump(experiment, f)

    return str(config_path)


@pytest.fixture
def manager(sample_experiment):
    """Create an ExperimentManager with the sample experiment."""
    return ExperimentManager(experiment_path=sample_experiment)


# ── Test: Loading ───────────────────────────────────────────────────


def test_load_experiment(manager):
    """Should load experiment name correctly."""
    assert manager.experiment_name == "Test Experiment"


def test_total_steps(manager):
    """Should report correct number of steps."""
    assert manager.total_steps == 3


# ── Test: Step Access ───────────────────────────────────────────────


def test_get_step_by_index(manager):
    """Should retrieve step by index."""
    step = manager.get_step(0)
    assert step is not None
    assert step.id == "step_one"


def test_get_step_by_id(manager):
    """Should retrieve step by ID."""
    step = manager.get_step_by_id("step_two")
    assert step is not None
    assert step.name == "Step Two"


def test_get_invalid_step(manager):
    """Invalid index should return None."""
    step = manager.get_step(99)
    assert step is None


def test_get_invalid_step_id(manager):
    """Invalid ID should return None."""
    step = manager.get_step_by_id("nonexistent")
    assert step is None


def test_get_all_steps(manager):
    """Should return all steps."""
    steps = manager.get_all_steps()
    assert len(steps) == 3
    assert steps[0].id == "step_one"
    assert steps[2].id == "step_three"


# ── Test: Error Handling ────────────────────────────────────────────


def test_missing_file():
    """Should handle missing config file gracefully."""
    manager = ExperimentManager(experiment_path="/nonexistent/path.json")
    assert manager.total_steps == 0


def test_invalid_json(tmp_path):
    """Should handle invalid JSON gracefully."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json {{{")
    manager = ExperimentManager(experiment_path=str(bad_file))
    assert manager.total_steps == 0


# ── Test: Step Properties ───────────────────────────────────────────


def test_step_has_expected_objects(manager):
    """Steps should have expected objects."""
    step = manager.get_step(1)
    assert "tool" in step.expected_objects


def test_step_has_interaction(manager):
    """Step with interaction should have it set."""
    step = manager.get_step(1)
    assert step.expected_interaction == "hand_near_tool"
