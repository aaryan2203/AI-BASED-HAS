"""
Tests for ExperimentSequenceValidator.

Covers: correct sequence, skipped steps, out-of-order, repeated steps,
low confidence, unknown activity, and full experiment completion.
"""

import sys
import time
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sequence_validator import (
    ExperimentSequenceValidator,
    StepStatus,
    ExperimentState,
    ValidationResult,
)

# ── Fixtures ────────────────────────────────────────────────────────

SAMPLE_STEPS = [
    {"id": "approach_rack", "name": "Approach Rack"},
    {"id": "pick_container", "name": "Pick Container"},
    {"id": "open_container", "name": "Open Container"},
    {"id": "remove_sample", "name": "Remove Sample"},
    {"id": "place_sample", "name": "Place Sample"},
    {"id": "press_button", "name": "Press Button"},
    {"id": "close_container", "name": "Close Container"},
    {"id": "return_tool", "name": "Return Tool"},
]


@pytest.fixture
def validator():
    """Create a validator with low temporal confirmation for fast testing."""
    return ExperimentSequenceValidator(
        experiment_steps=SAMPLE_STEPS,
        temporal_confirmation_frames=3,
        confidence_threshold=0.6,
    )


# ── Test: Correct Sequence ──────────────────────────────────────────


def test_correct_single_step(validator):
    """Confirming the first step advances the validator."""
    # Need 3 consecutive confirmations
    for _ in range(2):
        result = validator.validate("approach_rack", 0.9)
        assert result is None or result.status != StepStatus.CORRECT

    result = validator.validate("approach_rack", 0.9)
    assert result is not None
    assert result.status == StepStatus.CORRECT
    assert result.detected_activity == "approach_rack"
    assert validator.current_step_index == 1


def test_full_correct_sequence(validator):
    """All 8 steps completed in order should result in COMPLETED state."""
    for step in SAMPLE_STEPS:
        for _ in range(3):
            validator.validate(step["id"], 0.9)

    assert validator.is_complete()
    assert validator.experiment_state == ExperimentState.COMPLETED


# ── Test: Skipped Step ──────────────────────────────────────────────


def test_skipped_step(validator):
    """Skipping step 2 (pick_container) and doing step 3 should be SKIPPED."""
    # Complete step 1
    for _ in range(3):
        validator.validate("approach_rack", 0.9)

    # Now expected is pick_container, but we send open_container
    result = validator.validate("open_container", 0.9)
    assert result is not None
    assert result.status == StepStatus.SKIPPED
    assert result.expected_activity == "pick_container"
    assert result.detected_activity == "open_container"


# ── Test: Out-of-Sequence ───────────────────────────────────────────


def test_out_of_sequence(validator):
    """Doing a much later step should be OUT_OF_SEQUENCE."""
    # Complete step 1
    for _ in range(3):
        validator.validate("approach_rack", 0.9)

    # Expected is pick_container, but we send press_button (step 6)
    result = validator.validate("press_button", 0.9)
    assert result is not None
    assert result.status in (StepStatus.SKIPPED, StepStatus.OUT_OF_SEQUENCE)


# ── Test: Repeated Step ─────────────────────────────────────────────


def test_repeated_step(validator):
    """Repeating a completed step should be REPEATED."""
    # Complete step 1
    for _ in range(3):
        validator.validate("approach_rack", 0.9)

    # Complete step 2
    for _ in range(3):
        validator.validate("pick_container", 0.9)

    # Repeat step 1
    result = validator.validate("approach_rack", 0.9)
    assert result is not None
    assert result.status == StepStatus.REPEATED


# ── Test: Low Confidence ────────────────────────────────────────────


def test_low_confidence(validator):
    """Low confidence predictions should return LOW_CONFIDENCE status."""
    result = validator.validate("approach_rack", 0.3)
    assert result is not None
    assert result.status == StepStatus.LOW_CONFIDENCE
    # Should NOT advance step
    assert validator.current_step_index == 0


# ── Test: Unknown Activity ──────────────────────────────────────────


def test_unknown_activity(validator):
    """An activity not in the experiment should be UNKNOWN."""
    result = validator.validate("dance_in_space", 0.9)
    assert result is not None
    assert result.status == StepStatus.UNKNOWN


# ── Test: Reset ─────────────────────────────────────────────────────


def test_reset(validator):
    """Reset should return validator to initial state."""
    # Complete a few steps
    for _ in range(3):
        validator.validate("approach_rack", 0.9)

    validator.reset()
    assert validator.current_step_index == 0
    assert validator.experiment_state == ExperimentState.IDLE
    assert len(validator.completed_steps) == 0


# ── Test: Progress ──────────────────────────────────────────────────


def test_progress(validator):
    """Progress should reflect completed steps."""
    # Complete 2 steps
    for step in SAMPLE_STEPS[:2]:
        for _ in range(3):
            validator.validate(step["id"], 0.9)

    progress = validator.get_progress()
    assert progress["completed"] == 2
    assert progress["total"] == 8
    assert progress["percentage"] == 25.0


# ── Test: Temporal Confirmation ─────────────────────────────────────


def test_temporal_confirmation_resets_on_change(validator):
    """Changing activity mid-confirmation should reset the counter."""
    # Start confirming approach_rack
    validator.validate("approach_rack", 0.9)
    validator.validate("approach_rack", 0.9)

    # Switch to different activity before confirmation completes
    validator.validate("pick_container", 0.9)

    # Now go back to approach_rack - should need full 3 confirmations again
    validator.validate("approach_rack", 0.9)
    validator.validate("approach_rack", 0.9)
    result = validator.validate("approach_rack", 0.9)
    assert result is not None
    assert result.status == StepStatus.CORRECT
