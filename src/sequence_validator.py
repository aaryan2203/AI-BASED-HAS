"""
BAS-HAR: Experiment Sequence Validator

State-machine-based validator that checks whether detected activities
follow the expected experiment sequence. Handles correct, repeated,
skipped, out-of-sequence, unknown, and low-confidence scenarios.

Uses temporal confirmation to avoid false positives from single-frame
noise — requires N consecutive frames with the same prediction before
advancing the experiment state.
"""

import logging
import time
from enum import Enum, auto
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a detected activity relative to the expected sequence."""
    CORRECT = auto()
    REPEATED = auto()
    SKIPPED = auto()
    OUT_OF_SEQUENCE = auto()
    UNKNOWN = auto()
    LOW_CONFIDENCE = auto()


class ExperimentState(Enum):
    """Overall state of the experiment."""
    IDLE = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    ERROR = auto()
    PAUSED = auto()


@dataclass
class ValidationResult:
    """Result of validating a detected activity against the expected sequence."""
    detected_activity: str
    detected_display_name: str
    expected_activity: str
    expected_display_name: str
    status: StepStatus
    current_step_index: int
    confidence: float
    timestamp: float
    message: str


class ExperimentSequenceValidator:
    """Validates that detected activities follow the expected experiment sequence.

    Uses temporal confirmation: a step is only marked as CORRECT after
    `temporal_confirmation_frames` consecutive frames predict the same activity.

    Args:
        experiment_steps: List of step dicts, each with at least 'id' and 'name' keys.
        temporal_confirmation_frames: Consecutive frames needed to confirm a step.
        confidence_threshold: Minimum confidence to consider a prediction valid.
    """

    def __init__(
        self,
        experiment_steps: list[dict[str, str]],
        temporal_confirmation_frames: int = 10,
        confidence_threshold: float = 0.65,
    ):
        self.experiment_steps = experiment_steps
        self.temporal_confirmation_frames = temporal_confirmation_frames
        self.confidence_threshold = confidence_threshold

        # Build lookup: step_id -> index
        self._step_id_to_index: dict[str, int] = {}
        for i, step in enumerate(experiment_steps):
            step_id = step.get("id", "")
            if step_id:
                self._step_id_to_index[step_id] = i

        self._known_activities: set[str] = set(self._step_id_to_index.keys())

        # Mutable state
        self.current_step_index: int = 0
        self.completed_steps: list[int] = []
        self.experiment_state: ExperimentState = ExperimentState.IDLE
        self.confirmation_counter: int = 0
        self.pending_activity: str | None = None

        logger.info(
            "Validator initialized: %d steps, confirmation=%d frames, threshold=%.2f",
            len(experiment_steps),
            temporal_confirmation_frames,
            confidence_threshold,
        )

    def validate(self, activity: str, confidence: float) -> ValidationResult | None:
        """Validate a detected activity against the expected sequence.

        Returns:
            ValidationResult if a meaningful event occurred (step completed,
            error detected, etc.), or None if waiting for temporal confirmation.
        """
        if not self.experiment_steps:
            return None

        # Transition from IDLE on first call
        if self.experiment_state == ExperimentState.IDLE:
            self.experiment_state = ExperimentState.IN_PROGRESS

        if self.is_complete():
            return None

        expected_step = self.get_current_step()
        expected_id = expected_step.get("id", "") if expected_step else ""
        expected_name = expected_step.get("name", "") if expected_step else ""

        now = time.time()

        # --- Low confidence ---
        if confidence < self.confidence_threshold:
            return ValidationResult(
                detected_activity=activity,
                detected_display_name=activity,
                expected_activity=expected_id,
                expected_display_name=expected_name,
                status=StepStatus.LOW_CONFIDENCE,
                current_step_index=self.current_step_index,
                confidence=confidence,
                timestamp=now,
                message=f"Low confidence prediction: {confidence:.2f}",
            )

        # --- Temporal confirmation tracking ---
        if activity != self.pending_activity:
            self.pending_activity = activity
            self.confirmation_counter = 1
        else:
            self.confirmation_counter += 1

        # --- Unknown activity ---
        if activity not in self._known_activities:
            return ValidationResult(
                detected_activity=activity,
                detected_display_name=activity,
                expected_activity=expected_id,
                expected_display_name=expected_name,
                status=StepStatus.UNKNOWN,
                current_step_index=self.current_step_index,
                confidence=confidence,
                timestamp=now,
                message=f"Unknown activity: {activity}",
            )

        # --- Correct step ---
        if activity == expected_id:
            if self.confirmation_counter >= self.temporal_confirmation_frames:
                self.completed_steps.append(self.current_step_index)
                self.current_step_index += 1
                self.confirmation_counter = 0
                self.pending_activity = None

                if self.is_complete():
                    self.experiment_state = ExperimentState.COMPLETED

                logger.info("Step CORRECT: %s (index %d)", activity, self.current_step_index - 1)
                return ValidationResult(
                    detected_activity=activity,
                    detected_display_name=activity,
                    expected_activity=expected_id,
                    expected_display_name=expected_name,
                    status=StepStatus.CORRECT,
                    current_step_index=self.current_step_index,
                    confidence=confidence,
                    timestamp=now,
                    message=f"Step completed: {expected_name}",
                )
            # Still accumulating confirmation frames
            return None

        # --- Activity is known but not the expected step ---
        activity_index = self._step_id_to_index.get(activity, -1)

        # Repeated (already completed) step
        if activity_index in self.completed_steps:
            return ValidationResult(
                detected_activity=activity,
                detected_display_name=activity,
                expected_activity=expected_id,
                expected_display_name=expected_name,
                status=StepStatus.REPEATED,
                current_step_index=self.current_step_index,
                confidence=confidence,
                timestamp=now,
                message=f"Repeated completed step: {activity}",
            )

        # Skipped step (activity is a later step in the sequence)
        if activity_index > self.current_step_index:
            return ValidationResult(
                detected_activity=activity,
                detected_display_name=activity,
                expected_activity=expected_id,
                expected_display_name=expected_name,
                status=StepStatus.SKIPPED,
                current_step_index=self.current_step_index,
                confidence=confidence,
                timestamp=now,
                message=f"Step skipped! Expected: {expected_name}, Detected: {activity}",
            )

        # Out of sequence (earlier uncompleted step or other anomaly)
        return ValidationResult(
            detected_activity=activity,
            detected_display_name=activity,
            expected_activity=expected_id,
            expected_display_name=expected_name,
            status=StepStatus.OUT_OF_SEQUENCE,
            current_step_index=self.current_step_index,
            confidence=confidence,
            timestamp=now,
            message=f"Out of sequence: {activity} (expected {expected_name})",
        )

    def get_current_step(self) -> dict[str, str] | None:
        """Get the current expected step, or None if experiment is complete."""
        if self.is_complete():
            return None
        return self.experiment_steps[self.current_step_index]

    def get_next_step(self) -> dict[str, str] | None:
        """Get the step after the current one, or None if at the end."""
        next_idx = self.current_step_index + 1
        if next_idx < len(self.experiment_steps):
            return self.experiment_steps[next_idx]
        return None

    def get_completed_steps(self) -> list[dict[str, str]]:
        """Get list of all completed step dicts."""
        return [self.experiment_steps[i] for i in self.completed_steps]

    def get_progress(self) -> dict:
        """Get experiment progress summary."""
        total = len(self.experiment_steps)
        completed = len(self.completed_steps)
        pct = (completed / total * 100) if total > 0 else 0.0
        return {
            "completed": completed,
            "total": total,
            "percentage": pct,
            "state": self.experiment_state.name,
        }

    def reset(self) -> None:
        """Reset the validator to initial state for a new experiment run."""
        self.current_step_index = 0
        self.completed_steps = []
        self.experiment_state = ExperimentState.IDLE
        self.confirmation_counter = 0
        self.pending_activity = None
        logger.info("Experiment validator reset")

    def is_complete(self) -> bool:
        """Check if all experiment steps have been completed."""
        return self.current_step_index >= len(self.experiment_steps)
