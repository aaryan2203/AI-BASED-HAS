import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any

from src.config import Config, PROJECT_ROOT

logger = logging.getLogger(__name__)

@dataclass
class ExperimentStep:
    id: str
    name: str
    display_name: str
    description: str
    expected_objects: list[str]
    expected_interaction: str
    detection_hints: dict[str, Any]
    index: int

class ExperimentManager:
    def __init__(self, config: Config | None = None, experiment_path: str | None = None):
        self.config = config
        
        if experiment_path:
            self.experiment_file = Path(experiment_path)
        elif config:
            self.experiment_file = PROJECT_ROOT / config.experiment_config_path
        else:
            self.experiment_file = PROJECT_ROOT / "experiments" / "bas_sample_handling.json"
            
        self._experiment_name = "Unknown"
        self._steps: list[ExperimentStep] = []
        
        self.load_experiment(self.experiment_file)
        
    def load_experiment(self, filepath: Path | str):
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Experiment file not found: {filepath}")
            return
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._validate_and_parse(data)
            logger.info(f"Successfully loaded experiment '{self.experiment_name}' with {self.total_steps} steps.")
        except json.JSONDecodeError as e:
            logger.error(f"Malformed JSON in experiment file: {e}")
        except Exception as e:
            logger.exception(f"Error loading experiment file: {e}")
            
    def _validate_and_parse(self, data: dict):
        self._experiment_name = data.get('experiment_name', 'Unknown')
        steps_data = data.get('steps', [])
        
        self._steps = []
        for idx, step_dict in enumerate(steps_data):
            try:
                step = ExperimentStep(
                    id=step_dict.get('id', f'step_{idx}'),
                    name=step_dict.get('name', f'Step {idx}'),
                    display_name=step_dict.get('display_name', f'Step {idx}'),
                    description=step_dict.get('description', ''),
                    expected_objects=step_dict.get('expected_objects', []),
                    expected_interaction=step_dict.get('expected_interaction', ''),
                    detection_hints=step_dict.get('detection_hints', {}),
                    index=idx
                )
                self._steps.append(step)
            except Exception as e:
                logger.warning(f"Failed to parse step {idx}: {e}")

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def total_steps(self) -> int:
        return len(self._steps)
        
    def get_step(self, index: int) -> ExperimentStep | None:
        if 0 <= index < len(self._steps):
            return self._steps[index]
        return None
        
    def get_step_by_id(self, step_id: str) -> ExperimentStep | None:
        for step in self._steps:
            if step.id == step_id:
                return step
        return None
        
    def get_all_steps(self) -> list[ExperimentStep]:
        return self._steps
