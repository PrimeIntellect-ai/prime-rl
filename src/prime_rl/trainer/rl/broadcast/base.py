from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn

from prime_rl.utils.logger import get_logger
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


class WeightBroadcast(ABC):
    def __init__(self, output_dir: Path):
        self.logger = get_logger()
        self.output_dir = output_dir
        self.broadcast_dir = get_broadcast_dir(output_dir)

    @abstractmethod
    def broadcast_weights(self, model: nn.Module, step: int):
        pass

    def notify_orchestrator(self, step: int):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        step_path = get_step_path(self.broadcast_dir, step)
        step_path.mkdir(parents=True, exist_ok=True)
        stable_file = step_path / "STABLE"
        stable_file.touch()
