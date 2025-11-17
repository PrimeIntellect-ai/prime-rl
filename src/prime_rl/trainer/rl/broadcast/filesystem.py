from pathlib import Path
from typing import Literal

import torch.nn as nn

from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.weights import gather_weights, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_step_path


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(self, output_dir: Path, save_format: Literal["safetensors", "torch"], save_sharded: bool):
        super().__init__(output_dir)
        self.save_format: Literal["safetensors", "torch"] = save_format
        self.save_sharded = save_sharded
        self.world = get_world()
        self.logger.info(
            f"Initializing filesystem broadcast sender ({save_format=}, {save_sharded=}, broadcast_dir={self.broadcast_dir})"
        )

    def broadcast_weights(self, model: nn.Module, step: int):
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem."""
        state_dict = gather_weights(model, self.world.is_master)
        save_dir = get_step_path(self.broadcast_dir, step)
        save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded)
        self.notify_orchestrator(step)
