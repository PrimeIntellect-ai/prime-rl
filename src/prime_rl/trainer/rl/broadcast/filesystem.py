import time
from pathlib import Path
from typing import Literal

import torch.nn as nn

from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig
from prime_rl.trainer.weights import gather_weights, save_state_dict
from prime_rl.trainer.world import get_world
from prime_rl.utils.utils import get_step_path


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(self, output_dir: Path, config: FileSystemWeightBroadcastConfig):
        super().__init__(output_dir)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded
        self.world = get_world()
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={config.save_sharded}, broadcast_dir={self.broadcast_dir})"
        )

    def broadcast_weights(self, model: nn.Module, step: int):
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.time()
        state_dict = gather_weights(model, self.world.is_master)
        save_dir = get_step_path(self.broadcast_dir, step)
        save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded)
        # Notify the orchestrator at the end of step to signal that it is safe to load weights from shared filesystem
        self.notify_orchestrator(step)
        self.logger.debug(f"Weights broadcasted in {time.time() - start_time:.2f}s")
