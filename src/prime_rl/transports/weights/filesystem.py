from pathlib import Path

import torch.distributed as dist
import torch.nn as nn

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.pathing import wait_for_path
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict_parallel,
)


class FileSystemWeightSender(WeightSender):
    """Broadcast weights by saving a HF-compatible checkpoint to a shared filesystem."""

    def __init__(self, output_dir: Path, config: FileSystemWeightBroadcastConfig):
        super().__init__(output_dir, config.timeout)
        self.logger.debug("Initialized filesystem weight broadcast")

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        dist.barrier()
        state_dict = gather_weights_parallel(model)
        state_dict = convert_state_dict_to_hf(model, state_dict)
        self.logger.debug(f"Saving weights to {step_dir}")
        save_state_dict_parallel(state_dict, step_dir)


class FileSystemWeightReceiver(WeightReceiver):
    """Loads broadcasts from the shared filesystem. The acknowledgement lets
    the trainer start writing; the engines are only touched once the weights
    are fully on disk. The full checkpoint load pauses the engines."""

    async def receive(self, step: int) -> None:
        weights_dir = self.step_dir(step)
        self._ack(step)
        await wait_for_path(weights_dir / FINISHED_MARKER)
        await self.admin_plane.update_weights(weights_dir, transport="filesystem", step=step)
