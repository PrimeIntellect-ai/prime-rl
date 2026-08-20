import time
from pathlib import Path
from typing import Literal

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.lora import get_lora_state, save_lora_config
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    convert_state_dict_to_hf,
    gather_weights_on_master,
    gather_weights_parallel,
    save_state_dict,
    save_state_dict_parallel,
)
from prime_rl.trainer.world import get_world
from prime_rl.transports.weights.base import WeightBroadcast
from prime_rl.utils.utils import get_broadcast_dir, get_step_path


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via shared filesystem."""

    def __init__(
        self, output_dir: Path, config: FileSystemWeightBroadcastConfig, lora_config: LoRAConfig | None = None
    ):
        super().__init__(output_dir, lora_config)
        self.save_format: Literal["safetensors", "torch"] = config.save_format
        self.save_sharded = config.save_sharded if lora_config is None else False
        self.world = get_world()
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={self.save_sharded})"
        )

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        adapter_only = self.lora_config is not None
        save_parallel = not adapter_only and self.save_sharded and self.save_format == "safetensors"

        save_dir = get_step_path(get_broadcast_dir(self.output_dir), step)
        if self.world.is_master:
            save_dir.mkdir(parents=True, exist_ok=True)

        if save_parallel:
            dist.barrier()
            state_dict = gather_weights_parallel(model)
            state_dict = convert_state_dict_to_hf(model, state_dict)
            self.logger.debug(f"Saving weights to {save_dir}")
            save_state_dict_parallel(state_dict, save_dir)
            if self.world.is_master:
                self._notify_orchestrator(save_dir)
                self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")
            return

        if adapter_only:
            # All ranks must participate in DTensor gathering, but only master saves
            state_dict = get_lora_state().adapter_state_dict()
            for key, value in state_dict.items():
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                if self.world.is_master:
                    state_dict[key] = value.to("cpu", non_blocking=False)
        else:
            state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
            state_dict = convert_state_dict_to_hf(model, state_dict)

        if self.world.is_master:
            self.logger.debug(f"Saving weights to {save_dir}")
            save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
            if adapter_only:
                save_lora_config(
                    model,
                    save_dir,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
                )

            self._notify_orchestrator(save_dir)
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, step: int, interval_to_keep: int | None):
        maybe_clean(get_broadcast_dir(self.output_dir), step, interval_to_keep)
