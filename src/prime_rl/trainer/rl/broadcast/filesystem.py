import time
from pathlib import Path
from typing import Literal

import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig
from prime_rl.trainer.runs import get_runs
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    save_state_dict,
)
from prime_rl.trainer.world import get_world
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
        self.runs = get_runs()
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={self.save_sharded})"
        )

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        adapter_only = self.lora_config is not None

        if not adapter_only:
            state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_to_hf(state_dict)

        for idx in self.runs.used_idxs:
            if not self.runs.ready_to_update[idx]:
                continue
            self.logger.debug(f"Broadcasting weights for run {idx} (ready_to_update={self.runs.ready_to_update[idx]})")

            if adapter_only:
                # For adapter-only, Runs creates state dict directly for each run
                # All ranks must participate in DTensor gathering, but only master saves
                state_dict = self.runs.get_state_dict_for_run(idx)
                for key, value in state_dict.items():
                    if isinstance(value, DTensor):
                        value = value.full_tensor()
                    if self.world.is_master:
                        state_dict[key] = value.to("cpu", non_blocking=False)

            # TODO: Broadcast ready to update in sync, then we dont need to gather on not ready
            if self.world.is_master:
                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.runs.get_run_dir(idx)), self.runs.progress[idx].step
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
                    if adapter_only:
                        save_lora_config(self.lora_config, model, save_dir)

                    self._notify_orchestrator(save_dir)
                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.runs.ready_to_update[idx] = False

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, max_async_level: int, interval_to_keep: int | None):
        for idx in self.runs.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.runs.get_run_dir(idx)),
                self.runs.progress[idx].step,
                max_async_level,
                interval_to_keep,
            )
