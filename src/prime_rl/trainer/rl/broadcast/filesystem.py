import time
from pathlib import Path
from typing import Literal

import torch.nn as nn

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig
from prime_rl.trainer.runs import get_runs
from prime_rl.trainer.utils import maybe_clean
from prime_rl.trainer.weights import (
    gather_weights_on_master,
    get_adapter_state_dict,
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
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={config.save_sharded}"
        )

    def broadcast_weights(self, model: nn.Module, step: int, adapter_only: bool = False):
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        if adapter_only:
            state_dict = get_adapter_state_dict(model, is_master=self.world.is_master)
        else:
            state_dict = gather_weights_on_master(model, is_master=self.world.is_master)

        if self.world.is_master:
            # Convert PrimeRL format to HF format if needed
            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_to_hf(state_dict)

            for idx in self.runs.used_idxs:
                if not self.runs.ready_to_update[idx]:
                    continue

                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.runs.get_run_dir(idx)), self.runs.progress[idx].step
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    # Save weights to shared filesystem
                    save_state_dict(state_dict, save_dir, self.save_format, self.save_sharded, adapter=adapter_only)
                    if adapter_only:
                        save_lora_config(self.lora_config, model, save_dir)

                    # Notify the orchestrator at the end of step to signal that it is safe to load weights from shared filesystem
                    self.notify_orchestrator(save_dir)
                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.runs.ready_to_update[idx] = False
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def notify_orchestrator(self, save_dir: Path):
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
