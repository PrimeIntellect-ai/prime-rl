import shutil
import time
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.trainer.config import LoRAConfig
from prime_rl.trainer.lora import save_lora_config
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import FileSystemWeightBroadcastConfig
from prime_rl.trainer.runs import get_multi_run_manager
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
        self.multi_run_manager = get_multi_run_manager()
        self._logged_full_weights_with_lora = False
        self.logger.debug(
            f"Filesystem broadcast initialized (save_format={config.save_format}, save_sharded={self.save_sharded})"
        )

    def _has_trainable_non_lora_params(self, model: nn.Module) -> bool:
        return any(
            param.requires_grad and "lora_A" not in name and "lora_B" not in name
            for name, param in model.named_parameters()
        )

    def _get_adapter_state_dict_for_run(self, idx: int) -> dict[str, torch.Tensor]:
        state_dict = self.multi_run_manager.get_state_dict_for_run(idx)
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                value = value.full_tensor()
            if self.world.is_master:
                state_dict[key] = value.to("cpu", non_blocking=False)
        return state_dict

    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast weights by saving a HF-compatible checkpoint to shared filesystem and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via shared filesystem")
        start_time = time.perf_counter()
        has_trainable_non_lora_params = self._has_trainable_non_lora_params(model)
        adapter_only = self.lora_config is not None and not has_trainable_non_lora_params

        if not adapter_only:
            full_state_dict = gather_weights_on_master(model, is_master=self.world.is_master)
            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(full_state_dict):
                model.convert_to_hf(full_state_dict)
        else:
            full_state_dict = None

        if self.lora_config is not None and not adapter_only and not self._logged_full_weights_with_lora:
            self.logger.info(
                "Broadcasting full base weights plus LoRA adapters because non-LoRA trainable parameters are enabled."
            )
            self._logged_full_weights_with_lora = True

        for idx in self.multi_run_manager.ready_to_update_idxs:
            self.logger.debug(
                f"Broadcasting weights for run {idx} (ready_to_update={self.multi_run_manager.ready_to_update[idx]})"
            )

            adapter_state_dict = None
            if self.lora_config is not None:
                # All ranks must participate in DTensor gathering, but only master saves.
                adapter_state_dict = self._get_adapter_state_dict_for_run(idx)

            # TODO: Broadcast ready to update in sync, then we dont need to gather on not ready
            if self.world.is_master:
                try:
                    save_dir = get_step_path(
                        get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                        self.multi_run_manager.progress[idx].step,
                    )
                    save_dir.mkdir(parents=True, exist_ok=True)

                    self.logger.debug(f"Saving weights for run {idx} to {save_dir}")
                    if full_state_dict is not None:
                        save_state_dict(dict(full_state_dict), save_dir, self.save_format, self.save_sharded, adapter=False)

                    if adapter_state_dict is not None:
                        save_state_dict(adapter_state_dict, save_dir, self.save_format, save_sharded=False, adapter=True)
                        save_lora_config(self.lora_config, model, save_dir)

                    self._notify_orchestrator(save_dir)

                    # If the run is deleted, remove the run directory
                    # This is avoid the creation of zombie runs when the directory is deleted while we are broadcasting which recreates the directory
                    if self.multi_run_manager.get_orchestrator_config(self.multi_run_manager.idx_2_id[idx]) is None:
                        shutil.rmtree(self.multi_run_manager.get_run_dir(idx))

                except FileNotFoundError:
                    self.logger.warning(f"Run {idx} is deleted, skipping")
                except Exception as e:
                    self.logger.error(f"Error broadcasting weights for run {idx}: {e}")
                finally:
                    self.multi_run_manager.ready_to_update[idx] = False

        if self.world.is_master:
            self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self, save_dir: Path):
        """Notify the orchestrator that the weights have been broadcast by writing a 'STABLE' file to a shared filesystem."""
        stable_file = save_dir / "STABLE"
        stable_file.touch()

    def maybe_clean(self, max_async_level: int, interval_to_keep: int | None):
        for idx in self.multi_run_manager.used_idxs:
            maybe_clean(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
                max_async_level,
                interval_to_keep,
            )
