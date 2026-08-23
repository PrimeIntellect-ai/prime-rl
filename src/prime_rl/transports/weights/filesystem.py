from pathlib import Path

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import FileSystemWeightBroadcastConfig, LoRAConfig
from prime_rl.trainer.lora import get_lora_state, save_lora_config
from prime_rl.transports.weights.base import WeightBroadcast
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict,
    save_state_dict_parallel,
)


class FileSystemWeightBroadcast(WeightBroadcast):
    """Broadcast weights by saving a HF-compatible checkpoint (or, for LoRA
    runs, the PEFT-shaped adapter) to a shared filesystem."""

    def __init__(
        self,
        output_dir: Path,
        config: FileSystemWeightBroadcastConfig,
        lora_config: LoRAConfig | None = None,
    ):
        super().__init__(output_dir, config.timeout)
        self.lora_config = lora_config
        self.logger.debug("Initialized filesystem weight broadcast")

    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        if self.lora_config is not None:
            # All ranks must participate in DTensor gathering, but only master saves
            state_dict = get_lora_state().adapter_state_dict()
            for key, value in state_dict.items():
                if isinstance(value, DTensor):
                    value = value.full_tensor()
                if self.world.is_master:
                    state_dict[key] = value.to("cpu", non_blocking=False)
            if self.world.is_master:
                self.logger.debug(f"Saving adapter to {step_dir}")
                save_state_dict(state_dict, step_dir, save_sharded=False, adapter=True)
                save_lora_config(
                    model,
                    step_dir,
                    rank=self.lora_config.rank,
                    alpha=self.lora_config.alpha,
                    dropout=self.lora_config.dropout,
                )
        else:
            dist.barrier()
            state_dict = gather_weights_parallel(model)
            state_dict = convert_state_dict_to_hf(model, state_dict)
            self.logger.debug(f"Saving weights to {step_dir}")
            save_state_dict_parallel(state_dict, step_dir)
