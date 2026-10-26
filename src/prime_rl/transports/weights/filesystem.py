import asyncio
import time
from pathlib import Path

import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.shared import FileSystemWeightBroadcastConfig
from prime_rl.configs.trainer import LoRAConfig
from prime_rl.orchestrator.clients import load_lora_adapter
from prime_rl.trainer.lora import get_lora_state, save_lora_config
from prime_rl.transports.weights.base import FINISHED_MARKER, WeightReceiver, WeightSender
from prime_rl.utils.pathing import get_step_path
from prime_rl.utils.sparse_weights import DELTA_DIR, MANIFEST, revalidate
from prime_rl.utils.weight_store import storage_path
from prime_rl.utils.weights import (
    convert_state_dict_to_hf,
    gather_weights_parallel,
    save_state_dict,
    save_state_dict_parallel,
)


class FileSystemWeightSender(WeightSender):
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
            state_dict = gather_weights_parallel(model)
            state_dict = convert_state_dict_to_hf(model, state_dict)
            self.logger.debug(f"Saving weights to {step_dir}")
            save_state_dict_parallel(state_dict, step_dir)


class FileSystemWeightReceiver(WeightReceiver):
    """Loads broadcasts from the shared filesystem. The acknowledgement lets
    the trainer start writing; the engines are only touched once the weights
    are fully on disk. An adapter broadcast (PEFT dir) is hot-swapped under
    live traffic — an in-place adapter reload is a vLLM-native op that needs
    no engine pause; a full checkpoint pauses the engines for the load."""

    async def receive(self, step: int) -> None:
        weights_dir = self.step_dir(step)
        data_dir = get_step_path(storage_path(self.config.storage), step) if self.config.storage else weights_dir
        manifest, finished = data_dir / DELTA_DIR / MANIFEST, weights_dir / FINISHED_MARKER
        self._ack(step)
        # A sparse delta's manifest comes first and its part files as the trainer encodes: the engines
        # read and decode them while they keep serving, and the pause only applies the delta.
        while not (await asyncio.to_thread(manifest.exists) or finished.exists()):
            await asyncio.sleep(0.05)
            revalidate(manifest.parent)
            revalidate(weights_dir)
        if await asyncio.to_thread(manifest.exists):
            start = time.perf_counter()
            await self.admin_plane.prefetch_weights(data_dir)
            self.logger.debug(f"Prefetched sparse weight delta v{step} in {time.perf_counter() - start:.2f}s")
        while not finished.exists():
            await asyncio.sleep(0.05)
            revalidate(weights_dir)
        if (weights_dir / "adapter_config.json").exists():
            await load_lora_adapter(self.admin_plane, self.model_name, weights_dir, step)
        else:
            await self.admin_plane.update_weights(data_dir, transport="filesystem", step=step)
