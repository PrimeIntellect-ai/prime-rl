import json
import time
from pathlib import Path
from typing import Generator, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
)

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.utils import get_world
from prime_rl.utils.client import NCCL_MANIFEST, NCCL_READY_MARKER, get_nccl_chunk_manifest
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path
from prime_rl.utils.vlm import get_layer_prefix


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int, layer_prefix: str
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield non-layer weights first, then each layer's weights.

    Yields (layer_idx, layer_state_dict) where layer_idx is -1 for the non-layer
    dict and the actual layer index (0, 1, ...) for layer dicts.
    """
    yield -1, {key: value for key, value in state_dict.items() if not key.startswith(layer_prefix)}

    for i in range(num_layers):
        yield (
            i,
            {key: value for key, value in state_dict.items() if key.startswith(f"{layer_prefix}{i}.")},
        )


def preprocess_layer_checkpoint(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        world_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype

        if self.world.is_master:
            disable_nccl_p2p_if_unavailable()
            self.communicator = NCCLWeightTransferEngine.trainer_init(
                {
                    "master_address": host,
                    "master_port": port,
                    "world_size": world_size,
                }
            )
            self.logger.debug("NCCL broadcast initialized on master rank")
        else:
            self.logger.debug("NCCL broadcast initialized on non-master rank (no communicator)")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int, save_dir: Path) -> None:
        """Broadcast checkpoint-format model weights through vLLM's native NCCL engine."""
        self._broadcast_native_weights(model, save_dir)

    def _broadcast_native_weights(self, model: nn.Module, save_dir: Path) -> None:
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)

        if self.world.is_master:
            self._write_manifest(save_dir / NCCL_MANIFEST, {"num_chunks": num_layers + 1})

        for chunk_id, (layer_id, layer_state_dict) in enumerate(
            filter_state_dict_by_layers(state_dict, num_layers, layer_prefix)
        ):
            layer_state_dict = self._resolve_dtensors(layer_state_dict)
            layer_state_dict = preprocess_layer_checkpoint(model, layer_state_dict, layer_id)
            if not self.world.is_master:
                continue

            update_info = {
                "names": list(layer_state_dict),
                "dtype_names": [str(tensor.dtype).removeprefix("torch.") for tensor in layer_state_dict.values()],
                "shapes": [list(tensor.shape) for tensor in layer_state_dict.values()],
            }
            self._write_manifest(get_nccl_chunk_manifest(save_dir, chunk_id), update_info)
            NCCLWeightTransferEngine.trainer_send_weights(
                iter(layer_state_dict.items()),
                NCCLTrainerSendWeightsArgs(group=self.communicator),
            )

    @staticmethod
    def _write_manifest(path: Path, payload: dict) -> None:
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(json.dumps(payload))
        temporary_path.replace(path)

    def _resolve_dtensors(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                state_dict[key] = cast(DTensor, value.to(self.dtype)).full_tensor()
        return state_dict


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.nccl_broadcast_sender = NCCLWeightBroadcastSender(
            config.host,
            config.port,
            config.inference_world_size + 1,
            dtype,
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL and notifies the orchestrator."""
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()
        # Only the master touches the filesystem to notify the orchestrator, but all
        # ranks must wait for the inference pool before entering the broadcast path:
        # the broadcast preparation (DTensor resolution, quantization) enqueues
        # collectives on non-master ranks, and if those ranks start prep before
        # the orchestrator has paused inference, the collectives sit unmatched
        # until NCCL's watchdog kills the process after 10 min.
        save_dir = get_step_path(get_broadcast_dir(self.output_dir), step)
        if self.world.is_master:
            self._notify_orchestrator(save_dir)
            self._wait_for_nccl_ready(save_dir)
        if self.world.world_size > 1:
            dist.barrier()
        self.nccl_broadcast_sender.broadcast_weights(model, step, save_dir)
        self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")

    def _notify_orchestrator(self, save_dir: Path) -> None:
        """Create the STABLE marker the orchestrator's weight watcher polls for."""
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / NCCL_READY_MARKER).unlink(missing_ok=True)
        (save_dir / NCCL_MANIFEST).unlink(missing_ok=True)
        for chunk_manifest in save_dir.glob("NCCL_CHUNK_*.json"):
            chunk_manifest.unlink()
        (save_dir / "STABLE").touch()

    def _wait_for_nccl_ready(self, save_dir: Path):
        """Wait for inference workers to signal they are ready to receive NCCL broadcast."""
        nccl_ready_file = save_dir / NCCL_READY_MARKER
        self.logger.debug(f"Waiting for NCCL_READY marker at {nccl_ready_file}")
        sync_wait_for_path(nccl_ready_file, interval=0.1, log_interval=10)
        self.logger.debug("Inference workers ready for NCCL broadcast")
