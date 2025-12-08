import pickle
import time
from pathlib import Path
from typing import Generator, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.config import NCCLWeightBroadcastConfig
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.logger import get_logger


def broadcast_integer(integer: int, communicator: PyNcclCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(state_dict: dict[str, Tensor], communicator: PyNcclCommunicator) -> None:
    """Broadcast a state dict to NCCL process group using the PyNcclCommunicator."""
    # Group tensors by dtype
    dtype_groups: dict[torch.dtype, list[tuple[str, Tensor]]] = {}
    for key, value in state_dict.items():
        assert not isinstance(value, DTensor), (
            "DTensor is not supported for broadcast, should have been converted to tensor already"
        )
        dtype = value.dtype
        if dtype not in dtype_groups:
            dtype_groups[dtype] = []
        dtype_groups[dtype].append((key, value))

    # Build metadata: for each dtype group, store keys and shapes
    metadata = {}
    for dtype, items in dtype_groups.items():
        metadata[dtype] = [(key, value.shape, value.numel()) for key, value in items]

    # Send metadata
    state = pickle.dumps(metadata)
    size_tensor = torch.tensor([len(state)], dtype=torch.long).cuda()
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.ByteTensor(list(state)).cuda()
    communicator.broadcast(state_tensor, src=0)

    # Concatenate and broadcast tensors grouped by dtype
    for dtype, items in dtype_groups.items():
        # Flatten all tensors and concatenate
        flat_tensors = [value.flatten() for _, value in items]
        concatenated = torch.cat(flat_tensors)
        communicator.broadcast(concatenated, src=0)
        del concatenated
        # Clean up individual tensors
        for _, value in items:
            del value


def filter_state_dict_by_layers(
    state_dict: dict[str, torch.Tensor], num_layers: int
) -> Generator[tuple[int, dict[str, torch.Tensor]], None, None]:
    """Yield a generator of state dicts for each layer as well as the remaining weights."""
    yield 0, {key: value for key, value in state_dict.items() if "model.layers" not in key}

    for i in range(1, num_layers + 1):  # +1 because layer indices start from 1
        yield (
            i,
            {
                key: value
                for key, value in state_dict.items()
                if key.startswith(f"model.layers.{i}.") or key == f"model.layers.{i}"
            },
        )


class NCCLWeightBroadcastSender:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = dtype

        if self.world.is_master:
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("NCCL broadcast initialized on master rank")
        else:
            self.logger.debug("NCCL broadcast initialized on non-master rank (no communicator)")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        state_dict = model.state_dict()
        num_layers = get_max_layer_num(state_dict)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")

        for layer_id, state_dict in filter_state_dict_by_layers(state_dict, num_layers):
            self.logger.debug(f"Sending layer {layer_id + 1}/{num_state_dict_to_send} state dict")
            for key, value in list(state_dict.items()):
                if isinstance(value, DTensor):
                    value = cast(DTensor, value.to(self.dtype)).full_tensor()
                state_dict[key] = value

            # Convert PrimeRL format to HF format for this layer if needed
            if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(state_dict):
                model.convert_layer_to_hf(state_dict, layer_id)

            if self.world.is_master:
                broadcast_state_dict(state_dict, self.communicator)


class NCCLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(output_dir)
        self.logger = get_logger()
        self.world = get_world()
        self.nccl_broadcast_sender = NCCLWeightBroadcastSender(
            config.host, config.port, 0, config.inference_world_size + 1, device, config.timeout
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int, adapter_only: bool = False) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL and notifies the orchestrator."""
        if adapter_only:
            raise NotImplementedError("NCCL weight broadcast does not support adapter only yet")
        self.logger.debug("Starting broadcasting weights to inference engine via NCCL")
        start_time = time.perf_counter()
        if self.world.is_master:
            self.notify_orchestrator(step)
        self.nccl_broadcast_sender.broadcast_weights(model, step)
        self.logger.debug(f"Weights broadcasted in {time.perf_counter() - start_time:.2f}s")
