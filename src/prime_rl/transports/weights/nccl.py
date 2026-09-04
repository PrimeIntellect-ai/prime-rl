import pickle
from pathlib import Path
from typing import Callable, Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NCCLWeightBroadcastConfig
from prime_rl.orchestrator.clients import init_nccl_broadcast, update_weights
from prime_rl.trainer.conversion_utils import get_max_layer_num
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.utils import get_world
from prime_rl.transports.weights.base import WeightReceiver, WeightSender
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.utils.weights import resolve_wire_dtype


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


def resolve_dtensors(
    state_dict: dict[str, Tensor],
    keep_in_fp32: Callable[[str], bool] | None,
    default_dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Replace every sharded tensor with its full tensor, at the dtype it goes on the wire in.

    Only DTensors are touched, since only they need gathering. A buffer is never sharded, so it
    goes on the wire in whatever dtype it already holds and its fp32 declaration, if it has one,
    is never consulted. NIXL instead applies the declaration to every floating-point tensor, so
    the transports still disagree about buffer dtypes.
    """
    for key, value in list(state_dict.items()):
        if isinstance(value, DTensor):
            # only gather after the downcast as it will be faster
            target_dtype = resolve_wire_dtype(keep_in_fp32, key, default_dtype)
            state_dict[key] = value.to(target_dtype).full_tensor()
    return state_dict


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


def preprocess_layer_quantized(
    model: nn.Module,
    layer_state_dict: dict[str, Tensor],
    layer_idx: int,
) -> dict[str, Tensor]:
    if layer_idx < 0:
        return layer_state_dict
    return model.convert_layer_to_vllm_kernel(layer_state_dict, layer_idx, quantize_fp8=True)


class NCCLBroadcaster:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = torch.bfloat16
        self.quantize_in_weight_transfer = quantize_in_weight_transfer

        if self.world.is_master:
            disable_nccl_p2p_if_unavailable()
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = PyNcclCommunicator(pg, device=device)
            self.logger.debug("Initialized NCCL broadcast on master rank")
        else:
            self.logger.debug("Initialized NCCL broadcast on non-master rank (no communicator)")

    @torch.no_grad()
    def send(self, model: nn.Module) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        num_state_dict_to_send = num_layers + 1  # we send all layer plus the remaining weights

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")
        preprocess_fn: Callable[[nn.Module, dict[str, Tensor], int], dict[str, Tensor]]
        if self.quantize_in_weight_transfer:
            preprocess_fn = preprocess_layer_quantized
        else:
            preprocess_fn = preprocess_layer_checkpoint

        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        for layer_id, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            layer_state_dict = resolve_dtensors(layer_state_dict, keep_in_fp32, self.dtype)
            layer_state_dict = preprocess_fn(model, layer_state_dict, layer_id)
            if self.world.is_master:
                broadcast_state_dict(layer_state_dict, self.communicator)


class NCCLWeightSender(WeightSender):
    """Broadcast weights into the inference engine using NCCL."""

    def __init__(
        self,
        output_dir: Path,
        config: NCCLWeightBroadcastConfig,
        device: int | str | torch.device,
    ):
        super().__init__(output_dir, config.timeout)
        self.nccl_broadcast_sender = NCCLBroadcaster(
            config.host,
            config.port,
            0,
            config.inference_world_size + 1,
            device,
            config.timeout,
            quantize_in_weight_transfer=config.quantize_in_weight_transfer,
        )

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        # The master enters only after the receiver acknowledged the handshake,
        # but all ranks must be held back until then: the broadcast preparation
        # (DTensor resolution, quantization) enqueues collectives on non-master
        # ranks, and if those start before the receiver has paused inference,
        # the collectives sit unmatched until NCCL's watchdog kills the process.
        if self.world.world_size > 1:
            dist.barrier()
        self.nccl_broadcast_sender.send(model)


class NCCLWeightReceiver(WeightReceiver):
    """Joins the trainer's NCCL collective. The receiver pauses the engines,
    acknowledges, and sends them into the receive RPC — only then does the
    trainer enter the collective, so the handshake can never race a stale
    marker."""

    async def initialize(self) -> None:
        await init_nccl_broadcast(
            self.admin_clients,
            self.config.host,
            self.config.port,
            self.config.timeout,
            inference_world_size=self.config.inference_world_size,
            quantize_in_weight_transfer=self.config.quantize_in_weight_transfer,
        )

    async def receive(self, step: int) -> None:
        await update_weights(
            self.admin_clients,
            self.step_dir(step),
            step=step,
            on_paused=lambda: self._ack(step),
        )
