import pickle
from functools import partial
from pathlib import Path
from typing import Callable, Generator

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.shared import NCCLWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.utils import get_world
from prime_rl.transports.weights.base import WeightReceiver, WeightSender
from prime_rl.transports.weights.nccl_communicator import WeightBroadcastCommunicator
from prime_rl.utils.logger import get_logger
from prime_rl.utils.nccl import disable_nccl_p2p_if_unavailable, iter_tensor_buckets
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.utils.weights import get_chunked_conversion_ops, resolve_wire_dtype


def _gather_row_chunk_to_master(value: Tensor, start: int, stop: int, dtype: torch.dtype) -> Tensor | None:
    """Assemble one fragment on trainer rank 0 by reducing over its replica's row owners.

    Each owner contributes its rows in an otherwise zero buffer, so the sum on rank 0 is the fragment.
    Requires trainer rank 0 at the origin of `value`'s device mesh, as laid out by `init_device_mesh`.
    """
    rank = dist.get_rank()
    if not isinstance(value, DTensor):
        return value[start:stop].to(dtype) if rank == 0 else None
    mesh = value.device_mesh
    active_dims = [
        i for i, placement in enumerate(value.placements) if mesh.size(i) > 1 and not isinstance(placement, Replicate)
    ]
    if len(active_dims) > 1 or (active_dims and value.placements[active_dims[0]] != Shard(0)):
        raise ValueError("Embedding transfer requires replicated weights or a single row-sharding dimension")
    if not active_dims:
        return value.to_local()[start:stop].to(dtype) if rank == 0 else None
    mesh_dim = active_dims[0]
    index = [0] * mesh.ndim
    index[mesh_dim] = slice(None)
    owners = mesh.mesh[tuple(index)].tolist()
    if rank not in owners:
        return None
    rows_per_rank = (value.shape[0] + len(owners) - 1) // len(owners)
    local = value.to_local()
    offset = min(rows_per_rank * owners.index(rank), value.shape[0])
    left, right = max(start, offset), min(stop, offset + local.shape[0])
    chunk = torch.zeros((stop - start, *value.shape[1:]), dtype=dtype, device=value.device)
    if left < right:
        chunk[left - start : right - start].copy_(local[left - offset : right - offset])
    dist.reduce(chunk, dst=0, group=mesh.get_group(mesh_dim))
    return chunk if rank == 0 else None


def get_max_layer_num(state_dict: dict[str, Tensor], layer_prefix: str = "model.layers.") -> int:
    """Get the maximum number of layers in the model."""
    max_num = -1
    for key in state_dict:
        if not key.startswith(layer_prefix):
            continue
        layer_num_str = key[len(layer_prefix) :].split(".")[0]
        if layer_num_str.isdigit():
            max_num = max(max_num, int(layer_num_str))
    return max_num + 1


def broadcast_integer(integer: int, communicator: PyNcclCommunicator | WeightBroadcastCommunicator) -> None:
    """Broadcast an integer to a process group using NCCL communicator."""
    integer_tensor = torch.tensor([integer], dtype=torch.long).cuda()
    communicator.broadcast(integer_tensor, src=0)


def broadcast_state_dict(
    state_dict: dict[str, Tensor], communicator: PyNcclCommunicator | WeightBroadcastCommunicator
) -> None:
    """Broadcast a state dict to the inference NCCL process group."""
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
    for dtype, tensor_info in metadata.items():
        for bucket in iter_tensor_buckets(tensor_info, dtype):
            flat_tensors = [state_dict[key].flatten() for key, _, _ in bucket]
            concatenated = flat_tensors[0].contiguous() if len(flat_tensors) == 1 else torch.cat(flat_tensors)
            if concatenated.numel():
                communicator.broadcast(concatenated, src=0)
            del concatenated, flat_tensors


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
    is never consulted. TODO: NIXL transport does not use this function; unify logic.
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
    if not layer_state_dict:
        return layer_state_dict
    if isinstance(model, PreTrainedModelPrimeRL) and model.is_prime_state_dict(layer_state_dict):
        model.convert_layer_to_hf(layer_state_dict, layer_idx)
        return layer_state_dict

    from transformers.core_model_loading import revert_weight_conversion

    return revert_weight_conversion(model, layer_state_dict)


class NCCLBroadcaster:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
        recoverable: bool = False,
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = torch.bfloat16

        if self.world.is_master:
            disable_nccl_p2p_if_unavailable()
            # Trainer is on rank 0 in process group with all inference GPUs
            pg = StatelessProcessGroup.create(
                host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
            )
            self.communicator = (
                WeightBroadcastCommunicator(pg, device=device, timeout=timeout)
                if recoverable
                else PyNcclCommunicator(pg, device=device)
            )
            self.logger.debug("Initialized NCCL broadcast on master rank")
        else:
            self.logger.debug("Initialized NCCL broadcast on non-master rank (no communicator)")

    @torch.no_grad()
    def send(self, model: nn.Module) -> None:
        """Broadcast the state dict of a model into the inference pool using NCCL."""
        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        chunked_ops = get_chunked_conversion_ops(model, state_dict)
        num_state_dict_to_send = num_layers + 1 + sum(op.shard_count for op in chunked_ops.values())

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} layer state dicts")
        keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
        for layer_id, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            chunked_weights = {key: layer_state_dict.pop(key) for key in chunked_ops if key in layer_state_dict}
            layer_state_dict = resolve_dtensors(layer_state_dict, keep_in_fp32, self.dtype)
            layer_state_dict = preprocess_layer_checkpoint(model, layer_state_dict, layer_id)
            if self.world.is_master:
                broadcast_state_dict(layer_state_dict, self.communicator)
            del layer_state_dict
            for key, value in chunked_weights.items():
                op = chunked_ops[key]
                dtype = resolve_wire_dtype(keep_in_fp32, key, self.dtype) if isinstance(value, DTensor) else value.dtype
                resolve_rows = partial(_gather_row_chunk_to_master, dtype=dtype)
                for packet in op.iter_prime_to_hf(value, resolve_rows):
                    if self.world.is_master:
                        assert packet is not None
                        broadcast_state_dict(packet, self.communicator)
                    del packet


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
            config.recoverable,
        )

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        self.nccl_broadcast_sender.send(model)


class NCCLWeightReceiver(WeightReceiver):
    """Joins the trainer's NCCL collective. The receiver pauses the engines,
    acknowledges, and sends them into the receive RPC — only then does the
    trainer enter the collective, so the handshake can never race a stale
    marker."""

    async def initialize(self) -> None:
        await self.admin_plane.initialize_nccl(
            host=self.config.host,
            port=self.config.port,
            timeout=self.config.timeout,
            inference_world_size=self.config.inference_world_size,
            recoverable=self.config.recoverable,
        )

    async def reconnect(self) -> None:
        # Each worker aborts its communicator with the replaced trainer and joins a fresh one.
        await self.initialize()

    async def receive(self, step: int) -> None:
        await self.admin_plane.update_weights(
            self.step_dir(step),
            transport="nccl",
            step=step,
            on_paused=lambda: self._ack(step),
        )
