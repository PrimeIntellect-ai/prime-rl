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


# Rows of a row-sharded table sent per broadcast. One engram chunk is 512 MiB in fp8.
TABLE_CHUNK_ROWS = 1 << 21
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
UE8M0_BLOCK = 32


def row_sharded_tables(model: nn.Module, state_dict: dict[str, Tensor]) -> dict[str, nn.Module]:
    """The row-sharded embeddings (`AllToAllEmbeddingParallel`) among `state_dict`'s keys.

    FSDP never gathers them and they are too large to gather whole, so they are sent separately,
    in row chunks.
    """
    tables = {f"{name}.weight": module for name, module in model.named_modules() if hasattr(module, "rows_per_rank")}
    missing = set(tables) - set(state_dict)
    assert not missing, f"row-sharded tables {missing} are not in the state dict"
    return tables


def quantize_rows_ue8m0(rows: Tensor) -> tuple[Tensor, Tensor]:
    """Rows -> fp8 e4m3 values and power-of-two (ue8m0) scales per 1x32 block, as raw bytes.

    Exact for rows dequantized from this format: the new scale is never larger than the original,
    so every value is its original fp8 value times a power of two that keeps it in range.
    """
    blocks = rows.float().unflatten(-1, (-1, UE8M0_BLOCK))
    amax = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / FP8_MAX)).clamp(-127, 127)
    values = (blocks * torch.exp2(-exponent)).to(torch.float8_e4m3fn).flatten(-2)
    return values.view(torch.uint8), (exponent.squeeze(-1) + 127).to(torch.uint8)


def table_chunks(weight: DTensor, rows_per_rank: int) -> list[tuple[int, int, int]]:
    """`(owner, local_start, num_rows)` for every chunk of a `Shard(0)` table, owners in order."""
    num_rows, world = weight.shape[0], weight.device_mesh.size()
    chunks = []
    for owner in range(world):
        local_rows = max(0, min(rows_per_rank, num_rows - owner * rows_per_rank))
        for start in range(0, local_rows, TABLE_CHUNK_ROWS):
            chunks.append((owner, start, min(TABLE_CHUNK_ROWS, local_rows - start)))
    return chunks


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
    ):
        self.logger = get_logger()
        self.world = get_world()
        self.dtype = torch.bfloat16
        # Frozen row-sharded tables never change after their first broadcast, so they are sent once.
        self.sent_frozen_tables: set[str] = set()

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
        tables = row_sharded_tables(model, state_dict)
        table_weights = {name: state_dict.pop(name) for name in tables}
        for name in self.sent_frozen_tables:
            del tables[name], table_weights[name]
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)
        chunks = {name: table_chunks(table_weights[name], tables[name].rows_per_rank) for name in tables}
        chunked_ops = get_chunked_conversion_ops(model, state_dict)
        # every layer, the remaining weights, each chunked conversion's packets, then each row-sharded
        # table chunk by chunk
        num_state_dict_to_send = (
            num_layers + 1 + sum(op.shard_count for op in chunked_ops.values()) + sum(len(c) for c in chunks.values())
        )

        if self.world.is_master:
            broadcast_integer(num_state_dict_to_send, self.communicator)

        self.logger.debug(f"Broadcasting {num_state_dict_to_send} state dicts")
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

        for name, weight in table_weights.items():
            self._send_table(model, name, weight, tables[name].rows_per_rank, chunks[name])
            if not tables[name].weight.requires_grad:
                self.sent_frozen_tables.add(name)

    def _send_table(
        self, model: nn.Module, name: str, weight: DTensor, rows_per_rank: int, chunks: list[tuple[int, int, int]]
    ) -> None:
        """Send a row-sharded table as fp8 + ue8m0 chunks, each owner's rows relayed by the master.

        Every chunk goes out as `<table>.weight` and `<table>.scale` (raw bytes) under the
        checkpoint name, plus `<table>.row_start`, the chunk's first row in the whole table.
        """
        assert isinstance(model, PreTrainedModelPrimeRL)
        hf_name = next(iter(model.convert_to_hf({name: torch.empty(0)})))
        prefix = hf_name.removesuffix(".weight")
        group = weight.device_mesh.get_group()
        rank, master = dist.get_rank(group), dist.get_group_rank(group, 0)
        local, dim = weight.to_local(), weight.shape[1]
        for owner, start, num_rows in chunks:
            if rank == owner:
                values, scales = quantize_rows_ue8m0(local[start : start + num_rows])
                if owner != master:
                    dist.send(values, group_dst=master, group=group)
                    dist.send(scales, group_dst=master, group=group)
            elif rank == master:
                device = local.device
                values = torch.empty(num_rows, dim, dtype=torch.uint8, device=device)
                scales = torch.empty(num_rows, dim // UE8M0_BLOCK, dtype=torch.uint8, device=device)
                dist.recv(values, group_src=owner, group=group)
                dist.recv(scales, group_src=owner, group=group)
            if rank == master:
                row_start = torch.tensor([owner * rows_per_rank + start], dtype=torch.long, device=values.device)
                chunk = {f"{prefix}.weight": values, f"{prefix}.scale": scales, f"{prefix}.row_start": row_start}
                broadcast_state_dict(chunk, self.communicator)


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
        )

    async def receive(self, step: int) -> None:
        await self.admin_plane.update_weights(
            self.step_dir(step),
            transport="nccl",
            step=step,
            on_paused=lambda: self._ack(step),
        )
