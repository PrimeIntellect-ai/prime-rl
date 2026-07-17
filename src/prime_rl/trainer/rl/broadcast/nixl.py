"""Serve FP32 FSDP master shards through a reusable BF16 NIXL arena."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc, cuda_buffer_capacity
from prime_rl.weight_transfer.mx import MxRendezvous
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.wire import (
    TrainerAgent,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
    encode_table,
)

_WIRE_DTYPE = torch.bfloat16
_MASTER_DTYPE = torch.float32
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")
_LAYER_SESSION_SUFFIX = ":layers"
_BUFFER_POLL_INTERVAL = 0.01
_MAX_SOURCE_BUFFER_COUNT = 8


@dataclass
class TrainerShardSource:
    name: str
    full_shape: tuple[int, ...]
    group: int
    row_start: int
    source: torch.Tensor
    buffer: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.source.dtype != _MASTER_DTYPE:
            raise TypeError(f"NIXL source {self.name!r} must be an FP32 master shard, got {self.source.dtype}")
        if not self.source.is_contiguous():
            raise ValueError(f"NIXL source {self.name!r} must be contiguous")

    @property
    def num_rows(self) -> int:
        return self.source.shape[0] if self.source.ndim else 1

    @property
    def row_numel(self) -> int:
        return self.source[0].numel() if self.source.ndim else 1

    def bind(self, arena: torch.Tensor, offset: int) -> int:
        self.buffer = arena.narrow(0, offset, self.source.numel()).view(self.source.shape)
        return offset + self.source.numel()

    def refresh(self) -> None:
        assert self.buffer is not None
        self.buffer.copy_(self.source)


class NIXLWeightBroadcast(WeightBroadcast):
    def __init__(self, output_dir: Path, config: NIXLWeightBroadcastConfig, parallel_dims: ParallelDims) -> None:
        super().__init__(output_dir)
        self.config = config
        self.parallel_dims = parallel_dims
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        if parallel_dims.pp_enabled:
            raise NotImplementedError("NIXL weight transfer does not support pipeline parallelism")
        if config.session_id == "default":
            raise ValueError("NIXL weight transfer requires a run-unique, non-default session_id")
        if self.is_serving_rank:
            set_ucx_env_defaults()
            self.nixl_agent = NixlAgent(make_agent_name("trainer", self.world.rank))
        self.initialized = False
        self.groups: list[str] = []
        self.shards: list[TrainerShardSource] = []
        self.shards_by_group: dict[int, list[TrainerShardSource]] = {}
        self.arena: torch.Tensor | None = None
        self.buffer_count = 1

    @property
    def is_serving_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    @staticmethod
    def _transfer_groups(state_dict: dict[str, torch.Tensor]) -> tuple[list[str], dict[int, int]]:
        layer_numbers = sorted(
            {
                int(match.group(1))
                for name, value in state_dict.items()
                if value.is_floating_point() and (match := _LAYER_RE.search(name)) is not None
            }
        )
        return ["non_layer", *(f"layer.{layer}" for layer in layer_numbers)], {
            layer: group for group, layer in enumerate(layer_numbers, start=1)
        }

    @staticmethod
    def _group_for(name: str, layer_groups: dict[int, int]) -> int:
        match = _LAYER_RE.search(name)
        return 0 if match is None else layer_groups[int(match.group(1))]

    def _owned_shards(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_groups: dict[int, int],
    ) -> list[TrainerShardSource]:
        owned: list[TrainerShardSource] = []
        for name, value in state_dict.items():
            if not value.is_floating_point():
                continue
            full_shape = tuple(value.shape)
            group = self._group_for(name, layer_groups)
            if not isinstance(value, DTensor):
                if self.world.is_master:
                    owned.append(TrainerShardSource(name, full_shape, group, 0, value.detach()))
                continue

            placements = value.placements
            local_shape, global_offset = compute_local_shape_and_global_offset(
                value.shape, value.device_mesh, placements
            )
            local = value.to_local().detach()
            if tuple(local.shape) != tuple(local_shape):
                local = local[tuple(slice(size) for size in local_shape)]
            if all(placement.is_replicate() for placement in placements):
                if self.world.is_master:
                    owned.append(TrainerShardSource(name, full_shape, group, 0, local))
                continue

            if any(global_offset[1:]) or tuple(local_shape[1:]) != full_shape[1:]:
                raise NotImplementedError(
                    f"NIXL currently requires dim-0 FSDP shards; {name} has "
                    f"local_shape={local_shape}, global_offset={global_offset}"
                )
            num_rows = local_shape[0] if full_shape else 1
            if num_rows:
                row_start = global_offset[0] if full_shape else 0
                owned.append(TrainerShardSource(name, full_shape, group, row_start, local))
        return owned

    def _allocate_arena(self) -> tuple[int, int, int, int, int, int, int]:
        group_elements = [0] * len(self.groups)
        for shard in self.shards:
            group_elements[shard.group] += shard.source.numel()
        largest_group_elements = max(group_elements, default=0)
        largest_group_bytes = largest_group_elements * _WIRE_DTYPE.itemsize

        free_bytes = total_bytes = headroom_bytes = 0
        memory_buffer_count = len(self.groups)
        local_buffer_count = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT)
        if self.is_serving_rank and largest_group_bytes:
            device = self.shards[0].source.device
            memory_buffer_count, free_bytes, total_bytes, headroom_bytes = cuda_buffer_capacity(
                largest_group_bytes,
                len(self.groups),
                device,
            )
            local_buffer_count = min(memory_buffer_count, _MAX_SOURCE_BUFFER_COUNT)

        buffer_count = torch.tensor(
            local_buffer_count,
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(buffer_count, op=dist.ReduceOp.MIN)
        self.buffer_count = int(buffer_count.item())

        post_free_bytes = free_bytes
        if self.is_serving_rank and largest_group_elements:
            arena_elements = self.buffer_count * largest_group_elements
            arena_bytes = arena_elements * _WIRE_DTYPE.itemsize
            if free_bytes < arena_bytes:
                torch.cuda.empty_cache()
            with classic_cuda_alloc():
                self.arena = torch.empty(
                    arena_elements,
                    dtype=_WIRE_DTYPE,
                    device=self.shards[0].source.device,
                )
            offsets = [(group % self.buffer_count) * largest_group_elements for group in range(len(self.groups))]
            for shard in self.shards:
                offsets[shard.group] = shard.bind(self.arena, offsets[shard.group])
            self.nixl_agent.register_tensor(self.arena)
            post_free_bytes, _ = torch.cuda.mem_get_info(self.shards[0].source.device)

        grouped: dict[int, list[TrainerShardSource]] = defaultdict(list)
        for shard in self.shards:
            grouped[shard.group].append(shard)
        self.shards_by_group = dict(grouped)
        return (
            free_bytes,
            total_bytes,
            headroom_bytes,
            post_free_bytes,
            memory_buffer_count,
            sum(group_elements) * _WIRE_DTYPE.itemsize,
            largest_group_bytes,
        )

    def _lazy_init(self, model: nn.Module) -> None:
        if self.initialized:
            return
        state_dict = model.state_dict()
        self.groups, layer_groups = self._transfer_groups(state_dict)
        if self.is_serving_rank:
            self.shards = self._owned_shards(state_dict, layer_groups)
        buffer_stats = self._allocate_arena()

        payload = None
        if self.is_serving_rank:
            payload = (
                self.world.rank,
                self.nixl_agent.name,
                self.nixl_agent.get_metadata(),
                self.arena.nbytes if self.arena is not None else 0,
                buffer_stats,
                [
                    (
                        shard.name,
                        shard.full_shape,
                        shard.group,
                        shard.row_start,
                        shard.num_rows,
                        shard.buffer.data_ptr(),
                        shard.row_numel * shard.buffer.element_size(),
                        shard.buffer.device.index,
                    )
                    for shard in self.shards
                    if shard.buffer is not None
                ],
            )
        gathered: list | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(payload, gathered, dst=0)

        if self.world.is_master:
            assert gathered is not None
            parts = sorted((part for part in gathered if part is not None), key=lambda part: part[0])
            agents = [TrainerAgent(name=name, metadata=metadata) for _, name, metadata, _, _, _ in parts]
            tensors: dict[str, TrainerTensor] = {}
            for agent_index, (_, _, _, _, _, rows) in enumerate(parts):
                for name, shape, group, row_start, num_rows, addr, row_bytes, device_id in rows:
                    tensor = tensors.setdefault(
                        name,
                        TrainerTensor(
                            name=name,
                            master_dtype="float32",
                            dtype="bfloat16",
                            shape=tuple(shape),
                            group=group,
                            shards=[],
                        ),
                    )
                    if (
                        tensor.shape != tuple(shape)
                        or tensor.master_dtype != "float32"
                        or tensor.dtype != "bfloat16"
                        or tensor.group != group
                    ):
                        raise RuntimeError(f"inconsistent trainer metadata for tensor {name!r}")
                    tensor.shards.append(
                        TrainerShard(
                            agent=agent_index,
                            row_start=row_start,
                            num_rows=num_rows,
                            addr=addr,
                            row_bytes=row_bytes,
                            device_id=device_id,
                        )
                    )
            table = TrainerTable(
                agents=agents,
                groups=self.groups,
                buffer_count=self.buffer_count,
                tensors=list(tensors.values()),
            )
            self._validate_table(table)
            server_url = f"{self.config.host}:{self.config.port}"
            client = MxClient(server_url=server_url)
            self.buffer_rendezvous = []
            for buffer_index in range(self.buffer_count):
                rendezvous = MxRendezvous(
                    client=client,
                    role="trainer",
                    rank=0,
                    peer_world_size=self.config.inference_world_size,
                    session_id=f"{self.config.session_id}{_LAYER_SESSION_SUFFIX}:{buffer_index}",
                    worker_id=f"trainer-buffer-{buffer_index}",
                )
                rendezvous.publish()
                rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                self.buffer_rendezvous.append(rendezvous)
            self.rendezvous = MxRendezvous(
                client=client,
                role="trainer",
                rank=0,
                peer_world_size=self.config.inference_world_size,
                session_id=self.config.session_id,
                worker_id="trainer-table",
            )
            self.rendezvous.publish(nixl_metadata=encode_table(table))
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            total_bytes = sum(shard.num_rows * shard.row_bytes for tensor in table.tensors for shard in tensor.shards)
            max_arena_bytes = max((part[3] for part in parts), default=0)
            free_values = [part[4][0] for part in parts]
            total_values = [part[4][1] for part in parts]
            headroom_values = [part[4][2] for part in parts]
            post_free_values = [part[4][3] for part in parts]
            memory_buffer_values = [part[4][4] for part in parts]
            group_total = sum(part[4][5] for part in parts)
            buffer_min = min((part[4][6] for part in parts if part[4][6]), default=0)
            group_max = max((part[4][6] for part in parts), default=0)
            gib = 1024**3
            self.logger.info(
                f"NIXL staging ring selected {self.buffer_count} buffers from "
                f"{min(free_values) / gib:.2f}-{max(free_values) / gib:.2f} GiB CUDA free per rank "
                f"({min(total_values) / gib:.2f} GiB total, {max(headroom_values) / gib:.2f} GiB target headroom); "
                f"memory permits {min(memory_buffer_values)}-{max(memory_buffer_values)} buffers per rank, "
                f"each buffer {buffer_min / gib:.2f}-{group_max / gib:.2f} GiB across ranks, "
                f"arenas up to {max_arena_bytes / gib:.2f} GiB, "
                f"post-allocation free {min(post_free_values) / gib:.2f}-{max(post_free_values) / gib:.2f} GiB, "
                f"{group_total / max(1, len(parts) * len(table.groups)) / 1e6:.1f} MB average logical group payload"
            )
            self.logger.info(
                f"Published {len(table.tensors)} FP32-master/BF16-wire tensors in {len(table.groups)} groups "
                f"from {len(table.agents)} trainer agents ({total_bytes / 1e9:.2f} GB per update, "
                f"{max_arena_bytes / 1e9:.2f} GB largest local arena, {self.buffer_count} staging buffers)"
            )
        self.initialized = True

    @staticmethod
    def _validate_table(table: TrainerTable) -> None:
        """Fail before publication unless every logical tensor is tiled once."""
        for tensor in table.tensors:
            if not 0 <= tensor.group < len(table.groups):
                raise RuntimeError(f"{tensor.name}: invalid transfer group {tensor.group}")
            if tensor.master_dtype != "float32" or tensor.dtype != "bfloat16":
                raise RuntimeError(
                    f"{tensor.name}: expected FP32 master/BF16 wire metadata, got {tensor.master_dtype}/{tensor.dtype}"
                )
            rows = tensor.shape[0] if tensor.shape else 1
            expected_row_bytes = (prod(tensor.shape[1:]) if tensor.shape else 1) * _WIRE_DTYPE.itemsize
            cursor = 0
            for shard in sorted(tensor.shards, key=lambda item: item.row_start):
                if not 0 <= shard.agent < len(table.agents):
                    raise RuntimeError(f"{tensor.name}: invalid trainer agent index {shard.agent}")
                if shard.row_start != cursor:
                    raise RuntimeError(
                        f"{tensor.name}: shards do not tile dim 0 at row {cursor}; next starts at {shard.row_start}"
                    )
                if shard.num_rows <= 0 or shard.row_bytes != expected_row_bytes:
                    raise RuntimeError(
                        f"{tensor.name}: invalid shard rows/row_bytes "
                        f"({shard.num_rows}, {shard.row_bytes}); expected row_bytes={expected_row_bytes}"
                    )
                cursor += shard.num_rows
            if cursor != rows:
                raise RuntimeError(f"{tensor.name}: shard rows cover [0, {cursor}), expected [0, {rows})")

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        ready = list(self.multi_run_manager.ready_to_update_idxs)
        if not ready:
            self.logger.warning(f"No run requested NIXL weights at step {step}; skipping")
            return
        self._lazy_init(model)
        start = time.perf_counter()

        if self.world.is_master:
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.rendezvous.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.rendezvous.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )

        for group, group_name in enumerate(self.groups):
            group_start = time.perf_counter()
            buffer_index = group % self.buffer_count
            if group >= self.buffer_count:
                if self.world.is_master:
                    rendezvous = self.buffer_rendezvous[buffer_index]
                    rendezvous.wait_for(
                        "inference",
                        count=self.config.inference_world_size,
                        status=p2p_pb2.SOURCE_STATUS_READY,
                        timeout=self.config.timeout,
                        poll_interval=_BUFFER_POLL_INTERVAL,
                    )
                    rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                    rendezvous.wait_for(
                        "inference",
                        count=self.config.inference_world_size,
                        status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                        timeout=self.config.timeout,
                        poll_interval=_BUFFER_POLL_INTERVAL,
                    )
                dist.barrier()

            if self.is_serving_rank:
                for shard in self.shards_by_group.get(group, ()):
                    shard.refresh()
                torch.cuda.synchronize()
            dist.barrier()
            if self.world.is_master:
                self.buffer_rendezvous[buffer_index].set_status(p2p_pb2.SOURCE_STATUS_READY)
                self.logger.debug(
                    f"NIXL+MX policy v{step} group {group_name} staged in buffer {buffer_index} in "
                    f"{time.perf_counter() - group_start:.2f}s"
                )

        first_pending_group = max(0, len(self.groups) - self.buffer_count)
        for group in range(first_pending_group, len(self.groups)):
            buffer_index = group % self.buffer_count
            if self.world.is_master:
                rendezvous = self.buffer_rendezvous[buffer_index]
                rendezvous.wait_for(
                    "inference",
                    count=self.config.inference_world_size,
                    status=p2p_pb2.SOURCE_STATUS_READY,
                    timeout=self.config.timeout,
                    poll_interval=_BUFFER_POLL_INTERVAL,
                )
                rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                rendezvous.wait_for(
                    "inference",
                    count=self.config.inference_world_size,
                    status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                    timeout=self.config.timeout,
                    poll_interval=_BUFFER_POLL_INTERVAL,
                )
            dist.barrier()

        if self.world.is_master:
            self.rendezvous.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.rendezvous.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        dist.barrier()
        for run_index in ready:
            self.multi_run_manager.ready_to_update[run_index] = False
        self.logger.info(f"NIXL+MX policy v{step} synchronized in {time.perf_counter() - start:.2f}s")
