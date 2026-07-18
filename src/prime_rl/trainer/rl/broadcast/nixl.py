"""Serve FP32 FSDP master shards through a reusable BF16 NIXL arena."""

from __future__ import annotations

import re
import socket
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
    TrainerGatheredGroup,
    TrainerGatheredShard,
    TrainerReplica,
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
    serves_direct: bool = True
    serves_gather: bool = False
    buffer: torch.Tensor | None = None
    send_offset: int = 0

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

    def bind_gather(self, arena: torch.Tensor, offset: int) -> None:
        self.send_offset = offset
        self.buffer = arena.narrow(0, offset, self.source.numel()).view(self.source.shape)

    def refresh(self) -> None:
        assert self.buffer is not None
        self.buffer.copy_(self.source)


@dataclass(frozen=True)
class TrainerArenaStats:
    has_observed_peak_growth: bool
    free_before_reclaim: int
    free_after_reclaim: int
    total: int
    allocated: int
    peak_allocated: int
    recurring_peak_growth: int
    headroom: int
    post_allocation_free: int
    memory_buffer_count: int
    group_total: int
    largest_group: int
    gathered_roots: int
    gathered_logical_bytes: int
    gathered_output_bytes: int
    gathered_padding_bytes: int
    gather_scratch_bytes: int


@dataclass(frozen=True)
class GatherCandidate:
    name: str
    group: int
    full_shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        return prod(self.full_shape) if self.full_shape else 1

    @property
    def nbytes(self) -> int:
        return self.numel * _WIRE_DTYPE.itemsize


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
        self.serving_group = parallel_dims.get_mesh("dp_shard_cp").get_group()
        self.gather_rank = 0
        self.gather_world_size = 1
        if self.is_serving_rank:
            set_ucx_env_defaults()
            self.nixl_agent = NixlAgent(make_agent_name("trainer", self.world.rank))
            self.gather_rank = dist.get_rank(self.serving_group)
            self.gather_world_size = dist.get_world_size(self.serving_group)
        self.initialized = False
        self.groups: list[str] = []
        self.shards: list[TrainerShardSource] = []
        self.shards_by_group: dict[int, list[TrainerShardSource]] = {}
        self.gathered_names: set[str] = set()
        self.arena: torch.Tensor | None = None
        self.gather_send_arena: torch.Tensor | None = None
        self.gather_outputs: dict[int, torch.Tensor] = {}
        self.gather_send_elements: dict[int, int] = {}
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

    def _is_gather_candidate(self, value: torch.Tensor) -> bool:
        if not isinstance(value, DTensor):
            return True
        if all(placement.is_replicate() for placement in value.placements):
            mesh_ranks = set(value.device_mesh.mesh.flatten().tolist())
            return set(dist.get_process_group_ranks(self.serving_group)).issubset(mesh_ranks)
        shard_dims = [index for index, placement in enumerate(value.placements) if placement.is_shard()]
        if len(shard_dims) != 1 or value.placements[shard_dims[0]].dim != 0:
            return False
        owner_group = value.device_mesh.get_group(mesh_dim=shard_dims[0])
        return dist.get_process_group_ranks(owner_group) == dist.get_process_group_ranks(self.serving_group)

    def _gather_candidates(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_groups: dict[int, int],
    ) -> list[GatherCandidate]:
        return sorted(
            (
                GatherCandidate(name, self._group_for(name, layer_groups), tuple(value.shape))
                for name, value in state_dict.items()
                if value.is_floating_point() and self._is_gather_candidate(value)
            ),
            key=lambda candidate: candidate.name,
        )

    def _balanced_slice(
        self,
        value: torch.Tensor,
        full_shape: tuple[int, ...],
    ) -> tuple[int, torch.Tensor] | None:
        rows = full_shape[0] if full_shape else 1
        row_start = rows * self.gather_rank // self.gather_world_size
        row_end = rows * (self.gather_rank + 1) // self.gather_world_size
        if row_start == row_end:
            return None
        source = value if full_shape else value.reshape(1)
        return row_start, source.narrow(0, row_start, row_end - row_start)

    def _owned_shards(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_groups: dict[int, int],
        gather_candidates: list[GatherCandidate],
    ) -> list[TrainerShardSource]:
        owned: list[TrainerShardSource] = []
        gather_names = {candidate.name for candidate in gather_candidates}
        for name, value in state_dict.items():
            if not value.is_floating_point():
                continue
            full_shape = tuple(value.shape)
            group = self._group_for(name, layer_groups)
            if not isinstance(value, DTensor):
                if self.world.is_master:
                    owned.append(TrainerShardSource(name, full_shape, group, 0, value.detach()))
                if name in gather_names and (piece := self._balanced_slice(value.detach(), full_shape)) is not None:
                    row_start, local = piece
                    owned.append(
                        TrainerShardSource(
                            name,
                            full_shape,
                            group,
                            row_start,
                            local,
                            serves_direct=False,
                            serves_gather=True,
                        )
                    )
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
                if name in gather_names and (piece := self._balanced_slice(local, full_shape)) is not None:
                    row_start, gathered_local = piece
                    owned.append(
                        TrainerShardSource(
                            name,
                            full_shape,
                            group,
                            row_start,
                            gathered_local,
                            serves_direct=False,
                            serves_gather=True,
                        )
                    )
                continue

            if any(global_offset[1:]) or tuple(local_shape[1:]) != full_shape[1:]:
                raise NotImplementedError(
                    f"NIXL currently requires dim-0 FSDP shards; {name} has "
                    f"local_shape={local_shape}, global_offset={global_offset}"
                )
            num_rows = local_shape[0] if full_shape else 1
            if num_rows:
                row_start = global_offset[0] if full_shape else 0
                owned.append(
                    TrainerShardSource(
                        name,
                        full_shape,
                        group,
                        row_start,
                        local,
                        serves_gather=name in gather_names,
                    )
                )
        return owned

    def _allocate_arena(
        self,
        allocated_bytes: int,
        peak_allocated_bytes: int,
        gather_candidates: list[GatherCandidate],
    ) -> TrainerArenaStats:
        group_elements = [0] * len(self.groups)
        for shard in self.shards:
            if shard.serves_direct:
                group_elements[shard.group] += shard.source.numel()
        baseline_largest_elements = max(group_elements, default=0)
        baseline_largest_bytes = baseline_largest_elements * _WIRE_DTYPE.itemsize

        free_before_reclaim = free_after_reclaim = total_bytes = headroom_bytes = 0
        recurring_peak_growth = max(0, peak_allocated_bytes - allocated_bytes)
        has_observed_peak_growth = recurring_peak_growth > 0
        memory_buffer_count = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT)
        local_buffer_count = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT)
        selected_names: set[str] = set()
        gather_offsets: dict[str, int] = {}
        gathered_logical_bytes = gathered_output_bytes = gathered_padding_bytes = 0
        gather_scratch_elements = 0

        if self.is_serving_rank and baseline_largest_bytes:
            device = self.shards[0].source.device
            free_before_reclaim, total_bytes = torch.cuda.mem_get_info(device)
            max_buffers = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT) if has_observed_peak_growth else 1
            if has_observed_peak_growth or free_before_reclaim < baseline_largest_bytes:
                torch.cuda.empty_cache()
            memory_buffer_count, free_after_reclaim, total_bytes, headroom_bytes = cuda_buffer_capacity(
                baseline_largest_bytes,
                max_buffers,
                device,
                extra_headroom_bytes=recurring_peak_growth,
            )

            if gather_candidates:
                local_candidate_elements: dict[str, int] = defaultdict(int)
                local_candidate_direct_elements: dict[str, int] = defaultdict(int)
                for shard in self.shards:
                    if shard.serves_gather:
                        local_candidate_elements[shard.name] += shard.source.numel()
                    if shard.serves_direct:
                        local_candidate_direct_elements[shard.name] += shard.source.numel()
                local_layout = torch.tensor(
                    [
                        *group_elements,
                        *(local_candidate_elements.get(candidate.name, 0) for candidate in gather_candidates),
                        *(local_candidate_direct_elements.get(candidate.name, 0) for candidate in gather_candidates),
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                gathered_layout = torch.empty(
                    self.gather_world_size * local_layout.numel(),
                    dtype=torch.int64,
                    device=device,
                )
                dist.all_gather_into_tensor(gathered_layout, local_layout, group=self.serving_group)
                layouts = gathered_layout.view(self.gather_world_size, -1).cpu().tolist()
                group_layouts = [row[: len(self.groups)] for row in layouts]
                candidate_start = len(self.groups)
                direct_start = candidate_start + len(gather_candidates)
                candidate_layouts = [row[candidate_start:direct_start] for row in layouts]
                candidate_direct_layouts = [row[direct_start:] for row in layouts]

                local_capacity = torch.tensor(
                    [memory_buffer_count, max(0, free_after_reclaim - headroom_bytes)],
                    dtype=torch.int64,
                    device=device,
                )
                gathered_capacity = torch.empty(
                    self.gather_world_size * local_capacity.numel(),
                    dtype=torch.int64,
                    device=device,
                )
                dist.all_gather_into_tensor(gathered_capacity, local_capacity, group=self.serving_group)
                capacities = gathered_capacity.view(self.gather_world_size, -1).cpu().tolist()
                baseline_buffer_count = min(row[0] for row in capacities)
                available_by_rank = [row[1] for row in capacities]
                minimum_gather_buffers = min(4, len(self.groups), _MAX_SOURCE_BUFFER_COUNT)
                selected_buffer_count = baseline_buffer_count

                thresholds = sorted({candidate.nbytes for candidate in gather_candidates})
                selected_indices: list[int] = []
                for threshold in thresholds:
                    candidate_indices = [
                        index for index, candidate in enumerate(gather_candidates) if candidate.nbytes <= threshold
                    ]
                    send_by_group = [0] * len(self.groups)
                    selected_by_rank = [[0] * len(self.groups) for _ in range(self.gather_world_size)]
                    for candidate_index in candidate_indices:
                        group = gather_candidates[candidate_index].group
                        send_by_group[group] += max(row[candidate_index] for row in candidate_layouts)
                        for rank, row in enumerate(candidate_direct_layouts):
                            selected_by_rank[rank][group] += row[candidate_index]

                    group_need = [
                        [
                            group_layouts[rank][group]
                            - selected_by_rank[rank][group]
                            + self.gather_world_size * send_by_group[group]
                            for group in range(len(self.groups))
                        ]
                        for rank in range(self.gather_world_size)
                    ]
                    scratch_elements = max(send_by_group, default=0)
                    candidate_buffer_count = 0
                    for buffer_count in range(
                        minimum_gather_buffers,
                        min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT) + 1,
                    ):
                        if all(
                            (buffer_count * max(group_need[rank], default=0) + scratch_elements) * _WIRE_DTYPE.itemsize
                            <= available_by_rank[rank]
                            for rank in range(self.gather_world_size)
                        ):
                            candidate_buffer_count = buffer_count
                    if candidate_buffer_count:
                        selected_indices = candidate_indices
                        selected_buffer_count = candidate_buffer_count

                selected_names = {gather_candidates[index].name for index in selected_indices}
                selected_by_group: dict[int, list[int]] = defaultdict(list)
                for index in selected_indices:
                    selected_by_group[gather_candidates[index].group].append(index)
                    gathered_logical_bytes += gather_candidates[index].nbytes

                for group, candidate_indices in selected_by_group.items():
                    offset = 0
                    for candidate_index in candidate_indices:
                        gather_offsets[gather_candidates[candidate_index].name] = offset
                        offset += max(row[candidate_index] for row in candidate_layouts)
                    self.gather_send_elements[group] = offset
                    output_elements = self.gather_world_size * offset
                    gathered_output_bytes += output_elements * _WIRE_DTYPE.itemsize
                    gathered_padding_bytes += (
                        output_elements - sum(gather_candidates[index].numel for index in candidate_indices)
                    ) * _WIRE_DTYPE.itemsize
                gather_scratch_elements = max(self.gather_send_elements.values(), default=0)
                local_buffer_count = selected_buffer_count
            else:
                local_buffer_count = memory_buffer_count

        direct_group_elements = [0] * len(self.groups)
        for shard in self.shards:
            if shard.serves_direct and shard.name not in selected_names:
                direct_group_elements[shard.group] += shard.source.numel()
        arena_group_elements = [
            direct_group_elements[group] + self.gather_world_size * self.gather_send_elements.get(group, 0)
            for group in range(len(self.groups))
        ]
        largest_group_elements = max(arena_group_elements, default=0)
        largest_group_bytes = largest_group_elements * _WIRE_DTYPE.itemsize

        buffer_count = torch.tensor(
            local_buffer_count,
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(buffer_count, op=dist.ReduceOp.MIN)
        self.buffer_count = int(buffer_count.item())

        post_free_bytes = free_after_reclaim
        if self.is_serving_rank and largest_group_elements:
            arena_elements = self.buffer_count * largest_group_elements
            with classic_cuda_alloc():
                self.arena = torch.empty(
                    arena_elements,
                    dtype=_WIRE_DTYPE,
                    device=self.shards[0].source.device,
                )
                if gather_scratch_elements:
                    self.gather_send_arena = torch.empty(
                        gather_scratch_elements,
                        dtype=_WIRE_DTYPE,
                        device=self.shards[0].source.device,
                    )
            offsets = [(group % self.buffer_count) * largest_group_elements for group in range(len(self.groups))]
            direct_shards = [shard for shard in self.shards if shard.serves_direct and shard.name not in selected_names]
            gathered_shards = [shard for shard in self.shards if shard.serves_gather and shard.name in selected_names]
            for shard in direct_shards:
                offsets[shard.group] = shard.bind(self.arena, offsets[shard.group])
            for group, send_elements in self.gather_send_elements.items():
                output_elements = self.gather_world_size * send_elements
                self.gather_outputs[group] = self.arena.narrow(0, offsets[group], output_elements)
            assert self.gather_send_arena is not None or not gathered_shards
            for shard in gathered_shards:
                shard.bind_gather(self.gather_send_arena, gather_offsets[shard.name])
            self.nixl_agent.register_tensor(self.arena)
            post_free_bytes, _ = torch.cuda.mem_get_info(self.shards[0].source.device)

        grouped: dict[int, list[TrainerShardSource]] = defaultdict(list)
        for shard in self.shards:
            if (shard.serves_direct and shard.name not in selected_names) or (
                shard.serves_gather and shard.name in selected_names
            ):
                grouped[shard.group].append(shard)
        self.shards_by_group = dict(grouped)
        self.gathered_names = selected_names
        return TrainerArenaStats(
            has_observed_peak_growth=has_observed_peak_growth,
            free_before_reclaim=free_before_reclaim,
            free_after_reclaim=free_after_reclaim,
            total=total_bytes,
            allocated=allocated_bytes,
            peak_allocated=peak_allocated_bytes,
            recurring_peak_growth=recurring_peak_growth,
            headroom=headroom_bytes,
            post_allocation_free=post_free_bytes,
            memory_buffer_count=memory_buffer_count,
            group_total=sum(group_elements) * _WIRE_DTYPE.itemsize,
            largest_group=largest_group_bytes,
            gathered_roots=len(selected_names),
            gathered_logical_bytes=gathered_logical_bytes,
            gathered_output_bytes=gathered_output_bytes,
            gathered_padding_bytes=gathered_padding_bytes,
            gather_scratch_bytes=gather_scratch_elements * _WIRE_DTYPE.itemsize,
        )

    def _lazy_init(self, model: nn.Module) -> None:
        if self.initialized:
            return
        allocated_bytes = torch.cuda.memory_allocated() if self.is_serving_rank else 0
        peak_allocated_bytes = torch.cuda.max_memory_allocated() if self.is_serving_rank else 0
        state_dict = model.state_dict()
        self.groups, layer_groups = self._transfer_groups(state_dict)
        gather_candidates: list[GatherCandidate] = []
        if self.is_serving_rank:
            gather_candidates = self._gather_candidates(state_dict, layer_groups)
            self.shards = self._owned_shards(state_dict, layer_groups, gather_candidates)
        buffer_stats = self._allocate_arena(allocated_bytes, peak_allocated_bytes, gather_candidates)

        payload = None
        if self.is_serving_rank:
            payload = (
                self.world.rank,
                self.nixl_agent.name,
                self.nixl_agent.get_metadata(),
                self.arena.nbytes if self.arena is not None else 0,
                buffer_stats,
                socket.gethostname(),
                self.gather_rank,
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
                    if shard.serves_direct and shard.name not in self.gathered_names and shard.buffer is not None
                ],
                [
                    (
                        shard.name,
                        shard.full_shape,
                        shard.group,
                        shard.row_start,
                        shard.num_rows,
                        (self.gather_rank * self.gather_send_elements[shard.group] + shard.send_offset)
                        * _WIRE_DTYPE.itemsize,
                        shard.row_numel * _WIRE_DTYPE.itemsize,
                    )
                    for shard in self.shards
                    if shard.serves_gather and shard.name in self.gathered_names
                ],
                [(group, output.data_ptr(), output.device.index) for group, output in self.gather_outputs.items()],
            )
        gathered: list | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(payload, gathered, dst=0)

        if self.world.is_master:
            assert gathered is not None
            parts = sorted((part for part in gathered if part is not None), key=lambda part: part[0])
            agents = [TrainerAgent(name=part[1], metadata=part[2]) for part in parts]
            tensors: dict[str, TrainerTensor] = {}
            for agent_index, part in enumerate(parts):
                for name, shape, group, row_start, num_rows, addr, row_bytes, device_id in part[7]:
                    tensor = tensors.setdefault(
                        name,
                        TrainerTensor(
                            name=name,
                            master_dtype="float32",
                            dtype="bfloat16",
                            shape=tuple(shape),
                            group=group,
                            shards=[],
                            gathered_shards=[],
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
                for name, shape, group, row_start, num_rows, offset, row_bytes in part[8]:
                    tensor = tensors.setdefault(
                        name,
                        TrainerTensor(
                            name=name,
                            master_dtype="float32",
                            dtype="bfloat16",
                            shape=tuple(shape),
                            group=group,
                            shards=[],
                            gathered_shards=[],
                        ),
                    )
                    tensor.gathered_shards.append(
                        TrainerGatheredShard(
                            row_start=row_start,
                            num_rows=num_rows,
                            offset_bytes=offset,
                            row_bytes=row_bytes,
                        )
                    )

            host_agents: dict[str, list[int]] = defaultdict(list)
            for agent_index, part in enumerate(parts):
                host_agents[part[5]].append(agent_index)
            replica_order = [
                agents_on_host[local_index]
                for local_index in range(max(map(len, host_agents.values()), default=0))
                for agents_on_host in host_agents.values()
                if local_index < len(agents_on_host)
            ]
            replica_bases = [dict((group, (addr, device_id)) for group, addr, device_id in part[9]) for part in parts]
            gathered_groups = [
                TrainerGatheredGroup(
                    group=group,
                    replicas=[
                        TrainerReplica(
                            agent=agent_index,
                            addr=replica_bases[agent_index][group][0],
                            device_id=replica_bases[agent_index][group][1],
                        )
                        for agent_index in replica_order
                    ],
                )
                for group in sorted(self.gather_send_elements)
            ]
            table = TrainerTable(
                agents=agents,
                groups=self.groups,
                buffer_count=self.buffer_count,
                tensors=list(tensors.values()),
                gathered_groups=gathered_groups,
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
            total_bytes = sum(
                shard.num_rows * shard.row_bytes
                for tensor in table.tensors
                for shard in (*tensor.shards, *tensor.gathered_shards)
            )
            max_arena_bytes = max((part[3] for part in parts), default=0)
            stats = [part[4] for part in parts]
            sizing_modes = {"measured" if item.has_observed_peak_growth else "unmeasured" for item in stats}
            free_before_values = [item.free_before_reclaim for item in stats]
            free_after_values = [item.free_after_reclaim for item in stats]
            reclaimed_values = [after - before for before, after in zip(free_before_values, free_after_values)]
            total_values = [item.total for item in stats]
            allocated_values = [item.allocated for item in stats]
            peak_allocated_values = [item.peak_allocated for item in stats]
            peak_growth_values = [item.recurring_peak_growth for item in stats]
            headroom_values = [item.headroom for item in stats]
            post_free_values = [item.post_allocation_free for item in stats]
            memory_buffer_values = [item.memory_buffer_count for item in stats]
            group_total = sum(item.group_total for item in stats)
            buffer_min = min((item.largest_group for item in stats if item.largest_group), default=0)
            group_max = max((item.largest_group for item in stats), default=0)
            gathered_roots = min(item.gathered_roots for item in stats)
            gathered_logical_bytes = min(item.gathered_logical_bytes for item in stats)
            gathered_output_bytes = min(item.gathered_output_bytes for item in stats)
            gathered_padding_bytes = min(item.gathered_padding_bytes for item in stats)
            gather_scratch_bytes = max(item.gather_scratch_bytes for item in stats)
            gathered_root_limit = max(
                (
                    (prod(tensor.shape) if tensor.shape else 1) * _WIRE_DTYPE.itemsize
                    for tensor in table.tensors
                    if tensor.gathered_shards
                ),
                default=0,
            )
            gib = 1024**3
            self.logger.info(
                f"NIXL staging ring selected {self.buffer_count} buffers from one-time first-transfer sizing "
                f"({','.join(sorted(sizing_modes))} peak mode); CUDA free before cache reclaim "
                f"{min(free_before_values) / gib:.2f}-{max(free_before_values) / gib:.2f} GiB per rank, "
                f"after reclaim {min(free_after_values) / gib:.2f}-{max(free_after_values) / gib:.2f} GiB "
                f"({min(reclaimed_values) / gib:.2f}-{max(reclaimed_values) / gib:.2f} GiB reclaimed); "
                f"active allocation {min(allocated_values) / gib:.2f}-{max(allocated_values) / gib:.2f} GiB, "
                f"observed peak {min(peak_allocated_values) / gib:.2f}-{max(peak_allocated_values) / gib:.2f} GiB, "
                f"preserving {min(peak_growth_values) / gib:.2f}-{max(peak_growth_values) / gib:.2f} GiB peak growth "
                f"inside {min(headroom_values) / gib:.2f}-{max(headroom_values) / gib:.2f} GiB total headroom "
                f"on {min(total_values) / gib:.2f} GiB GPUs; effective local ring candidates "
                f"{min(memory_buffer_values)}-{max(memory_buffer_values)} "
                f"(policy cap {_MAX_SOURCE_BUFFER_COUNT}), "
                f"each buffer {buffer_min / gib:.2f}-{group_max / gib:.2f} GiB across ranks, "
                f"arenas up to {max_arena_bytes / gib:.2f} GiB, "
                f"post-allocation free {min(post_free_values) / gib:.2f}-{max(post_free_values) / gib:.2f} GiB, "
                f"{group_total / max(1, len(parts) * len(table.groups)) / 1e6:.1f} MB average logical group payload"
            )
            self.logger.info(
                f"Replicated gather selected {gathered_roots} roots / {gathered_logical_bytes / 1e9:.2f} GB "
                f"per update up to {gathered_root_limit / 1e6:.1f} MB/root; "
                f"{gathered_output_bytes / 1e9:.2f} GB rank-major group output "
                f"({gathered_padding_bytes / 1e6:.1f} MB padding), {gather_scratch_bytes / 1e9:.2f} GB scratch, "
                f"{len(table.gathered_groups)} gathered groups across {len(parts)} replicas"
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
        gathered_groups = {group.group for group in table.gathered_groups}
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
            shards = tensor.gathered_shards or tensor.shards
            if tensor.gathered_shards and tensor.group not in gathered_groups:
                raise RuntimeError(f"{tensor.name}: missing gathered replicas for group {tensor.group}")
            for shard in sorted(shards, key=lambda item: item.row_start):
                if isinstance(shard, TrainerShard) and not 0 <= shard.agent < len(table.agents):
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
        gather_seconds = 0.0

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
                output = self.gather_outputs.get(group)
                if output is not None:
                    gather_started = time.perf_counter()
                    assert self.gather_send_arena is not None
                    dist.all_gather_into_tensor(
                        output,
                        self.gather_send_arena.narrow(0, 0, self.gather_send_elements[group]),
                        group=self.serving_group,
                    )
                    torch.cuda.synchronize()
                    gather_seconds += time.perf_counter() - gather_started
                else:
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
        self.logger.info(
            f"NIXL+MX policy v{step} synchronized in {time.perf_counter() - start:.2f}s "
            f"(replicated gather {gather_seconds:.2f}s)"
        )
