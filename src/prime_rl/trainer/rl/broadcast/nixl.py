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

    def _allocate_arena(self, allocated_bytes: int, peak_allocated_bytes: int) -> TrainerArenaStats:
        group_elements = [0] * len(self.groups)
        for shard in self.shards:
            group_elements[shard.group] += shard.source.numel()
        largest_group_elements = max(group_elements, default=0)
        largest_group_bytes = largest_group_elements * _WIRE_DTYPE.itemsize

        free_before_reclaim = free_after_reclaim = total_bytes = headroom_bytes = 0
        recurring_peak_growth = max(0, peak_allocated_bytes - allocated_bytes)
        has_observed_peak_growth = recurring_peak_growth > 0
        memory_buffer_count = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT)
        local_buffer_count = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT)
        if self.is_serving_rank and largest_group_bytes:
            device = self.shards[0].source.device
            free_before_reclaim, total_bytes = torch.cuda.mem_get_info(device)
            max_buffers = min(len(self.groups), _MAX_SOURCE_BUFFER_COUNT) if has_observed_peak_growth else 1
            if has_observed_peak_growth or free_before_reclaim < largest_group_bytes:
                torch.cuda.empty_cache()
            memory_buffer_count, free_after_reclaim, total_bytes, headroom_bytes = cuda_buffer_capacity(
                largest_group_bytes,
                max_buffers,
                device,
                extra_headroom_bytes=recurring_peak_growth,
            )
            local_buffer_count = memory_buffer_count

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
            offsets = [(group % self.buffer_count) * largest_group_elements for group in range(len(self.groups))]
            for shard in self.shards:
                offsets[shard.group] = shard.bind(self.arena, offsets[shard.group])
            self.nixl_agent.register_tensor(self.arena)
            post_free_bytes, _ = torch.cuda.mem_get_info(self.shards[0].source.device)

        grouped: dict[int, list[TrainerShardSource]] = defaultdict(list)
        for shard in self.shards:
            grouped[shard.group].append(shard)
        self.shards_by_group = dict(grouped)
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
        )

    def _lazy_init(self, model: nn.Module) -> None:
        if self.initialized:
            return

        init_started = time.perf_counter()
        source_plan_started = time.perf_counter()
        allocated_bytes = torch.cuda.memory_allocated() if self.is_serving_rank else 0
        peak_allocated_bytes = torch.cuda.max_memory_allocated() if self.is_serving_rank else 0
        state_dict = model.state_dict()
        self.groups, layer_groups = self._transfer_groups(state_dict)
        if self.is_serving_rank:
            self.shards = self._owned_shards(state_dict, layer_groups)
        source_plan_seconds = time.perf_counter() - source_plan_started

        arena_started = time.perf_counter()
        buffer_stats = self._allocate_arena(allocated_bytes, peak_allocated_bytes)
        arena_seconds = time.perf_counter() - arena_started

        metadata_gather_started = time.perf_counter()
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
        metadata_gather_seconds = time.perf_counter() - metadata_gather_started

        if self.world.is_master:
            table_started = time.perf_counter()
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
            table_seconds = time.perf_counter() - table_started

            mx_publish_started = time.perf_counter()
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
            mx_publish_seconds = time.perf_counter() - mx_publish_started
            total_bytes = sum(shard.num_rows * shard.row_bytes for tensor in table.tensors for shard in tensor.shards)
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
                f"Published {len(table.tensors)} FP32-master/BF16-wire tensors in {len(table.groups)} groups "
                f"from {len(table.agents)} trainer agents ({total_bytes / 1e9:.2f} GB per update, "
                f"{max_arena_bytes / 1e9:.2f} GB largest local arena, {self.buffer_count} staging buffers)"
            )
            init_seconds = time.perf_counter() - init_started
            measured_seconds = (
                source_plan_seconds + arena_seconds + metadata_gather_seconds + table_seconds + mx_publish_seconds
            )
            self.logger.info(
                "Weight update initialization profile role=trainer rank=0: "
                f"source_plan_rank0={source_plan_seconds:.3f}s, arena_rank0={arena_seconds:.3f}s, "
                f"metadata_gather_wall={metadata_gather_seconds:.3f}s, table_rank0={table_seconds:.3f}s, "
                f"mx_publish_rank0={mx_publish_seconds:.3f}s, "
                f"unattributed_rank0={max(0.0, init_seconds - measured_seconds):.3f}s, "
                f"total_rank0={init_seconds:.3f}s"
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

        call_started = time.perf_counter()
        lazy_init_started = time.perf_counter()
        self._lazy_init(model)
        lazy_init_seconds = time.perf_counter() - lazy_init_started
        sync_started = time.perf_counter()

        policy_ready_publish_seconds = 0.0
        orchestrator_ready_wait_seconds = 0.0
        inference_initializing_wait_seconds = 0.0
        recycle_ready_wait_seconds = 0.0
        recycle_reset_publish_seconds = 0.0
        recycle_initializing_wait_seconds = 0.0
        recycle_barrier_seconds = 0.0
        stage_copy_seconds = 0.0
        stage_barrier_seconds = 0.0
        buffer_ready_publish_seconds = 0.0
        tail_ready_wait_seconds = 0.0
        tail_reset_publish_seconds = 0.0
        tail_initializing_wait_seconds = 0.0
        tail_barrier_seconds = 0.0
        inference_complete_wait_seconds = 0.0
        orchestrator_complete_wait_seconds = 0.0
        policy_reset_publish_seconds = 0.0
        slowest_group_seconds = 0.0
        slowest_group_name = self.groups[0]
        slowest_stage_seconds = 0.0
        slowest_stage_name = self.groups[0]

        if self.world.is_master:
            phase_started = time.perf_counter()
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
            policy_ready_publish_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self.rendezvous.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            orchestrator_ready_wait_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self.rendezvous.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            inference_initializing_wait_seconds = time.perf_counter() - phase_started

        for group, group_name in enumerate(self.groups):
            group_start = time.perf_counter()
            buffer_index = group % self.buffer_count
            if group >= self.buffer_count:
                if self.world.is_master:
                    rendezvous = self.buffer_rendezvous[buffer_index]
                    phase_started = time.perf_counter()
                    rendezvous.wait_for(
                        "inference",
                        count=self.config.inference_world_size,
                        status=p2p_pb2.SOURCE_STATUS_READY,
                        timeout=self.config.timeout,
                        poll_interval=_BUFFER_POLL_INTERVAL,
                    )
                    recycle_ready_wait_seconds += time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                    recycle_reset_publish_seconds += time.perf_counter() - phase_started

                    phase_started = time.perf_counter()
                    rendezvous.wait_for(
                        "inference",
                        count=self.config.inference_world_size,
                        status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                        timeout=self.config.timeout,
                        poll_interval=_BUFFER_POLL_INTERVAL,
                    )
                    recycle_initializing_wait_seconds += time.perf_counter() - phase_started
                phase_started = time.perf_counter()
                dist.barrier()
                recycle_barrier_seconds += time.perf_counter() - phase_started

            stage_wall_started = time.perf_counter()
            phase_started = time.perf_counter()
            if self.is_serving_rank:
                for shard in self.shards_by_group.get(group, ()):
                    shard.refresh()
                torch.cuda.synchronize()
            stage_copy_seconds += time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            dist.barrier()
            stage_barrier_seconds += time.perf_counter() - phase_started
            group_stage_seconds = time.perf_counter() - stage_wall_started
            if group_stage_seconds > slowest_stage_seconds:
                slowest_stage_seconds = group_stage_seconds
                slowest_stage_name = group_name
            if self.world.is_master:
                phase_started = time.perf_counter()
                self.buffer_rendezvous[buffer_index].set_status(p2p_pb2.SOURCE_STATUS_READY)
                buffer_ready_publish_seconds += time.perf_counter() - phase_started
                group_seconds = time.perf_counter() - group_start
                if group_seconds > slowest_group_seconds:
                    slowest_group_seconds = group_seconds
                    slowest_group_name = group_name
                self.logger.debug(
                    f"NIXL+MX policy v{step} group {group_name} staged in buffer {buffer_index} in {group_seconds:.2f}s"
                )

        first_pending_group = max(0, len(self.groups) - self.buffer_count)
        for group in range(first_pending_group, len(self.groups)):
            buffer_index = group % self.buffer_count
            if self.world.is_master:
                rendezvous = self.buffer_rendezvous[buffer_index]
                phase_started = time.perf_counter()
                rendezvous.wait_for(
                    "inference",
                    count=self.config.inference_world_size,
                    status=p2p_pb2.SOURCE_STATUS_READY,
                    timeout=self.config.timeout,
                    poll_interval=_BUFFER_POLL_INTERVAL,
                )
                tail_ready_wait_seconds += time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                tail_reset_publish_seconds += time.perf_counter() - phase_started

                phase_started = time.perf_counter()
                rendezvous.wait_for(
                    "inference",
                    count=self.config.inference_world_size,
                    status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                    timeout=self.config.timeout,
                    poll_interval=_BUFFER_POLL_INTERVAL,
                )
                tail_initializing_wait_seconds += time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            dist.barrier()
            tail_barrier_seconds += time.perf_counter() - phase_started

        if self.world.is_master:
            phase_started = time.perf_counter()
            self.rendezvous.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            inference_complete_wait_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self.rendezvous.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            orchestrator_complete_wait_seconds = time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            policy_reset_publish_seconds = time.perf_counter() - phase_started

        final_barrier_started = time.perf_counter()
        dist.barrier()
        final_barrier_seconds = time.perf_counter() - final_barrier_started

        bookkeeping_started = time.perf_counter()
        for run_index in ready:
            self.multi_run_manager.ready_to_update[run_index] = False
        bookkeeping_seconds = time.perf_counter() - bookkeeping_started

        if self.world.is_master:
            sync_seconds = time.perf_counter() - sync_started
            self.logger.info(
                f"Weight update profile v{step} role=trainer rank=0: "
                f"totals(call={time.perf_counter() - call_started:.3f}s, "
                f"lazy_init={lazy_init_seconds:.3f}s, sync={sync_seconds:.3f}s); "
                f"sync_additive_phases: policy_ready_publish={policy_ready_publish_seconds:.3f}s, "
                f"orchestrator_ready_wait={orchestrator_ready_wait_seconds:.3f}s, "
                f"inference_initializing_wait={inference_initializing_wait_seconds:.3f}s, "
                f"recycle_ready_wait={recycle_ready_wait_seconds:.3f}s, "
                f"recycle_reset_publish={recycle_reset_publish_seconds:.3f}s, "
                f"recycle_initializing_wait={recycle_initializing_wait_seconds:.3f}s, "
                f"recycle_barrier_rank0={recycle_barrier_seconds:.3f}s, "
                f"stage_copy_rank0={stage_copy_seconds:.3f}s, "
                f"stage_barrier_rank0={stage_barrier_seconds:.3f}s, "
                f"buffer_ready_publish={buffer_ready_publish_seconds:.3f}s, "
                f"tail_ready_wait={tail_ready_wait_seconds:.3f}s, "
                f"tail_reset_publish={tail_reset_publish_seconds:.3f}s, "
                f"tail_initializing_wait={tail_initializing_wait_seconds:.3f}s, "
                f"tail_barrier_rank0={tail_barrier_seconds:.3f}s, "
                f"inference_complete_wait={inference_complete_wait_seconds:.3f}s, "
                f"orchestrator_complete_wait={orchestrator_complete_wait_seconds:.3f}s, "
                f"policy_reset_publish={policy_reset_publish_seconds:.3f}s, "
                f"final_barrier_rank0={final_barrier_seconds:.3f}s, "
                f"bookkeeping_rank0={bookkeeping_seconds:.3f}s; diagnostic_subsets: "
                f"slowest_group_wall={slowest_group_name}/{slowest_group_seconds:.3f}s, "
                f"slowest_stage_wall_rank0={slowest_stage_name}/{slowest_stage_seconds:.3f}s, "
                f"groups={len(self.groups)}, buffers={self.buffer_count}"
            )
