"""Serve FP32 FSDP master shards through reusable typed NIXL arenas."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
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
from prime_rl.weight_transfer.cuda_malloc_memory import (
    size_cuda_buffers,
    use_cuda_malloc_pool,
)
from prime_rl.weight_transfer.model_express import ModelExpressSession
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.trainer_tensor_table import (
    TrainerAgent,
    TrainerGroup,
    TrainerShard,
    TrainerTensor,
    TrainerTensorTable,
)

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")
BUFFER_POLL_INTERVAL = 0.01
MAX_STAGING_BUFFER_COUNT = 8


@dataclass
class StagedTensorShard:
    name: str
    global_shape: tuple[int, ...]
    group_index: int
    tensor_offset: int
    source_tensor: torch.Tensor
    wire_dtype: torch.dtype
    staging_tensor: torch.Tensor | None = None

    def assign_staging_tensor(self, arena: torch.Tensor, arena_offset: int) -> None:
        self.staging_tensor = arena.narrow(0, arena_offset, self.source_tensor.numel()).view(
            self.source_tensor.shape
        )

    def copy_to_staging(self) -> None:
        assert self.staging_tensor is not None
        self.staging_tensor.copy_(self.source_tensor)


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
        if self.is_serving_rank:
            set_ucx_env_defaults()
            self.nixl_agent = NixlAgent(make_agent_name("trainer", self.world.rank))
        self.initialized = False
        self.transfer_group_names: list[str] = []
        self.staged_shards: list[StagedTensorShard] = []
        self.staged_shards_by_group: dict[int, list[StagedTensorShard]] = {}
        self.staging_arenas: dict[torch.dtype, torch.Tensor] = {}
        self.staging_buffer_count: int

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
                if value.is_floating_point() and (match := LAYER_RE.search(name)) is not None
            }
        )
        return ["non_layer", *(f"layer.{layer}" for layer in layer_numbers)], {
            layer: group for group, layer in enumerate(layer_numbers, start=1)
        }

    @staticmethod
    def _group_for(name: str, layer_groups: dict[int, int]) -> int:
        match = LAYER_RE.search(name)
        return 0 if match is None else layer_groups[int(match.group(1))]

    def _owned_shards(
        self,
        state_dict: dict[str, torch.Tensor],
        layer_groups: dict[int, int],
        keep_in_fp32: Callable[[str], bool],
    ) -> list[StagedTensorShard]:
        owned: list[StagedTensorShard] = []
        for name, value in state_dict.items():
            if not value.is_floating_point():
                continue
            full_shape = tuple(value.shape)
            group = self._group_for(name, layer_groups)
            wire_dtype = torch.float32 if keep_in_fp32(name) else torch.bfloat16
            if not isinstance(value, DTensor):
                if self.world.is_master:
                    owned.append(
                        StagedTensorShard(
                            name=name,
                            global_shape=full_shape,
                            group_index=group,
                            tensor_offset=0,
                            source_tensor=value.detach(),
                            wire_dtype=wire_dtype,
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
                    owned.append(
                        StagedTensorShard(
                            name=name,
                            global_shape=full_shape,
                            group_index=group,
                            tensor_offset=0,
                            source_tensor=local,
                            wire_dtype=wire_dtype,
                        )
                    )
                continue

            if any(global_offset[1:]) or tuple(local_shape[1:]) != full_shape[1:]:
                raise NotImplementedError(
                    f"NIXL currently requires dim-0 FSDP shards; {name} has "
                    f"local_shape={local_shape}, global_offset={global_offset}"
                )
            if local.numel():
                row_numel = prod(full_shape[1:]) if full_shape else 1
                offset = global_offset[0] * row_numel if full_shape else 0
                owned.append(
                    StagedTensorShard(
                        name=name,
                        global_shape=full_shape,
                        group_index=group,
                        tensor_offset=offset,
                        source_tensor=local,
                        wire_dtype=wire_dtype,
                    )
                )
        return owned

    def _allocate_arena(self, allocated_bytes: int, peak_allocated_bytes: int) -> TrainerArenaStats:
        group_elements = {
            dtype: [0] * len(self.transfer_group_names) for dtype in (torch.bfloat16, torch.float32)
        }
        for shard in self.staged_shards:
            group_elements[shard.wire_dtype][shard.group_index] += shard.source_tensor.numel()
        largest_group_elements = {dtype: max(elements, default=0) for dtype, elements in group_elements.items()}
        largest_group_bytes = sum(elements * dtype.itemsize for dtype, elements in largest_group_elements.items())

        free_before_reclaim = free_after_reclaim = total_bytes = headroom_bytes = 0
        recurring_peak_growth = max(0, peak_allocated_bytes - allocated_bytes)
        has_observed_peak_growth = recurring_peak_growth > 0
        memory_buffer_count = min(len(self.transfer_group_names), MAX_STAGING_BUFFER_COUNT)
        local_buffer_count = min(len(self.transfer_group_names), MAX_STAGING_BUFFER_COUNT)
        if self.is_serving_rank and largest_group_bytes:
            device = self.staged_shards[0].source_tensor.device
            free_before_reclaim, total_bytes = torch.cuda.mem_get_info(device)
            max_buffers = (
                min(len(self.transfer_group_names), MAX_STAGING_BUFFER_COUNT) if has_observed_peak_growth else 1
            )
            if has_observed_peak_growth or free_before_reclaim < largest_group_bytes:
                torch.cuda.empty_cache()
            sizing = size_cuda_buffers(
                largest_group_bytes,
                max_buffers,
                device,
                extra_headroom_bytes=recurring_peak_growth,
            )
            memory_buffer_count = sizing.buffer_count
            free_after_reclaim = sizing.free_bytes
            total_bytes = sizing.total_bytes
            headroom_bytes = sizing.headroom_bytes
            local_buffer_count = memory_buffer_count

        staging_buffer_count = torch.tensor(
            local_buffer_count,
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(staging_buffer_count, op=dist.ReduceOp.MIN)
        self.staging_buffer_count = int(staging_buffer_count.item())

        post_free_bytes = free_after_reclaim
        if self.is_serving_rank and largest_group_bytes:
            device = self.staged_shards[0].source_tensor.device
            with use_cuda_malloc_pool():
                self.staging_arenas = {
                    dtype: torch.empty(
                        self.staging_buffer_count * elements,
                        dtype=dtype,
                        device=device,
                    )
                    for dtype, elements in largest_group_elements.items()
                    if elements
                }
            offsets = {
                dtype: [
                    (group % self.staging_buffer_count) * largest_group_elements[dtype]
                    for group in range(len(self.transfer_group_names))
                ]
                for dtype in self.staging_arenas
            }
            for shard in self.staged_shards:
                dtype_offsets = offsets[shard.wire_dtype]
                shard.assign_staging_tensor(
                    self.staging_arenas[shard.wire_dtype],
                    dtype_offsets[shard.group_index],
                )
                dtype_offsets[shard.group_index] += shard.source_tensor.numel()
            for arena in self.staging_arenas.values():
                self.nixl_agent.register_tensor(arena)
            post_free_bytes, _ = torch.cuda.mem_get_info(device)

        grouped: dict[int, list[StagedTensorShard]] = defaultdict(list)
        for shard in self.staged_shards:
            grouped[shard.group_index].append(shard)
        self.staged_shards_by_group = dict(grouped)
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
            group_total=sum(sum(elements) * dtype.itemsize for dtype, elements in group_elements.items()),
            largest_group=largest_group_bytes,
        )

    def _lazy_init(self, model: nn.Module) -> None:
        if self.initialized:
            return
        allocated_bytes = torch.cuda.memory_allocated() if self.is_serving_rank else 0
        peak_allocated_bytes = torch.cuda.max_memory_allocated() if self.is_serving_rank else 0
        state_dict = model.state_dict()
        self.transfer_group_names, layer_groups = self._transfer_groups(state_dict)
        if self.is_serving_rank:
            keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", lambda _name: False)
            self.staged_shards = self._owned_shards(state_dict, layer_groups, keep_in_fp32)
        buffer_stats = self._allocate_arena(allocated_bytes, peak_allocated_bytes)

        payload = None
        if self.is_serving_rank:
            payload = (
                self.world.rank,
                self.nixl_agent.name,
                self.nixl_agent.get_metadata(),
                torch.cuda.current_device(),
                sum(arena.nbytes for arena in self.staging_arenas.values()),
                buffer_stats,
                [
                    (
                        shard.name,
                        str(shard.wire_dtype).removeprefix("torch."),
                        shard.global_shape,
                        shard.group_index,
                        shard.tensor_offset,
                        shard.source_tensor.numel(),
                        shard.staging_tensor.data_ptr(),
                    )
                    for shard in self.staged_shards
                    if shard.staging_tensor is not None
                ],
            )
        gathered: list | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(payload, gathered, dst=0)

        if self.world.is_master:
            assert gathered is not None
            parts = sorted((part for part in gathered if part is not None), key=lambda part: part[0])
            agents = [
                TrainerAgent(name=name, metadata=metadata, device_id=device_id)
                for _, name, metadata, device_id, _, _, _ in parts
            ]
            tensors: dict[str, TrainerTensor] = {}
            tensor_groups: dict[str, int] = {}
            for agent_index, (_, _, _, _, _, _, rows) in enumerate(parts):
                for name, wire_dtype, shape, group, offset, numel, addr in rows:
                    tensor = tensors.setdefault(
                        name,
                        TrainerTensor(
                            name=name,
                            wire_dtype=wire_dtype,
                            shape=tuple(shape),
                            shards=[],
                        ),
                    )
                    previous_group = tensor_groups.setdefault(name, group)
                    if (
                        tensor.shape != tuple(shape)
                        or tensor.wire_dtype != wire_dtype
                        or previous_group != group
                    ):
                        raise RuntimeError(f"inconsistent trainer metadata for tensor {name!r}")
                    tensor.shards.append(
                        TrainerShard(
                            agent=agent_index,
                            offset=offset,
                            numel=numel,
                            addr=addr,
                        )
                    )

            for tensor in tensors.values():
                tensor.shards.sort(key=lambda shard: shard.offset)

            groups = [
                TrainerGroup(
                    name=group_name,
                    tensors=[tensor for name, tensor in tensors.items() if tensor_groups[name] == group_index],
                )
                for group_index, group_name in enumerate(self.transfer_group_names)
            ]
            table = TrainerTensorTable(
                agents=agents,
                staging_buffer_count=self.staging_buffer_count,
                groups=groups,
            )
            self._validate_table(table)
            server_url = f"{self.config.host}:{self.config.port}"
            client = MxClient(server_url=server_url)
            self.buffer_sessions = []
            for buffer_index in range(self.staging_buffer_count):
                session = ModelExpressSession(
                    client=client,
                    role="trainer",
                    rank=0,
                    session_id=f"{self.config.session_id}:layers:{buffer_index}",
                    worker_id=f"trainer-buffer-{buffer_index}",
                )
                session.publish()
                session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                self.buffer_sessions.append(session)
            self.model_express = ModelExpressSession(
                client=client,
                role="trainer",
                rank=0,
                session_id=self.config.session_id,
                worker_id="trainer-table",
            )
            self.model_express.publish(nixl_metadata=table.encode())
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            total_bytes = sum(
                prod(tensor.shape) * getattr(torch, tensor.wire_dtype).itemsize
                for group in table.groups
                for tensor in group.tensors
            )
            max_arena_bytes = max((part[4] for part in parts), default=0)
            stats = [part[5] for part in parts]
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
                f"NIXL staging ring selected {self.staging_buffer_count} buffers from one-time first-transfer sizing "
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
                f"(policy cap {MAX_STAGING_BUFFER_COUNT}), "
                f"each buffer {buffer_min / gib:.2f}-{group_max / gib:.2f} GiB across ranks, "
                f"arenas up to {max_arena_bytes / gib:.2f} GiB, "
                f"post-allocation free {min(post_free_values) / gib:.2f}-{max(post_free_values) / gib:.2f} GiB, "
                f"{group_total / max(1, len(parts) * len(table.groups)) / 1e6:.1f} MB average logical group payload"
            )
            self.logger.info(
                f"Published {len(tensors)} FP32-master tensors with BF16-default/FP32-exception wire "
                f"precision in {len(table.groups)} groups "
                f"from {len(table.agents)} trainer agents ({total_bytes / 1e9:.2f} GB per update, "
                f"{max_arena_bytes / 1e9:.2f} GB largest local arena, {self.staging_buffer_count} staging buffers)"
            )
        self.initialized = True

    @staticmethod
    def _validate_table(table: TrainerTensorTable) -> None:
        """Fail before publication unless every logical tensor has a valid flat partition."""
        if not 1 <= table.staging_buffer_count <= len(table.groups):
            raise RuntimeError(
                f"invalid staging buffer count {table.staging_buffer_count} for {len(table.groups)} transfer groups"
            )

        group_names: set[str] = set()
        tensor_names: set[str] = set()
        for group in table.groups:
            if group.name in group_names:
                raise RuntimeError(f"duplicate transfer group {group.name!r}")
            group_names.add(group.name)
            for tensor in group.tensors:
                if tensor.name in tensor_names:
                    raise RuntimeError(f"duplicate trainer tensor {tensor.name!r}")
                tensor_names.add(tensor.name)
                if tensor.wire_dtype not in {"bfloat16", "float32"}:
                    raise RuntimeError(f"{tensor.name}: unsupported wire dtype {tensor.wire_dtype!r}")
                if not tensor.shards:
                    raise RuntimeError(f"{tensor.name}: tensor has no trainer shards")

                total_numel = prod(tensor.shape)
                cursor = 0
                for shard in tensor.shards:
                    if not 0 <= shard.agent < len(table.agents):
                        raise RuntimeError(f"{tensor.name}: invalid trainer agent index {shard.agent}")
                    if shard.numel <= 0:
                        raise RuntimeError(f"{tensor.name}: invalid trainer shard size {shard.numel}")
                    if shard.offset != cursor:
                        raise RuntimeError(
                            f"{tensor.name}: trainer shards do not tile flat elements at {cursor}; "
                            f"next range is [{shard.offset}, {shard.offset + shard.numel})"
                        )
                    cursor += shard.numel
                if cursor != total_numel:
                    raise RuntimeError(
                        f"{tensor.name}: trainer shards cover [0, {cursor}), expected [0, {total_numel})"
                    )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        ready_runs = list(self.multi_run_manager.ready_to_update_idxs)
        self._lazy_init(model)
        start = time.perf_counter()

        if self.world.is_master:
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.model_express.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )

        for group, group_name in enumerate(self.transfer_group_names):
            group_start = time.perf_counter()
            buffer_index = group % self.staging_buffer_count
            if group >= self.staging_buffer_count:
                if self.world.is_master:
                    session = self.buffer_sessions[buffer_index]
                    session.wait_for(
                        "inference",
                        count=self.config.inference_world_size,
                        status=p2p_pb2.SOURCE_STATUS_READY,
                        timeout=self.config.timeout,
                        poll_interval=BUFFER_POLL_INTERVAL,
                    )
                    session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                    session.wait_for(
                        "inference",
                        count=self.config.inference_world_size,
                        status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                        timeout=self.config.timeout,
                        poll_interval=BUFFER_POLL_INTERVAL,
                    )
                dist.barrier()

            if self.is_serving_rank:
                for shard in self.staged_shards_by_group.get(group, ()):
                    shard.copy_to_staging()
                torch.cuda.synchronize()
            dist.barrier()
            if self.world.is_master:
                self.buffer_sessions[buffer_index].set_status(p2p_pb2.SOURCE_STATUS_READY)
                self.logger.debug(
                    f"NIXL+ModelExpress policy v{step} group {group_name} staged in buffer {buffer_index} in "
                    f"{time.perf_counter() - group_start:.2f}s"
                )

        first_pending_group = max(0, len(self.transfer_group_names) - self.staging_buffer_count)
        for group in range(first_pending_group, len(self.transfer_group_names)):
            buffer_index = group % self.staging_buffer_count
            if self.world.is_master:
                session = self.buffer_sessions[buffer_index]
                session.wait_for(
                    "inference",
                    count=self.config.inference_world_size,
                    status=p2p_pb2.SOURCE_STATUS_READY,
                    timeout=self.config.timeout,
                    poll_interval=BUFFER_POLL_INTERVAL,
                )
                session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
                session.wait_for(
                    "inference",
                    count=self.config.inference_world_size,
                    status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                    timeout=self.config.timeout,
                    poll_interval=BUFFER_POLL_INTERVAL,
                )
            dist.barrier()

        if self.world.is_master:
            self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.model_express.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
        dist.barrier()
        for run_index in ready_runs:
            self.multi_run_manager.ready_to_update[run_index] = False
        self.logger.info(
            f"NIXL+ModelExpress policy v{step} synchronized in {time.perf_counter() - start:.2f}s"
        )
