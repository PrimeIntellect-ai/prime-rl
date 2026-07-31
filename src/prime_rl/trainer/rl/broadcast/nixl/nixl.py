"""Serve FP32 FSDP master shards through reusable typed NIXL arenas."""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import cast

import httpx
import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.nixl.agent import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.trainer.rl.broadcast.nixl.cuda_malloc_memory import (
    size_cuda_buffers,
    use_cuda_malloc_pool,
)
from prime_rl.trainer.rl.broadcast.nixl.model_express import ModelExpressSession
from prime_rl.trainer.rl.broadcast.nixl.trainer_tensor_table import (
    TrainerAgent,
    TrainerGroup,
    TrainerShard,
    TrainerTensor,
    TrainerTensorTable,
)
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")
BUFFER_POLL_INTERVAL = 0.01
MAX_STAGING_BUFFER_COUNT = 8
WORKER_DISCOVERY_POLL_INTERVAL = 1.0


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
        self.staging_tensor = arena.narrow(0, arena_offset, self.source_tensor.numel()).view(self.source_tensor.shape)

    def copy_to_staging(self) -> None:
        assert self.staging_tensor is not None
        self.staging_tensor.copy_(self.source_tensor)


@dataclass(frozen=True)
class TransferGroupIndex:
    group_names: list[str]
    layer_to_group: dict[int, int]


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
        self.inference_worker_rank_offsets: dict[str, int] = {}
        self.ranks_per_inference_worker: int | None = None
        self.inference_rank_count = config.inference_world_size

    @property
    def is_serving_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    @staticmethod
    def build_transfer_group_index(state_dict: dict[str, torch.Tensor]) -> TransferGroupIndex:
        layer_numbers = sorted(
            {
                int(match.group(1))
                for name, value in state_dict.items()
                if value.is_floating_point() and (match := LAYER_RE.search(name)) is not None
            }
        )
        return TransferGroupIndex(
            group_names=["non_layer", *(f"layer.{layer}" for layer in layer_numbers)],
            layer_to_group={layer: group for group, layer in enumerate(layer_numbers, start=1)},
        )

    @staticmethod
    def find_transfer_group_index(tensor_name: str, transfer_groups: TransferGroupIndex) -> int:
        match = LAYER_RE.search(tensor_name)
        return 0 if match is None else transfer_groups.layer_to_group[int(match.group(1))]

    def collect_local_tensor_shards(
        self,
        state_dict: dict[str, torch.Tensor],
        transfer_groups: TransferGroupIndex,
        keep_in_fp32: Callable[[str], bool],
    ) -> list[StagedTensorShard]:
        local_shards: list[StagedTensorShard] = []
        for name, value in state_dict.items():
            # Non-floating state is not part of model weight transfer.
            if not value.is_floating_point():
                continue
            full_shape = tuple(value.shape)
            group_index = self.find_transfer_group_index(name, transfer_groups)
            wire_dtype = torch.float32 if keep_in_fp32(name) else torch.bfloat16

            # Unsharded tensors are identical on every rank, so rank 0 serves the only copy.
            if not isinstance(value, DTensor):
                if self.world.is_master:
                    local_shards.append(
                        StagedTensorShard(
                            name=name,
                            global_shape=full_shape,
                            group_index=group_index,
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

            # Replicated DTensors are identical on every rank, so rank 0 serves the only copy.
            if all(placement.is_replicate() for placement in placements):
                if self.world.is_master:
                    local_shards.append(
                        StagedTensorShard(
                            name=name,
                            global_shape=full_shape,
                            group_index=group_index,
                            tensor_offset=0,
                            source_tensor=local,
                            wire_dtype=wire_dtype,
                        )
                    )
                continue

            # FSDP DTensors contribute this rank's contiguous shard along tensor dimension 0.
            if local.numel():
                row_numel = prod(full_shape[1:]) if full_shape else 1
                offset = global_offset[0] * row_numel if full_shape else 0
                local_shards.append(
                    StagedTensorShard(
                        name=name,
                        global_shape=full_shape,
                        group_index=group_index,
                        tensor_offset=offset,
                        source_tensor=local,
                        wire_dtype=wire_dtype,
                    )
                )
        return local_shards

    def choose_staging_buffer_count(self, largest_group_bytes: int) -> int:
        local_buffer_count = min(len(self.transfer_group_names), MAX_STAGING_BUFFER_COUNT)
        if self.is_serving_rank and largest_group_bytes:
            device = self.staged_shards[0].source_tensor.device
            allocated_bytes = torch.cuda.memory_allocated()
            peak_growth_bytes = max(0, torch.cuda.max_memory_allocated() - allocated_bytes)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            max_buffers = local_buffer_count if peak_growth_bytes else 1
            if peak_growth_bytes or free_bytes < largest_group_bytes:
                torch.cuda.empty_cache()
            local_buffer_count = size_cuda_buffers(
                largest_group_bytes,
                max_buffers,
                device,
                extra_headroom_bytes=peak_growth_bytes,
            )

        staging_buffer_count = torch.tensor(
            local_buffer_count,
            dtype=torch.int64,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(staging_buffer_count, op=dist.ReduceOp.MIN)
        return int(staging_buffer_count.item())

    def allocate_staging_arenas(self, largest_group_elements: dict[torch.dtype, int]) -> None:
        if not self.is_serving_rank or not any(largest_group_elements.values()):
            return

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
            group_offsets = offsets[shard.wire_dtype]
            shard.assign_staging_tensor(
                self.staging_arenas[shard.wire_dtype],
                group_offsets[shard.group_index],
            )
            group_offsets[shard.group_index] += shard.source_tensor.numel()

        for arena in self.staging_arenas.values():
            self.nixl_agent.register_tensor(arena)

    def prepare_staging_buffers(self) -> None:
        group_elements = {dtype: [0] * len(self.transfer_group_names) for dtype in (torch.bfloat16, torch.float32)}
        for shard in self.staged_shards:
            group_elements[shard.wire_dtype][shard.group_index] += shard.source_tensor.numel()
        largest_group_elements = {dtype: max(elements, default=0) for dtype, elements in group_elements.items()}
        largest_group_bytes = sum(elements * dtype.itemsize for dtype, elements in largest_group_elements.items())
        self.staging_buffer_count = self.choose_staging_buffer_count(largest_group_bytes)
        self.allocate_staging_arenas(largest_group_elements)

        grouped: dict[int, list[StagedTensorShard]] = defaultdict(list)
        for shard in self.staged_shards:
            grouped[shard.group_index].append(shard)
        self.staged_shards_by_group = dict(grouped)

    def build_local_trainer_table_fragment(self) -> TrainerTensorTable:
        tensors_by_group: list[dict[str, TrainerTensor]] = [{} for _ in self.transfer_group_names]
        for shard in self.staged_shards:
            tensors = tensors_by_group[shard.group_index]
            tensor = tensors.setdefault(
                shard.name,
                TrainerTensor(
                    name=shard.name,
                    wire_dtype=str(shard.wire_dtype).removeprefix("torch."),
                    shape=shard.global_shape,
                    shards=[],
                ),
            )
            tensor.shards.append(
                TrainerShard(
                    agent=0,
                    offset=shard.tensor_offset,
                    numel=shard.source_tensor.numel(),
                    addr=cast(torch.Tensor, shard.staging_tensor).data_ptr(),
                )
            )

        return TrainerTensorTable(
            agents=[
                TrainerAgent(
                    name=self.nixl_agent.name,
                    metadata=self.nixl_agent.get_metadata(),
                    device_id=torch.cuda.current_device(),
                )
            ],
            staging_buffer_count=self.staging_buffer_count,
            groups=[
                TrainerGroup(name=group_name, tensors=list(tensors.values()))
                for group_name, tensors in zip(self.transfer_group_names, tensors_by_group)
            ],
        )

    def gather_trainer_table_fragments(self) -> list[bytes] | None:
        table_fragment = self.build_local_trainer_table_fragment().encode() if self.is_serving_rank else None
        gathered: list[bytes | None] | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(table_fragment, gathered, dst=0)
        if gathered is None:
            return None
        return [fragment for fragment in gathered if fragment is not None]

    def merge_trainer_table_fragments(self, table_fragments: list[bytes]) -> TrainerTensorTable:
        agents: list[TrainerAgent] = []
        tensors_by_group: list[dict[str, TrainerTensor]] = [{} for _ in self.transfer_group_names]
        for agent_index, encoded_fragment in enumerate(table_fragments):
            fragment = TrainerTensorTable.decode(encoded_fragment)
            agents.append(fragment.agents[0])
            for group_index, group in enumerate(fragment.groups):
                tensors = tensors_by_group[group_index]
                for fragment_tensor in group.tensors:
                    tensor = tensors.setdefault(
                        fragment_tensor.name,
                        TrainerTensor(
                            name=fragment_tensor.name,
                            wire_dtype=fragment_tensor.wire_dtype,
                            shape=fragment_tensor.shape,
                            shards=[],
                        ),
                    )
                    tensor.shards.extend(
                        TrainerShard(
                            agent=agent_index,
                            offset=shard.offset,
                            numel=shard.numel,
                            addr=shard.addr,
                        )
                        for shard in fragment_tensor.shards
                    )

        for tensors in tensors_by_group:
            for tensor in tensors.values():
                tensor.shards.sort(key=lambda shard: shard.offset)

        return TrainerTensorTable(
            agents=agents,
            staging_buffer_count=self.staging_buffer_count,
            groups=[
                TrainerGroup(name=group_name, tensors=list(tensors.values()))
                for group_name, tensors in zip(self.transfer_group_names, tensors_by_group)
            ],
        )

    @staticmethod
    def worker_admin_url(url: str) -> str:
        return re.sub(r"@\d+$", "", url.rstrip("/")).removesuffix("/v1")

    def discover_inference_workers(self) -> list[str]:
        router_url = self.worker_admin_url(cast(str, self.config.router_url))
        deadline = time.monotonic() + self.config.timeout
        last_error: httpx.HTTPError | None = None
        while True:
            try:
                response = httpx.get(f"{router_url}/workers", timeout=10.0)
                response.raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.is_client_error:
                    raise
                last_error = error
            except httpx.TransportError as error:
                last_error = error
            else:
                workers = response.json()["workers"]
                if self.ranks_per_inference_worker is None:
                    initial_workers = [worker for worker in workers if worker["is_usable"]]
                    healthy_workers = (
                        initial_workers
                        if initial_workers and all(worker["is_healthy"] for worker in initial_workers)
                        else []
                    )
                else:
                    healthy_workers = [worker for worker in workers if worker["is_healthy"]]
                urls = sorted({self.worker_admin_url(worker["url"]) for worker in healthy_workers})
                if urls:
                    return urls

            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for inference workers at {router_url}/workers") from last_error
            time.sleep(WORKER_DISCOVERY_POLL_INTERVAL)

    def initialize_inference_workers(self) -> None:
        worker_urls = self.discover_inference_workers()
        if self.ranks_per_inference_worker is None:
            self.ranks_per_inference_worker = self.config.inference_world_size // len(worker_urls)

        new_worker_urls = [url for url in worker_urls if url not in self.inference_worker_rank_offsets]
        next_rank_offset = len(self.inference_worker_rank_offsets) * self.ranks_per_inference_worker
        new_workers = {
            worker_url: next_rank_offset + index * self.ranks_per_inference_worker
            for index, worker_url in enumerate(new_worker_urls)
        }
        current_rank_count = len(worker_urls) * self.ranks_per_inference_worker
        if not new_workers:
            self.inference_rank_count = current_rank_count
            return

        timeout = httpx.Timeout(connect=10.0, read=self.config.timeout, write=60.0, pool=10.0)
        deadline = time.monotonic() + self.config.timeout
        pending = dict(new_workers)
        last_errors: dict[str, httpx.HTTPError] = {}
        with httpx.Client(timeout=timeout) as client:
            while pending:
                for worker_url, rank_offset in list(pending.items()):
                    try:
                        response = client.post(
                            f"{worker_url}/init_broadcaster",
                            json={
                                "host": self.config.host,
                                "port": self.config.port,
                                "rank_offset": rank_offset,
                                "inference_world_size": current_rank_count,
                                "timeout": self.config.timeout,
                                "quantize_in_weight_transfer": False,
                                "session_id": self.config.session_id,
                            },
                        )
                        response.raise_for_status()
                    except httpx.HTTPStatusError as error:
                        if error.response.is_client_error:
                            raise
                        last_errors[worker_url] = error
                    except httpx.TransportError as error:
                        last_errors[worker_url] = error
                    else:
                        del pending[worker_url]

                if pending:
                    if time.monotonic() >= deadline:
                        pending_urls = ", ".join(pending)
                        raise TimeoutError(f"timed out initializing inference workers: {pending_urls}") from next(
                            iter(last_errors.values()), None
                        )
                    time.sleep(WORKER_DISCOVERY_POLL_INTERVAL)
        self.inference_worker_rank_offsets.update(new_workers)
        self.inference_rank_count = current_rank_count
        self.logger.info(
            f"Initialized NIXL transfer on {len(new_workers)} new inference servers; "
            f"{len(worker_urls)} servers and {self.inference_rank_count} ranks now participate"
        )

    def initialize_transfer(self, model: nn.Module) -> None:
        if self.initialized:
            return
        model = cast(PreTrainedModelPrimeRL, model)
        state_dict = model.state_dict()
        transfer_groups = self.build_transfer_group_index(state_dict)
        self.transfer_group_names = transfer_groups.group_names
        if self.is_serving_rank:
            self.staged_shards = self.collect_local_tensor_shards(
                state_dict,
                transfer_groups,
                model.keep_in_fp32_for_weight_transfer,
            )
        self.prepare_staging_buffers()
        table_fragments = self.gather_trainer_table_fragments()

        if table_fragments is not None:
            table = self.merge_trainer_table_fragments(table_fragments)
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
            tensor_count = sum(len(group.tensors) for group in table.groups)
            self.logger.info(
                f"Published {tensor_count} trainer tensors in {len(table.groups)} groups "
                f"from {len(table.agents)} agents with {self.staging_buffer_count} staging buffers"
            )
        self.initialized = True

    def finish_staging_buffer_transfer(self, buffer_index: int) -> None:
        if self.world.is_master:
            session = self.buffer_sessions[buffer_index]
            session.wait_for(
                "inference",
                count=self.inference_rank_count,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
                poll_interval=BUFFER_POLL_INTERVAL,
            )
            session.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            session.wait_for(
                "inference",
                count=self.inference_rank_count,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
                poll_interval=BUFFER_POLL_INTERVAL,
            )
        dist.barrier()

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        ready_runs = list(self.multi_run_manager.ready_to_update_idxs)
        self.initialize_transfer(model)
        start = time.perf_counter()

        if self.world.is_master:
            if self.config.router_url is not None:
                self.initialize_inference_workers()
            self.model_express.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.model_express.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
            self.model_express.wait_for(
                "inference",
                count=self.inference_rank_count,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )

        for group, group_name in enumerate(self.transfer_group_names):
            group_start = time.perf_counter()
            buffer_index = group % self.staging_buffer_count
            if group >= self.staging_buffer_count:
                self.finish_staging_buffer_transfer(buffer_index)

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
            self.finish_staging_buffer_transfer(buffer_index)

        if self.world.is_master:
            self.model_express.wait_for(
                "inference",
                count=self.inference_rank_count,
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
        self.logger.info(f"NIXL+ModelExpress policy v{step} synchronized in {time.perf_counter() - start:.2f}s")
