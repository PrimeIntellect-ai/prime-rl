"""Serve FP32 FSDP master shards through reusable typed NIXL arenas."""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress.client import MxClient
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.orchestrator.clients import init_nixl_broadcast, update_weights
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transports.weights.base import WeightReceiver, WeightSender
from prime_rl.transports.weights.nixl.agent import (
    NixlAgent,
    NixlPeer,
    group_notification,
    make_agent_name,
    policy_notification,
    set_ucx_env_defaults,
)
from prime_rl.transports.weights.nixl.device_memory import use_registerable_pool
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.transports.weights.nixl.trainer_tensor_table import (
    TrainerAgent,
    TrainerGroup,
    TrainerShard,
    TrainerTensor,
    TrainerTensorTable,
)

LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?=\.|$)")


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


class NIXLWeightSender(WeightSender):
    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__(output_dir, config.timeout)
        self.config = config
        self.parallel_dims = parallel_dims
        if self.is_serving_rank:
            set_ucx_env_defaults(torch.cuda.current_device())
            self.nixl_agent = NixlAgent(make_agent_name("trainer", self.world.rank))
        self.initialized = False
        self.transfer_group_names: list[str] = []
        self.staged_shards: list[StagedTensorShard] = []
        self.staged_shards_by_group: dict[int, list[StagedTensorShard]] = {}
        self.staging_arenas: dict[torch.dtype, torch.Tensor] = {}
        self.staging_buffer_count: int
        self.inference_peers: list[NixlPeer]
        self.orchestrator_peer: NixlPeer
        self.group_generations: list[int]
        self.broadcast_count = 0

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

    def allocate_staging_arenas(self, largest_group_elements: dict[torch.dtype, int]) -> None:
        if not self.is_serving_rank or not any(largest_group_elements.values()):
            return

        device = self.staged_shards[0].source_tensor.device
        with use_registerable_pool(device):
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
        self.staging_buffer_count = min(
            len(self.transfer_group_names),
            2 if self.config.overlap_transfer_and_replay else 1,
        )
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
            self.model_express = ModelExpressSession(
                client=client,
                role="trainer",
                rank=0,
                session_id=self.config.session_id,
                worker_id="trainer-table",
            )
            self.model_express.publish(nixl_metadata=table.encode())
            tensor_count = sum(len(group.tensors) for group in table.groups)
            self.logger.info(
                f"Published {tensor_count} trainer tensors in {len(table.groups)} groups "
                f"from {len(table.agents)} agents with {self.staging_buffer_count} staging buffers"
            )
        self.initialized = True

    def finish_transfer_group(self, group_index: int) -> None:
        if not self.is_serving_rank:
            return
        notification = group_notification(group_index, self.group_generations[group_index])
        self.nixl_agent.wait_for_notification(
            self.inference_peers,
            notification,
            timeout=self.config.timeout,
        )
        self.group_generations[group_index] += 1

    @torch.no_grad()
    def _broadcast(self, model: nn.Module, step: int, step_dir: Path) -> None:
        self.initialize_transfer(model)
        start = time.perf_counter()

        startup = self.broadcast_count == 0
        if self.world.is_master and startup:
            orchestrator_ref = self.model_express.wait_for(
                "orchestrator",
                count=1,
                timeout=self.config.timeout,
            )[0]
            inference_refs = self.model_express.wait_for(
                "inference",
                count=self.config.inference_world_size,
                timeout=self.config.timeout,
            )
            inference_metadata: list[bytes] | None = [
                self.model_express.fetch(ref).nixl_metadata for ref in inference_refs
            ]
            orchestrator_metadata = self.model_express.fetch(orchestrator_ref).nixl_metadata
        else:
            inference_metadata = None

        if startup:
            objects = [inference_metadata]
            dist.broadcast_object_list(objects, src=0)
            if self.is_serving_rank:
                self.inference_peers = []
                for metadata in cast(list[bytes], objects[0]):
                    peer = self.nixl_agent.add_remote_agent(metadata)
                    self.nixl_agent.make_connection(peer)
                    self.inference_peers.append(peer)
                self.group_generations = [0] * len(self.transfer_group_names)
            if self.world.is_master:
                self.orchestrator_peer = self.nixl_agent.add_remote_agent(orchestrator_metadata)
                self.nixl_agent.make_connection(self.orchestrator_peer)

        if self.world.is_master:
            self.nixl_agent.send_notification(
                self.orchestrator_peer,
                policy_notification(step, "ready"),
            )

        for group, group_name in enumerate(self.transfer_group_names):
            group_start = time.perf_counter()
            buffer_index = group % self.staging_buffer_count
            if group >= self.staging_buffer_count:
                self.finish_transfer_group(group - self.staging_buffer_count)

            if self.is_serving_rank:
                for shard in self.staged_shards_by_group.get(group, ()):
                    shard.copy_to_staging()
                torch.cuda.synchronize()
                notification = group_notification(group, self.group_generations[group])
                for peer in self.inference_peers:
                    self.nixl_agent.send_notification(peer, notification)
            if self.world.is_master:
                self.logger.debug(
                    f"NIXL+ModelExpress policy v{step} group {group_name} staged in buffer {buffer_index} in "
                    f"{time.perf_counter() - group_start:.2f}s"
                )

        first_pending_group = max(0, len(self.transfer_group_names) - self.staging_buffer_count)
        for group in range(first_pending_group, len(self.transfer_group_names)):
            self.finish_transfer_group(group)

        if self.world.is_master:
            self.nixl_agent.wait_for_notification(
                [self.orchestrator_peer],
                policy_notification(step, "complete"),
                timeout=self.config.timeout,
            )
        dist.barrier()
        self.broadcast_count += 1
        self.logger.info(f"NIXL+ModelExpress policy v{step} synchronized in {time.perf_counter() - start:.2f}s")


class NIXLWeightReceiver(WeightReceiver):
    """Drives NIXL discovery and policy synchronization for the orchestrator."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        set_ucx_env_defaults(0)
        self.nixl_agent = NixlAgent(make_agent_name("orchestrator", 0))
        self.trainer_peer: NixlPeer | None = None
        self.model_express = ModelExpressSession(
            client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
            role="orchestrator",
            rank=0,
            session_id=self.config.session_id,
            worker_id="orchestrator",
        )

    async def initialize(self) -> None:
        await init_nixl_broadcast(
            self.admin_clients,
            self.config.host,
            self.config.port,
            self.config.timeout,
            self.config.inference_world_size,
            self.config.session_id,
        )
        self.model_express.publish(nixl_metadata=self.nixl_agent.get_metadata())

    async def receive(self, step: int) -> None:
        self._ack(step)
        if self.trainer_peer is None:
            trainer_refs = await asyncio.to_thread(
                self.model_express.wait_for,
                "trainer",
                count=1,
                timeout=self.config.timeout,
            )
            trainer_worker = await asyncio.to_thread(self.model_express.fetch, trainer_refs[0])
            trainer_table = TrainerTensorTable.decode(trainer_worker.nixl_metadata)
            self.trainer_peer = self.nixl_agent.add_remote_agent(trainer_table.agents[0].metadata)
            self.nixl_agent.make_connection(self.trainer_peer)

        trainer_peer = self.trainer_peer
        await asyncio.to_thread(
            self.nixl_agent.wait_for_notification,
            [trainer_peer],
            policy_notification(step, "ready"),
            timeout=self.config.timeout,
        )
        await update_weights(self.admin_clients, None, step=step)
        self.nixl_agent.send_notification(
            trainer_peer,
            policy_notification(step, "complete"),
        )
