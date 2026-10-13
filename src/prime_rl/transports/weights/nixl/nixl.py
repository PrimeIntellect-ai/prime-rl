"""Serve FP32 FSDP master shards through reusable typed NIXL arenas."""

from __future__ import annotations

import asyncio
import json
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
from httpx import AsyncClient
from modelexpress.client import MxClient
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.shared import NIXLWeightBroadcastConfig
from prime_rl.orchestrator.clients import ADMIN_TIMEOUT_S, init_nixl_broadcast
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.world import wait_for_master
from prime_rl.transports.weights.base import WeightReceiver, WeightSender
from prime_rl.transports.weights.nixl.agent import (
    NixlAgent,
    NixlPeer,
    drop_marker,
    group_notification,
    make_agent_name,
    policy_notification,
    set_ucx_env_defaults,
)
from prime_rl.transports.weights.nixl.cuda_malloc_memory import use_cuda_malloc_pool
from prime_rl.transports.weights.nixl.model_express import ModelExpressSession
from prime_rl.transports.weights.nixl.trainer_tensor_table import (
    TrainerAgent,
    TrainerGroup,
    TrainerShard,
    TrainerTensor,
    TrainerTensorTable,
)
from prime_rl.utils.pathing import get_broadcast_dir

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
        # Inference rank -> its peer and the metadata it was added from
        self.inference_peers: dict[int, NixlPeer] = {}
        self.inference_metadata: dict[int, bytes] = {}
        self.orchestrator_peer: NixlPeer
        # Last inference group the orchestrator published after evicting or re-adding engines
        self.generation = 0
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

    def finish_transfer_group(self, group_index: int, step: int, step_dir: Path) -> None:
        if not self.is_serving_rank:
            return
        self.nixl_agent.wait_for_notification(
            list(self.inference_peers.values()),
            group_notification(group_index, step),
            timeout=self.config.timeout,
            skip=lambda: self.dropped_peers(step_dir),
        )

    def dropped_peers(self, step_dir: Path) -> set[str]:
        """Inference peers the orchestrator dropped from this transfer because their engine died or hung."""
        ranks = {int(marker.name.removeprefix(".drop_")) for marker in step_dir.glob(".drop_*")}
        return {peer.remote_agent_name for rank, peer in self.inference_peers.items() if rank in ranks}

    def discover_inference(self, startup: bool) -> dict[int, bytes] | None:
        """NIXL metadata of the inference ranks taking part: at startup, and whenever the
        orchestrator published a new group after evicting or re-adding engines. It publishes the
        group before acknowledging the version, and a rejoining engine republishes its metadata
        before that. None when nothing changed."""
        group_file = nixl_group_file(get_broadcast_dir(self.output_dir))
        group = json.loads(group_file.read_text()) if group_file.exists() else None
        if group is not None and group["generation"] > self.generation:
            self.generation = group["generation"]
            self.logger.info(
                f"Updating the NIXL inference peers (generation {self.generation}): {len(group['ranks'])} ranks"
            )
            refs = {ref.worker_rank: ref for ref in self.model_express.list_sources("inference")}
            return {rank: self.model_express.fetch(refs[rank]).nixl_metadata for rank in group["ranks"]}
        if not startup:
            return None
        refs = self.model_express.wait_for(
            "inference", count=self.config.inference_world_size, timeout=self.config.timeout
        )
        return {rank: self.model_express.fetch(ref).nixl_metadata for rank, ref in enumerate(refs)}

    def connect_inference_peers(self, metadata: dict[int, bytes]) -> None:
        """Drop the peers of evicted ranks and (re)connect ranks whose metadata is new, e.g. a
        relaunched engine's."""
        for rank in self.inference_peers.keys() - metadata.keys():
            self.nixl_agent.remove_remote_agent(self.inference_peers.pop(rank))
            del self.inference_metadata[rank]
        for rank, rank_metadata in metadata.items():
            if self.inference_metadata.get(rank) == rank_metadata:
                continue
            if rank in self.inference_peers:
                self.nixl_agent.remove_remote_agent(self.inference_peers[rank])
            peer = self.nixl_agent.add_remote_agent(rank_metadata)
            self.nixl_agent.make_connection(peer)
            self.inference_peers[rank] = peer
            self.inference_metadata[rank] = rank_metadata

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
            orchestrator_metadata = self.model_express.fetch(orchestrator_ref).nixl_metadata
            self.orchestrator_peer = self.nixl_agent.add_remote_agent(orchestrator_metadata)
            self.nixl_agent.make_connection(self.orchestrator_peer)
        inference_metadata = self.discover_inference(startup) if self.world.is_master else None

        if startup:
            # The master's ModelExpress discovery can outlast the NCCL timeout; hold the others on the store.
            wait_for_master(f"weight_broadcast/nixl_discovery/{step}", self.config.timeout)
        objects = [inference_metadata]
        dist.broadcast_object_list(objects, src=0)
        if self.is_serving_rank and objects[0] is not None:
            self.connect_inference_peers(cast(dict[int, bytes], objects[0]))

        unreachable: set[int] = set()
        for group, group_name in enumerate(self.transfer_group_names):
            group_start = time.perf_counter()
            buffer_index = group % self.staging_buffer_count
            if group >= self.staging_buffer_count:
                self.finish_transfer_group(group - self.staging_buffer_count, step, step_dir)

            if self.is_serving_rank:
                for shard in self.staged_shards_by_group.get(group, ()):
                    shard.copy_to_staging()
                torch.cuda.synchronize()
                notification = group_notification(group, step)
                for rank, peer in self.inference_peers.items():
                    try:
                        self.nixl_agent.send_notification(peer, notification)
                    except Exception as error:
                        # A dead engine: the orchestrator drops it from the transfer
                        if rank not in unreachable:
                            unreachable.add(rank)
                            self.logger.warning(f"Could not notify inference rank {rank}: {error!r}")
            if self.world.is_master:
                self.logger.debug(
                    f"NIXL+ModelExpress policy v{step} group {group_name} staged in buffer {buffer_index} in "
                    f"{time.perf_counter() - group_start:.2f}s"
                )

        first_pending_group = max(0, len(self.transfer_group_names) - self.staging_buffer_count)
        for group in range(first_pending_group, len(self.transfer_group_names)):
            self.finish_transfer_group(group, step, step_dir)

        if self.world.is_master:
            self.nixl_agent.wait_for_notification(
                [self.orchestrator_peer],
                policy_notification(step, "complete"),
                timeout=self.config.timeout,
            )
        wait_for_master(f"weight_broadcast/nixl_complete/{step}", self.config.timeout)
        dist.barrier()
        self.broadcast_count += 1
        self.logger.info(f"NIXL+ModelExpress policy v{step} synchronized in {time.perf_counter() - start:.2f}s")


def nixl_group_file(broadcast_dir: Path) -> Path:
    """Where the orchestrator publishes the inference ranks taking part after evicting or re-adding an engine."""
    return broadcast_dir / "nixl_group.json"


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
        # A group left by a previous run must not steer the new trainer
        nixl_group_file(self.broadcast_dir).unlink(missing_ok=True)
        await init_nixl_broadcast(
            self.admin_plane,
            self.config.host,
            self.config.port,
            self.config.timeout,
            self.config.inference_world_size,
            self.config.session_id,
        )
        # An engine keeps its inference ranks (its ModelExpress identities) when it is evicted and rejoins
        workers = self.config.inference_world_size // len(self.admin_plane.clients)
        self.ranks = {
            client: range(index * workers, (index + 1) * workers)
            for index, client in enumerate(self.admin_plane.clients)
        }
        self.model_express.publish(nixl_metadata=self.nixl_agent.get_metadata())

    async def receive(self, step: int) -> None:
        await self.admin_plane.update_weights(
            self.step_dir(step),
            transport="nixl",
            step=step,
            on_paused=lambda: self._ack(step),
            on_membership_change=self.on_membership_change,
            abort=lambda client: self.drop(client, step),
        )
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
        self.nixl_agent.send_notification(
            self.trainer_peer,
            policy_notification(step, "complete"),
        )

    def drop(self, client: AsyncClient, step: int) -> None:
        """Take a dead or hung engine out of v{step}'s transfer: the trainer stops waiting for its
        ranks and the other engines finish their pulls."""
        for rank in self.ranks[client]:
            drop_marker(self.step_dir(step), rank).touch()

    async def on_membership_change(self, members: list[AsyncClient], generation: int) -> None:
        """Set up the rejoining engines, then publish the inference ranks taking part for the
        trainer, which reads them after this version's acknowledgement. A rejoining engine may
        have been relaunched: ``/init_broadcaster`` gives it a fresh NIXL agent, published to
        ModelExpress under its old ranks. One that fails it is evicted again."""

        async def init_broadcaster(client: AsyncClient) -> None:
            await self.admin_plane._each(
                [client],
                "/init_broadcaster",
                timeout_s=max(ADMIN_TIMEOUT_S, self.config.timeout),
                json={
                    "host": self.config.host,
                    "port": self.config.port,
                    "rank_offset": self.ranks[client].start,
                    "inference_world_size": self.config.inference_world_size,
                    "timeout": self.config.timeout,
                    "session_id": self.config.session_id,
                },
            )

        await asyncio.gather(*(init_broadcaster(client) for client in members if client in self.admin_plane.evicted))
        members = self.admin_plane._standing(members)
        ranks = [rank for client in members for rank in self.ranks[client]]
        self.logger.info(f"Updating the NIXL inference group (generation {generation}): {len(members)} engines")
        group_file = nixl_group_file(self.broadcast_dir)
        tmp = group_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"generation": generation, "ranks": ranks}))
        tmp.replace(group_file)
