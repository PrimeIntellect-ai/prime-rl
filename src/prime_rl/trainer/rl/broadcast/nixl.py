"""Trainer-side sharded source for pull-based NIXL weight updates."""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import msgspec
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.weights import gather_weights_on_master
from prime_rl.trainer.world import get_world
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
from prime_rl.weight_transfer.diagnostics import fingerprint_tensor
from prime_rl.weight_transfer.mx import MxChannel
from prime_rl.weight_transfer.nixl import NixlAgent, agent_name, configure_ucx
from prime_rl.weight_transfer.ownership import ShardCandidate, select_shard_owners, select_source_tensors
from prime_rl.weight_transfer.publication import publish_hf_tensors
from prime_rl.weight_transfer.wire import (
    AgentDescriptor,
    DiagnosticSnapshot,
    SyncSignal,
    WeightManifest,
    decode_signal,
    encode_diagnostics,
    encode_manifest,
    encode_signal,
)

_DIAGNOSTIC_SAMPLES_PER_SHARD = 64


def _local_source_signature(shards: list["_LocalShard"]) -> tuple[int, int, float, float, float, float]:
    """Bounded-cost signature of the exact registered source buffers."""

    accumulator = None
    shard_count = 0
    total_numel = 0
    for index, shard in enumerate(shards):
        if shard.buffer is None or shard.buffer.numel() == 0:
            continue
        flat = shard.buffer.detach().reshape(-1)
        stride = max(flat.numel() // _DIAGNOSTIC_SAMPLES_PER_SHARD, 1)
        values = flat[::stride][:_DIAGNOSTIC_SAMPLES_PER_SHARD].to(torch.float64)
        if accumulator is None:
            accumulator = torch.zeros(4, dtype=torch.float64, device=values.device)
        salt = 1 + sum((i + 1) * ord(char) for i, char in enumerate(shard.candidate.name)) % 997
        positions = torch.arange(1, values.numel() + 1, dtype=torch.float64, device=values.device)
        accumulator[0] += values.sum() * salt
        accumulator[1] += values.abs().sum() * salt
        accumulator[2] += values.square().sum() * salt
        accumulator[3] += (values * positions).sum() * (salt + index % 31)
        shard_count += 1
        total_numel += flat.numel()
    if accumulator is None:
        return (0, 0, 0.0, 0.0, 0.0, 0.0)
    return (shard_count, total_numel, *accumulator.cpu().tolist())


@dataclass
class _LocalShard:
    candidate: ShardCandidate
    source: torch.Tensor
    buffer: torch.Tensor | None = None

    @property
    def key(self) -> tuple[int, str, tuple[int, ...], tuple[int, ...]]:
        return (
            self.candidate.rank,
            self.candidate.name,
            self.candidate.global_offset,
            self.candidate.shape,
        )

    def allocate(self, device: torch.device) -> None:
        # FSDP/DTensor state-dict tensors may be views into allocator-owned
        # flattened storage. Register a dedicated, stable cudaMalloc region
        # with the same dtype instead of exposing that aliased storage to
        # long-lived RDMA reads.
        with classic_cuda_alloc():
            self.buffer = torch.empty(self.source.shape, dtype=self.source.dtype, device=device)

    def refresh(self) -> None:
        if self.buffer is None:
            raise RuntimeError(f"source buffer for {self.candidate.name!r} has not been allocated")
        self.buffer.copy_(self.source, non_blocking=True)

    def finalized_candidate(self) -> ShardCandidate:
        if self.buffer is None:
            raise RuntimeError(f"source buffer for {self.candidate.name!r} has not been allocated")
        return ShardCandidate(
            rank=self.candidate.rank,
            name=self.candidate.name,
            dtype=self.buffer.dtype,
            full_shape=self.candidate.full_shape,
            global_offset=self.candidate.global_offset,
            shape=self.candidate.shape,
            address=self.buffer.data_ptr(),
            device_id=self.buffer.device.index or 0,
        )


class NIXLWeightBroadcast(WeightBroadcast):
    def __init__(self, output_dir: Path, config: NIXLWeightBroadcastConfig) -> None:
        super().__init__(output_dir)
        self.config = config
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.epoch = 0
        self.agent: NixlAgent | None = None
        self.local_shards: list[_LocalShard] = []
        self.manifest: WeightManifest | None = None
        self.initialized = False
        self._previous_source_signature: tuple[int, int, float, float, float, float] | None = None

    def _local_candidates(self, model: nn.Module) -> list[_LocalShard]:
        local: list[_LocalShard] = []
        for name, value in model.state_dict().items():
            if isinstance(value, DTensor):
                full_shape = tuple(value.shape)
                tensor = value.to_local().detach()
                local_shape, global_offset = compute_local_shape_and_global_offset(
                    value.shape,
                    value._spec.mesh,
                    value._spec.placements,
                )
                shape = tuple(local_shape)
                offset = tuple(global_offset)
                if not shape or shape[0] == 0:
                    continue
                tensor = tensor[tuple(slice(0, size) for size in shape)]
            else:
                tensor = value.detach()
                full_shape = tuple(tensor.shape)
                if not full_shape or full_shape[0] == 0:
                    continue
                shape = full_shape
                offset = (0,) * len(full_shape)

            candidate = ShardCandidate(
                rank=self.world.rank,
                name=name,
                dtype=tensor.dtype,
                full_shape=full_shape,
                global_offset=offset,
                shape=shape,
                address=tensor.data_ptr(),
                device_id=tensor.device.index or 0,
            )
            local.append(_LocalShard(candidate, tensor))
        return local

    def _all_gather_objects(self, value):
        gathered = [None] * self.world.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    @torch.no_grad()
    def _initialize(self, model: nn.Module) -> None:
        if self.initialized:
            return
        if not isinstance(model, PreTrainedModelPrimeRL):
            raise TypeError("NIXL weight broadcast requires a PrimeRL model with a declarative conversion chain")

        epoch_box = [uuid.uuid4().int & ((1 << 63) - 1) if self.world.is_master else 0]
        dist.broadcast_object_list(epoch_box, src=0)
        self.epoch = epoch_box[0]

        local = self._local_candidates(model)
        candidate_groups = self._all_gather_objects(tuple(shard.candidate for shard in local))
        all_candidates = tuple(candidate for group in candidate_groups for candidate in group)
        selected = select_shard_owners(all_candidates)
        selected_keys = {
            (
                candidate.rank,
                candidate.name,
                candidate.global_offset,
                candidate.shape,
            )
            for candidate in selected
        }
        self.local_shards = [shard for shard in local if shard.key in selected_keys]

        agent_payload = None
        if self.local_shards:
            device = torch.device("cuda", torch.cuda.current_device())
            configure_ucx(device.index or 0)
            self.agent = NixlAgent(agent_name("trainer", self.world.rank, f"{self.config.session_id}-{self.epoch}"))
            registered: set[tuple[int, int]] = set()
            for shard in self.local_shards:
                shard.allocate(device)
                shard.refresh()
                assert shard.buffer is not None
                region = (shard.buffer.data_ptr(), shard.buffer.numel() * shard.buffer.element_size())
                if region not in registered:
                    self.agent.register_tensor(shard.buffer)
                    registered.add(region)
            torch.cuda.synchronize(device)
            agent_payload = (
                self.world.rank,
                AgentDescriptor(self.agent.name, self.agent.metadata()),
                tuple(shard.finalized_candidate() for shard in self.local_shards),
            )

        agent_groups = self._all_gather_objects(agent_payload)
        if self.world.is_master:
            active = sorted(
                (payload for payload in agent_groups if payload is not None), key=lambda payload: payload[0]
            )
            agents = tuple(payload[1] for payload in active)
            rank_to_agent = {payload[0]: index for index, payload in enumerate(active)}
            finalized = tuple(candidate for payload in active for candidate in payload[2])
            sources = select_source_tensors(finalized, rank_to_agent)
            tensors = publish_hf_tensors(model, sources)
            fingerprint = sha256(msgspec.msgpack.encode(tensors)).hexdigest()
            self.manifest = WeightManifest(
                session_id=self.config.session_id,
                epoch=self.epoch,
                model=self.config.model_name,
                fingerprint=fingerprint,
                agents=agents,
                tensors=tensors,
            )
            MxChannel(
                f"{self.config.host}:{self.config.port}",
                self.config.session_id,
                self.config.model_name,
                "trainer",
                "manifest",
                0,
            ).publish(encode_manifest(self.manifest))
            dtype_counts = Counter(tensor.dtype for tensor in tensors)
            self.logger.info(
                "Published {} HF logical tensors over {} trainer NIXL agents; source_dtypes={}",
                len(tensors),
                len(agents),
                dict(sorted(dtype_counts.items())),
            )
        self.initialized = True

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        started = time.perf_counter()
        self._initialize(model)
        for shard in self.local_shards:
            shard.refresh()
        if self.local_shards:
            torch.cuda.synchronize()
        if getattr(self.config, "validate_reload", False):
            signature = _local_source_signature(self.local_shards)
            changed = self._previous_source_signature is not None and signature != self._previous_source_signature
            diagnostics = self._all_gather_objects((self.world.rank, len(self.local_shards), changed, signature))
            self._previous_source_signature = signature
            if self.world.is_master:
                changed_ranks = [rank for rank, _, did_change, _ in diagnostics if did_change]
                self.logger.info(
                    "NIXL trainer source diagnostics v{}: ranks={} changed_ranks={}/{} "
                    "local_shards={} signatures={}",
                    step,
                    len(diagnostics),
                    len(changed_ranks),
                    len(diagnostics),
                    [count for _, count, _, _ in diagnostics],
                    [signature for _, _, _, signature in diagnostics],
                )

            reference_state = gather_weights_on_master(model, is_master=self.world.is_master)
            if self.world.is_master:
                assert isinstance(model, PreTrainedModelPrimeRL)
                model.convert_to_hf(reference_state)
                reference = DiagnosticSnapshot(
                    session_id=self.config.session_id,
                    model=self.config.model_name,
                    step=step,
                    tensors=tuple(
                        fingerprint_tensor(name, tensor) for name, tensor in sorted(reference_state.items())
                    ),
                )
                MxChannel(
                    f"{self.config.host}:{self.config.port}",
                    self.config.session_id,
                    self.config.model_name,
                    "trainer",
                    "diagnostics",
                    0,
                ).publish(encode_diagnostics(reference))
                self.logger.info(
                    "NIXL trainer reference diagnostics v{}: hf_tensors={}", step, len(reference.tensors)
                )
        dist.barrier()

        if self.world.is_master:
            assert self.manifest is not None
            signal = SyncSignal(
                session_id=self.config.session_id,
                epoch=self.epoch,
                step=step,
                phase="trainer_ready",
                rank=0,
                fingerprint=self.manifest.fingerprint,
            )
            channel = MxChannel(
                f"{self.config.host}:{self.config.port}",
                self.config.session_id,
                self.config.model_name,
                "trainer",
                "sync",
                0,
            )
            channel.publish(encode_signal(signal))

            def applied(payload: bytes) -> bool:
                candidate = decode_signal(payload)
                return (
                    candidate.session_id == signal.session_id
                    and candidate.epoch == signal.epoch
                    and candidate.step == signal.step
                    and candidate.phase == "inference_applied"
                    and candidate.fingerprint == signal.fingerprint
                )

            channel.wait_for(
                "inference",
                "sync",
                self.config.inference_world_size,
                applied,
                self.config.timeout,
            )
            for index in self.multi_run_manager.used_idxs:
                self.multi_run_manager.ready_to_update[index] = False
        dist.barrier()
        self.logger.info(f"NIXL weight update v{step} completed in {time.perf_counter() - started:.2f}s")
