"""Serve FP32 FSDP master shards through persistent BF16 NIXL buffers."""

from __future__ import annotations

import re
import time
from math import prod
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
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
_EXPERT_RE = re.compile(r"\.mlp\.experts\.w[123]$")


class _OwnedShard:
    def __init__(self, name: str, full_shape: tuple[int, ...], row_start: int, source: torch.Tensor):
        if source.dtype != _MASTER_DTYPE:
            raise TypeError(f"NIXL source {name!r} must be an FP32 master shard, got {source.dtype}")
        if not source.is_contiguous():
            raise ValueError(f"NIXL source {name!r} must be contiguous")
        self.name = name
        self.full_shape = full_shape
        self.row_start = row_start
        self.source = source
        self.num_rows = source.shape[0] if source.ndim else 1
        self.row_numel = source[0].numel() if source.ndim and self.num_rows else 1
        self.buffer: torch.Tensor | None = None

    def allocate(self) -> None:
        with classic_cuda_alloc():
            self.buffer = torch.empty(self.source.shape, dtype=_WIRE_DTYPE, device=self.source.device)

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
        self._initialized = False
        self._shards: list[_OwnedShard] = []

    @property
    def is_serving_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    def _owned_shards(self, model: nn.Module) -> list[_OwnedShard]:
        if self.parallel_dims.ep_enabled:
            ep_mesh = self.parallel_dims.get_mesh("ep")
            fsdp_mod_ep = self.parallel_dims.get_mesh("dp_shard_mod_ep")
            ep_rank = ep_mesh.get_local_rank()
            fsdp_mod_ep_size = fsdp_mod_ep.size()
            fsdp_mod_ep_rank = fsdp_mod_ep.get_local_rank()
        else:
            ep_rank = fsdp_mod_ep_rank = 0
            fsdp_mod_ep_size = 1

        owned: list[_OwnedShard] = []
        for name, value in model.state_dict().items():
            if not value.is_floating_point():
                continue
            full_shape = tuple(value.shape)
            if not isinstance(value, DTensor):
                if self.world.is_master:
                    owned.append(_OwnedShard(name, full_shape, 0, value.detach()))
                continue

            local = value.to_local().detach()
            if _EXPERT_RE.search(name):
                num_local = local.shape[0]
                experts_per_ep = num_local * fsdp_mod_ep_size
                row_start = ep_rank * experts_per_ep + fsdp_mod_ep_rank * num_local
                if num_local:
                    owned.append(_OwnedShard(name, full_shape, row_start, local))
                continue

            placements = value._spec.placements
            if all(placement.is_replicate() for placement in placements):
                if self.world.is_master:
                    owned.append(_OwnedShard(name, full_shape, 0, local))
                continue

            local_shape, global_offset = compute_local_shape_and_global_offset(
                value.shape, value._spec.mesh, placements
            )
            if any(global_offset[1:]) or tuple(local_shape[1:]) != full_shape[1:]:
                raise NotImplementedError(
                    f"NIXL currently requires dim-0 FSDP shards; {name} has "
                    f"local_shape={local_shape}, global_offset={global_offset}"
                )
            if local_shape[0]:
                owned.append(_OwnedShard(name, full_shape, global_offset[0], local[: local_shape[0]]))
        return owned

    def _lazy_init(self, model: nn.Module) -> None:
        if self._initialized:
            return
        if self.is_serving_rank:
            self._shards = self._owned_shards(model)
            for shard in self._shards:
                shard.allocate()
                shard.refresh()
                assert shard.buffer is not None
                self.nixl_agent.register_tensor(shard.buffer)
            torch.cuda.synchronize()

        payload = None
        if self.is_serving_rank:
            payload = (
                self.world.rank,
                self.nixl_agent.name,
                self.nixl_agent.get_metadata(),
                [
                    (
                        shard.name,
                        shard.full_shape,
                        shard.row_start,
                        shard.num_rows,
                        shard.buffer.data_ptr(),
                        shard.row_numel * shard.buffer.element_size(),
                        shard.buffer.device.index,
                    )
                    for shard in self._shards
                    if shard.buffer is not None
                ],
            )
        gathered: list | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(payload, gathered, dst=0)

        if self.world.is_master:
            assert gathered is not None
            parts = sorted((part for part in gathered if part is not None), key=lambda part: part[0])
            agents = [TrainerAgent(name=name, metadata=metadata) for _, name, metadata, _ in parts]
            tensors: dict[str, TrainerTensor] = {}
            for agent_index, (_, _, _, rows) in enumerate(parts):
                for name, shape, row_start, num_rows, addr, row_bytes, device_id in rows:
                    tensor = tensors.setdefault(
                        name,
                        TrainerTensor(
                            name=name,
                            master_dtype="float32",
                            dtype="bfloat16",
                            shape=tuple(shape),
                            shards=[],
                        ),
                    )
                    if tensor.shape != tuple(shape) or tensor.master_dtype != "float32" or tensor.dtype != "bfloat16":
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
            table = TrainerTable(agents=agents, tensors=list(tensors.values()))
            self._validate_table(table)
            from modelexpress.client import MxClient

            self.rendezvous = MxRendezvous(
                client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
                role="trainer",
                rank=0,
                peer_world_size=self.config.inference_world_size,
                session_id=self.config.session_id,
                worker_id="trainer-table",
            )
            self.rendezvous.publish(nixl_metadata=encode_table(table))
            total_bytes = sum(shard.num_rows * shard.row_bytes for tensor in table.tensors for shard in tensor.shards)
            self.logger.info(
                f"Published {len(table.tensors)} FP32-master/BF16-wire tensors from "
                f"{len(table.agents)} trainer agents ({total_bytes / 1e9:.2f} GB)"
            )
        self._initialized = True

    @staticmethod
    def _validate_table(table: TrainerTable) -> None:
        """Fail before publication unless every logical tensor is tiled once."""
        for tensor in table.tensors:
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
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)
            self.rendezvous.wait_for(
                "orchestrator",
                count=1,
                status=p2p_pb2.SOURCE_STATUS_READY,
                timeout=self.config.timeout,
            )
        dist.barrier()

        if self.is_serving_rank:
            for shard in self._shards:
                shard.refresh()
            torch.cuda.synchronize()
        dist.barrier()

        if self.world.is_master:
            # The transition through INITIALIZING makes the following READY
            # acknowledgements generation-safe despite MX exposing only status.
            self.rendezvous.wait_for(
                "inference",
                count=self.config.inference_world_size,
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
                timeout=self.config.timeout,
            )
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
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
        dist.barrier()
        for run_index in ready:
            self.multi_run_manager.ready_to_update[run_index] = False
        self.logger.info(f"NIXL+MX policy v{step} synchronized in {time.perf_counter() - start:.2f}s")
