"""Serve weights to the inference engine as a sharded NIXL pull source.

The trainer side of the NIXL weight broadcast is a *passive, sharded weight
store*: every rank registers its own local DTensor shards — no all-gather, no
format/naming conversion — and the master publishes one table to Model
Express describing, for each state-dict tensor, which dim-0 row range lives on
which rank's buffer. Inference workers bake their own pull plans from the
table (see ``prime_rl.inference.vllm.worker.nixl``) and RDMA-READ only the
slices they consume directly from the owning ranks. The trainer never learns
who its consumers are, so inference can scale out, restart, or fail freely.

Parameters are sharded only along dim 0 (FSDP shards the output/vocab dim,
expert-parallel shards the expert dim). For an EP-sharded expert weight, the
owned global-expert block follows the same convention vLLM's FusedMoE uses
(``ep_rank * experts_per_ep + fsdp_rank * num_local``); for everything else
the owned dim-0 range comes straight from the DTensor placement.

Serving dtype is bf16 (vLLM inference dtype). When the trainer already holds
bf16 params, their local shard storage is registered and served live — zero
per-sync work. When it holds fp32 master weights (the default
``optimization_dtype``), each rank casts its *local shard* into a persistent
bf16 buffer per sync (a sharded cast, still no gather).

Per sync (coordinated by the existing filesystem markers):

1. master touches STABLE -> orchestrator pauses engines -> ready marker,
2. each rank refreshes its shard buffers (no-op when serving live bf16),
3. master touches NIXL_DONE; workers pull, then ack; the trainer waits for all
   acks before returning so no shard buffer is overwritten under a live pull.

HSDP: only the primary replica (``dp_replicate`` rank 0) serves shards; the
other replicas hold identical weights and do nothing.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.markers import OrchestratorMarkers
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
from prime_rl.weight_transfer.mx import MX_MODEL_NAME, MxRendezvous
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.wire import (
    NIXL_DONE_MARKER,
    NIXL_PULLED_MARKER,
    TrainerAgent,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
    encode_table,
)

_SERVE_DTYPE = torch.bfloat16
_EXPERT_RE = re.compile(r"\.mlp\.experts\.w[123]$")


class _OwnedShard:
    """One local shard this rank serves: its global dim-0 range, the live
    source tensor, and the registered bf16 buffer NIXL reads from."""

    def __init__(self, name: str, full_shape: tuple[int, ...], row_start: int, source: torch.Tensor):
        self.name = name
        self.full_shape = full_shape
        self.row_start = row_start
        self.source = source  # local shard (fp32 or bf16), cast into `buffer`
        self.num_rows = source.shape[0]
        self.row_numel = source[0].numel() if self.num_rows else 0
        # Serve live storage when already bf16; otherwise a persistent bf16 cast buffer.
        self.live = source.dtype == _SERVE_DTYPE
        self.buffer = source if self.live else None

    def allocate_buffer(self) -> None:
        if self.buffer is None:
            with classic_cuda_alloc():
                self.buffer = torch.empty(self.source.shape, dtype=_SERVE_DTYPE, device=self.source.device)

    def refresh(self) -> None:
        if not self.live:
            assert self.buffer is not None
            self.buffer.copy_(self.source)


class NIXLWeightBroadcast(WeightBroadcast, OrchestratorMarkers):
    """Serve weights to the inference engine as a sharded NIXL pull source."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__(output_dir)
        self.logger = get_logger()
        self.config = config
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.parallel_dims = parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError("NIXL weight broadcast does not support pipeline parallelism")

        if self.is_primary_replica_rank:
            set_ucx_env_defaults()
            self.nixl_agent = NixlAgent(name=make_agent_name("trainer", self.world.rank))

        self.is_initialized = False
        self._shards: list[_OwnedShard] = []

    @property
    def is_primary_replica_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    # ------------------------------ ownership ------------------------------ #

    def _owned_shards(self, model: nn.Module) -> list[_OwnedShard]:
        """This rank's local shards (primary replica only).

        Replicated tensors are served from the global master alone to avoid
        duplicate registrations; sharded tensors are served per rank from the
        dim-0 range the placement assigns.
        """
        if self.parallel_dims.ep_enabled:
            ep_mesh = self.parallel_dims.get_mesh("ep")
            fsdp_mod_ep = self.parallel_dims.get_mesh("dp_shard_mod_ep")
            ep_rank = ep_mesh.get_local_rank()
            fsdp_mod_ep_size, fsdp_mod_ep_rank = fsdp_mod_ep.size(), fsdp_mod_ep.get_local_rank()
        else:
            ep_rank = fsdp_mod_ep_size = fsdp_mod_ep_rank = 0

        shards: list[_OwnedShard] = []
        for name, value in model.state_dict().items():
            full_shape = tuple(value.shape)
            if not isinstance(value, DTensor):
                # Plain replicated tensor — served by the master only.
                if self.world.is_master:
                    shards.append(_OwnedShard(name, full_shape, 0, value.detach()))
                continue

            local = value.to_local().detach()
            if _EXPERT_RE.search(name):
                # EP + FSDP both shard dim 0 (experts). Owned global-expert
                # block matches vLLM's FusedMoE numbering.
                num_local = local.shape[0]
                experts_per_ep = num_local * max(fsdp_mod_ep_size, 1)
                row_start = ep_rank * experts_per_ep + fsdp_mod_ep_rank * num_local
                if num_local:
                    shards.append(_OwnedShard(name, full_shape, row_start, local))
                continue

            placements = value._spec.placements
            if all(p.is_replicate() for p in placements):
                if self.world.is_master:
                    shards.append(_OwnedShard(name, full_shape, 0, local))
                continue

            # Sharded on dim 0 (FSDP / HSDP). The DTensor placement gives this
            # rank's logical row range directly (handles uneven splits).
            local_shape, global_offset = compute_local_shape_and_global_offset(
                value.shape, value._spec.mesh, placements
            )
            if any(o != 0 for o in global_offset[1:]) or tuple(local_shape[1:]) != full_shape[1:]:
                raise NotImplementedError(
                    f"NIXL weight broadcast supports dim-0 sharding only; {name} is sharded on a "
                    f"non-zero dim (local_shape={local_shape}, global_offset={global_offset})"
                )
            num_rows = local_shape[0]
            if num_rows:
                shards.append(_OwnedShard(name, full_shape, global_offset[0], local[:num_rows]))
        return shards

    # ------------------------------ init ------------------------------ #

    def _lazy_init(self, model: nn.Module) -> None:
        if self.is_initialized:
            return

        if self.is_primary_replica_rank:
            self._shards = self._owned_shards(model)
            for shard in self._shards:
                shard.allocate_buffer()
                shard.refresh()
                self.nixl_agent.register_tensor(shard.buffer)
            torch.cuda.synchronize()

        # Master gathers every rank's agent + shard rows and publishes one table.
        payload = None
        if self.is_primary_replica_rank:
            rows = [
                (
                    s.name,
                    str(s.buffer.dtype).removeprefix("torch."),
                    s.full_shape,
                    s.row_start,
                    s.num_rows,
                    s.buffer.data_ptr(),
                    s.row_numel * s.buffer.element_size(),
                    s.buffer.device.index,
                )
                for s in self._shards
            ]
            payload = (self.world.rank, self.nixl_agent.name, self.nixl_agent.get_metadata(), rows)
        gathered: list | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(payload, gathered, dst=0)

        if self.world.is_master:
            parts = sorted((p for p in gathered if p is not None), key=lambda p: p[0])
            agents = [TrainerAgent(name=name, metadata=meta) for _, name, meta, _ in parts]
            by_name: dict[str, TrainerTensor] = {}
            for i, (_, agent_name, _, rows) in enumerate(parts):
                for name, dtype, shape, row_start, num_rows, addr, row_bytes, device_id in rows:
                    t = by_name.get(name)
                    if t is None:
                        t = by_name[name] = TrainerTensor(name=name, dtype=dtype, shape=tuple(shape), shards=[])
                    t.shards.append(
                        TrainerShard(
                            agent=i,
                            row_start=row_start,
                            num_rows=num_rows,
                            addr=addr,
                            row_bytes=row_bytes,
                            device_id=device_id,
                        )
                    )
            self._publish(TrainerTable(agents=agents, tensors=list(by_name.values())))

        self.is_initialized = True

    def _publish(self, table: TrainerTable) -> None:
        from modelexpress.client import MxClient

        rendezvous = MxRendezvous(
            client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
            role="trainer",
            rank=0,
            peer_world_size=0,
            model_name=MX_MODEL_NAME,
            # Fixed worker_id: a restarted trainer overwrites its table.
            worker_id="trainer-table",
        )
        rendezvous.publish(nixl_metadata=encode_table(table))
        total = sum(s.num_rows * s.row_bytes for t in table.tensors for s in t.shards)
        self.logger.info(
            f"NIXL shard table published: {len(table.tensors)} tensors across {len(table.agents)} agents "
            f"({total / 1e9:.2f} GB) in MX"
        )

    # ------------------------------ sync ------------------------------ #

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        start = time.perf_counter()
        self._lazy_init(model)

        # One-sided pulls must never race a shard refresh or the next optimizer
        # step: only proceed once the orchestrator has paused the engines
        # (ready marker). No notified run => no pause this step => skip.
        notified_runs = self._compute_notified_runs()
        if not notified_runs:
            self.logger.warning(f"No runs ready for weight broadcast at step {step}, skipping NIXL refresh")
            return
        if self.world.is_master:
            self._notify_orchestrator(notified_runs)
        self._wait_for_ready_marker(notified_runs)

        if self.is_primary_replica_rank:
            for shard in self._shards:
                shard.refresh()
            torch.cuda.synchronize()

        # Every rank's shards must be current before workers may pull; the
        # master then drops the marker, and we wait for pull acks so no buffer
        # is overwritten (next refresh / optimizer step) under a live read.
        dist.barrier()
        if self.world.is_master:
            for _, save_dir in notified_runs:
                (save_dir / NIXL_DONE_MARKER).touch()
            self._wait_for_pull_acks(notified_runs)
        dist.barrier()
        self.logger.debug(f"NIXL shards refreshed + pulled in {time.perf_counter() - start:.2f}s")

    def _wait_for_pull_acks(self, notified_runs: list[tuple[int, Path]]) -> None:
        deadline = time.monotonic() + self.config.timeout
        for _, save_dir in notified_runs:
            while True:
                acked = len(list(save_dir.glob(f"{NIXL_PULLED_MARKER}.*")))
                if acked >= self.config.inference_world_size:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for pull acks in {save_dir} ({acked}/{self.config.inference_world_size})"
                    )
                time.sleep(0.1)
