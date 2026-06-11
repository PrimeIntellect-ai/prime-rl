"""Serve weights to the inference engine as a NIXL pull source.

The trainer side of the NIXL weight broadcast is a *passive weight store*:
it holds a full bf16 copy of the model's native state dict — partitioned
across the participating ranks, in persistent NIXL-registered buffers — and
publishes one table to Model Express describing where every tensor lives
(name, shape, dtype, device address, owning agent). It knows nothing about
its consumers: inference workers bake their own pull plans from the table
(see ``prime_rl.inference.vllm.worker.nixl``) and issue RDMA READs, so
inference can scale out, restart, or fail without any trainer-side
coordination. No model conversion runs anywhere — tensors are served
exactly as the trainer holds them, and vLLM's own weight loaders define
the slicing on the consumer side.

Per sync (coordinated by the existing filesystem markers):

1. master touches STABLE -> orchestrator pauses engines -> ready marker,
2. all ranks gather the state dict layer by layer (DTensor collectives);
   each participating rank copies the tensors it owns into its store,
3. after a barrier, the master touches NIXL_DONE in the step's broadcast
   dir; workers (blocked in their ``update_weights`` RPC) pull and resume.

The store is only rewritten inside the next sync's paused window, which the
orchestrator serializes behind this sync's pulls — so pulls never race a
store refresh.

HSDP: only the primary replica (``dp_replicate`` rank 0) holds store
partitions; the other replicas hold bit-identical weights and only
participate in the gather collectives.
"""

from __future__ import annotations

import heapq
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.markers import OrchestratorMarkers
from prime_rl.trainer.rl.broadcast.nccl import filter_state_dict_by_layers
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.trainer.weights import get_max_layer_num
from prime_rl.utils.logger import get_logger
from prime_rl.utils.vlm import get_layer_prefix
from prime_rl.weight_transfer.cuda_pool import classic_cuda_alloc
from prime_rl.weight_transfer.mx import MX_MODEL_NAME, MxRendezvous
from prime_rl.weight_transfer.nixl import NixlAgent, make_agent_name, set_ucx_env_defaults
from prime_rl.weight_transfer.wire import NIXL_DONE_MARKER, TrainerAgent, TrainerTable, TrainerTensor, encode_table

_STORE_ALIGN = 256


class NIXLWeightBroadcast(WeightBroadcast, OrchestratorMarkers):
    """Serve weights to the inference engine as a NIXL pull source."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        parallel_dims: ParallelDims,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(output_dir)
        self.logger = get_logger()
        self.config = config
        self.world = get_world()
        self.multi_run_manager = get_multi_run_manager()
        self.parallel_dims = parallel_dims
        self.dtype = dtype

        if parallel_dims.pp_enabled:
            raise NotImplementedError("NIXL weight broadcast does not support pipeline parallelism")
        self.num_participants = self.world.world_size // parallel_dims.dp_replicate
        # The device mesh orders dp_replicate outermost, so the primary replica
        # is exactly the first `num_participants` global ranks. Verified below.
        is_primary_by_rank = self.world.rank < self.num_participants
        if self.is_primary_replica_rank != is_primary_by_rank:
            raise RuntimeError(
                "NIXL weight broadcast assumes dp_replicate is the outermost mesh dim "
                f"(rank={self.world.rank}, num_participants={self.num_participants})"
            )

        if self.is_primary_replica_rank:
            set_ucx_env_defaults()
            self.nixl_agent = NixlAgent(name=make_agent_name("trainer", self.world.rank))

        self.is_initialized = False

    @property
    def is_primary_replica_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    # ------------------------------ store ------------------------------ #

    def _lazy_init(self, model: nn.Module) -> None:
        """Allocate this rank's store partition and publish the combined table.

        Runs once, on the first broadcast. Every rank derives the identical
        tensor->owner assignment from the (replicated) state-dict metadata;
        the master gathers all partitions' addresses and agent metadata and
        publishes one TrainerTable to Model Express.
        """
        if self.is_initialized:
            return
        start = time.perf_counter()

        # (name, dtype, full shape) of the served state dict. DTensors report
        # their full (global) shape and are served in self.dtype, mirroring
        # the gather cast in _resolve_dtensors.
        metas = [
            (name, self.dtype if isinstance(value, DTensor) else value.dtype, tuple(value.shape))
            for name, value in model.state_dict().items()
        ]
        owners = self._assign_owners(metas)

        my_rows: list[TrainerTensor] = []
        if self.is_primary_replica_rank:
            my_rows = self._allocate_store(metas, owners)

        # The master needs every participant's agent metadata and rows to
        # publish one atomic table (workers wait for exactly one MX entry).
        payload = None
        if self.is_primary_replica_rank:
            payload = (self.world.rank, self.nixl_agent.name, self.nixl_agent.get_metadata(), my_rows)
        gathered: list | None = [None] * self.world.world_size if self.world.is_master else None
        dist.gather_object(payload, gathered, dst=0)

        if self.world.is_master:
            parts = sorted((p for p in gathered if p is not None), key=lambda p: p[0])
            assert len(parts) == self.num_participants
            agents = [TrainerAgent(name=agent_name, metadata=metadata) for _, agent_name, metadata, _ in parts]
            tensors = [
                TrainerTensor(
                    name=row.name,
                    dtype=row.dtype,
                    shape=row.shape,
                    addr=row.addr,
                    device_id=row.device_id,
                    agent=agent_idx,
                )
                for agent_idx, (_, _, _, rows) in enumerate(parts)
                for row in rows
            ]
            from modelexpress.client import MxClient

            rendezvous = MxRendezvous(
                client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
                role="trainer",
                rank=0,
                peer_world_size=0,
                model_name=MX_MODEL_NAME,
                # Fixed worker_id: a restarted trainer overwrites its table
                # instead of leaving workers a stale duplicate to race on.
                worker_id="trainer-table",
            )
            rendezvous.publish(nixl_metadata=encode_table(TrainerTable(agents=agents, tensors=tensors)))
            total_bytes = sum(dtype.itemsize * _numel(shape) for _, dtype, shape in metas)
            self.logger.info(
                f"NIXL weight store published: {len(tensors)} tensors ({total_bytes / 1e9:.2f} GB) "
                f"across {len(agents)} agents in {time.perf_counter() - start:.2f}s"
            )
        self.is_initialized = True

    def _assign_owners(self, metas: list[tuple[str, torch.dtype, tuple[int, ...]]]) -> dict[str, int]:
        """Deterministic greedy bin-packing of tensors over participating ranks.

        Identical on every rank (pure function of the replicated metadata), so
        owners, store layouts, and the published table always agree.
        """
        ordered = sorted(metas, key=lambda m: (-m[1].itemsize * _numel(m[2]), m[0]))
        bins: list[tuple[int, int]] = [(0, rank) for rank in range(self.num_participants)]
        heapq.heapify(bins)
        owners: dict[str, int] = {}
        for name, dtype, shape in ordered:
            load, rank = heapq.heappop(bins)
            owners[name] = rank
            heapq.heappush(bins, (load + dtype.itemsize * _numel(shape), rank))
        return owners

    def _allocate_store(
        self, metas: list[tuple[str, torch.dtype, tuple[int, ...]]], owners: dict[str, int]
    ) -> list[TrainerTensor]:
        """Allocate one flat registered buffer holding this rank's tensors."""
        mine = [(name, dtype, shape) for name, dtype, shape in metas if owners[name] == self.world.rank]
        offsets: dict[str, int] = {}
        total = 0
        for name, dtype, shape in mine:
            offsets[name] = total
            nbytes = dtype.itemsize * _numel(shape)
            total += (nbytes + _STORE_ALIGN - 1) // _STORE_ALIGN * _STORE_ALIGN

        device_index = torch.cuda.current_device()
        with classic_cuda_alloc():
            self._store = torch.empty(max(total, 1), dtype=torch.uint8, device=device_index)
        self.nixl_agent.register_tensor(self._store)

        self._store_views: dict[str, torch.Tensor] = {}
        rows: list[TrainerTensor] = []
        for name, dtype, shape in mine:
            nbytes = dtype.itemsize * _numel(shape)
            view = self._store.narrow(0, offsets[name], nbytes).view(dtype).view(shape)
            self._store_views[name] = view
            rows.append(
                TrainerTensor(
                    name=name,
                    dtype=str(dtype).removeprefix("torch."),
                    shape=shape,
                    addr=view.data_ptr(),
                    device_id=device_index,
                    agent=-1,  # patched in by the master when merging partitions
                )
            )
        self.logger.debug(f"NIXL weight store: this rank serves {len(mine)} tensors ({total / 1e9:.2f} GB)")
        return rows

    # ------------------------------ sync ------------------------------ #

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        start = time.perf_counter()
        self._lazy_init(model)

        # The store must never be rewritten while engines run (workers pull
        # from it lazily): only refresh once the orchestrator has paused them
        # (ready marker below). Without a notified run there is no pause this
        # step, so skip.
        notified_runs = self._compute_notified_runs()
        if not notified_runs:
            self.logger.warning(f"No runs ready for weight broadcast at step {step}, skipping NIXL store refresh")
            return
        if self.world.is_master:
            self._notify_orchestrator(notified_runs)
        self._wait_for_ready_marker(notified_runs)

        state_dict = model.state_dict()
        layer_prefix = get_layer_prefix(model.config)
        num_layers = get_max_layer_num(state_dict, layer_prefix)

        filled = 0
        for _, layer_state_dict in filter_state_dict_by_layers(state_dict, num_layers, layer_prefix):
            # DTensor resolution is collective — every rank participates, even
            # the non-primary HSDP replicas that own no store partition.
            layer_state_dict = self._resolve_dtensors(layer_state_dict)
            if not self.is_primary_replica_rank:
                continue
            for name, tensor in layer_state_dict.items():
                view = self._store_views.get(name)
                if view is not None:
                    view.copy_(tensor)
                    filled += 1
        if self.is_primary_replica_rank:
            if filled != len(self._store_views):
                raise RuntimeError(
                    f"state dict filled {filled}/{len(self._store_views)} owned store tensors — "
                    "served table is out of sync with the model"
                )
            torch.cuda.synchronize()

        # Every rank's store partition must be filled before workers may pull;
        # the master then drops the step-scoped marker they block on.
        dist.barrier()
        if self.world.is_master:
            for _, save_dir in notified_runs:
                (save_dir / NIXL_DONE_MARKER).touch()
        self.logger.debug(f"NIXL weight store refreshed in {time.perf_counter() - start:.2f}s")

    def _resolve_dtensors(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key, value in list(state_dict.items()):
            if isinstance(value, DTensor):
                state_dict[key] = value.to(self.dtype).full_tensor()
        return state_dict


def _numel(shape: tuple[int, ...]) -> int:
    numel = 1
    for s in shape:
        numel *= s
    return numel
