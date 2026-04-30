"""Trainer-side NIXL (UCX/RDMA) weight sender.

Thin orchestrator around :class:`~prime_rl.trainer.rl.broadcast.transport_plan.TransportPlan`:
owns the NIXL agent + SPG, delegates every model/FSDP/EP-aware concern to
the plan, and runs the orchestrator handshake around each push.

HSDP: when ``dp_replicate > 1``, only the primary replica (``dp_replicate
rank 0``) participates in the NIXL transfer. Non-primary replicas hold
bit-identical weights, so a second copy over the wire would be pure
waste. Both replicas still hit ``broadcast_weights`` — the non-primary
path is a pair of ``dist.barrier()`` calls that keep the replicas in
lockstep with the primary's push.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from vllm.distributed.utils import StatelessProcessGroup

from prime_rl.configs.trainer import NIXLWeightBroadcastConfig
from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.rl.broadcast.transport_plan import TransportPlan
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.utils.logger import get_logger
from prime_rl.utils.mx_rendezvous import (
    discover_spg_coordinator,
    mx_rendezvous_enabled,
)
from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path

NIXL_READY_MARKER = "NIXL_READY"


class NIXLWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via NIXL (zero-copy RDMA)."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLWeightBroadcastConfig,
        model: PreTrainedModelPrimeRL,
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__(output_dir)
        self.logger = get_logger()
        self.config = config
        self.world = get_world()
        self.parallel_dims = parallel_dims
        self._multi_run_manager = None

        # Only the primary HSDP replica runs NIXL. For dp_replicate == 1 this
        # is every rank.
        if parallel_dims.dp_replicate_enabled:
            self.is_primary_replica = parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        else:
            self.is_primary_replica = True

        if not self.is_primary_replica:
            return

        self._plan = TransportPlan(model, parallel_dims)
        self._agent = NixlAgentWrapper(
            name=make_agent_name("trainer", self.world.rank),
            local_rank=self.world.local_rank,
            backends=config.backends,
        )
        self._plan.register(self._agent)

        # SPG ranks cover replica-0 trainer ranks plus every inference rank.
        # Non-primary replicas are absent from the SPG entirely.
        trainer_ws_per_replica = parallel_dims.dp_shard * parallel_dims.cp
        spg_rank = parallel_dims.get_mesh("dp_shard_cp").get_local_rank() if dist.is_initialized() else 0

        # ModelExpress overlay: when PRIME_RL_MX_RENDEZVOUS is set, discover
        # the SPG coordinator host/port via MX Server instead of using the
        # hard-coded config values. Default (unset) preserves the exact
        # upstream PI behavior.
        if mx_rendezvous_enabled():
            endpoint = discover_spg_coordinator(
                role="trainer",
                rank=spg_rank,
                expected_trainer_ws=trainer_ws_per_replica,
                expected_inference_ws=config.inference_world_size,
                fallback_host=config.host,
                fallback_port=config.port,
            )
            spg_host, spg_port = endpoint.host, endpoint.port
            self.logger.info(
                f"[mx-rendezvous] trainer rank={spg_rank} using discovered "
                f"SPG coordinator {spg_host}:{spg_port} (source_id={endpoint.source_id})"
            )
        else:
            spg_host, spg_port = config.host, config.port

        self._spg = StatelessProcessGroup.create(
            host=spg_host,
            port=spg_port,
            rank=spg_rank,
            world_size=trainer_ws_per_replica + config.inference_world_size,
            store_timeout=config.timeout,
        )
        self._plan.rendezvous(self._spg, self._agent, config.inference_world_size)

    @torch.no_grad()
    def push_once(self, model: PreTrainedModelPrimeRL) -> None:
        self._plan.push_once(model, self._agent, self._spg)

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        start = time.perf_counter()
        notified_runs: list[tuple[int, Path]] = []
        if self.world.is_master:
            notified_runs = self._notify_orchestrator()
            self._wait_for_nixl_ready(notified_runs)
        # All trainer ranks (primary + replica copies) barrier before any
        # RDMA WRITE starts — non-master ranks must not race past NIXL_READY.
        dist.barrier()
        if self.is_primary_replica:
            self.push_once(model)
        # Second barrier so non-primary replicas exit in lockstep with the
        # primary's push rather than running ahead into the next step while
        # the primary is still draining / in SPG barrier.
        dist.barrier()
        self.logger.debug(f"NIXL weights broadcasted in {time.perf_counter() - start:.2f}s")

    @property
    def multi_run_manager(self):
        if self._multi_run_manager is None:
            self._multi_run_manager = get_multi_run_manager()
        return self._multi_run_manager

    def _notify_orchestrator(self) -> list[tuple[int, Path]]:
        notified_runs: list[tuple[int, Path]] = []
        for idx in self.multi_run_manager.used_idxs:
            if not self.multi_run_manager.ready_to_update[idx]:
                continue
            save_dir = get_step_path(
                get_broadcast_dir(self.multi_run_manager.get_run_dir(idx)),
                self.multi_run_manager.progress[idx].step,
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "STABLE").touch()
            notified_runs.append((idx, save_dir))
            self.multi_run_manager.ready_to_update[idx] = False
        return notified_runs

    def _wait_for_nixl_ready(self, notified_runs: list[tuple[int, Path]]) -> None:
        for idx, save_dir in notified_runs:
            sync_wait_for_path(save_dir / NIXL_READY_MARKER, interval=0.1, log_interval=10)
