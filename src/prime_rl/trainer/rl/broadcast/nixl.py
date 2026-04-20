"""Trainer-side NIXL (UCX/RDMA) weight sender.

Thin orchestrator around :class:`~prime_rl.trainer.rl.broadcast.transport_plan.TransportPlan`:
owns the NIXL agent + SPG, delegates every model/FSDP/EP-aware concern to
the plan, and runs the orchestrator handshake around each push.
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
        self._multi_run_manager = None

        self._plan = TransportPlan(model, parallel_dims)

        self._agent = NixlAgentWrapper(
            name=make_agent_name("trainer", self.world.rank),
            local_rank=self.world.local_rank,
            backends=config.backends,
        )
        self._plan.register(self._agent)

        self._spg = StatelessProcessGroup.create(
            host=config.host,
            port=config.port,
            rank=self.world.rank,
            world_size=self.world.world_size + config.inference_world_size,
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
        # Only rank 0 waits for NIXL_READY (the orchestrator drops it after all
        # inference engines acknowledge /pause). Hold every other trainer rank
        # here so no rank starts RDMA-writing into inference memory before all
        # engines have drained — otherwise a worker still running a forward pass
        # races the write into its own weight buffers.
        dist.barrier()
        self.push_once(model)
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
