"""Broadcast weights into the inference engine via NIXL + Model Express.

Thin lifecycle wrapper around :class:`TrainerPublisher` +
:class:`TransportPlan`. Slot allocation and the MX rendezvous are deferred
to the first :meth:`broadcast_weights` call because the trainer model is
not available at ``setup_weight_broadcast`` time.

HSDP: when ``dp_replicate > 1`` only the primary replica (``dp_replicate
rank 0``) participates. Non-primary replicas hold bit-identical weights
so a second copy over the wire would be pure waste; they barrier-sync
to stay in lockstep.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from transformers import AutoConfig

from prime_rl.configs.trainer import NIXLMxWeightBroadcastConfig
from prime_rl.trainer.models.conversions import select_default_conversion
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.transport.trainer_publisher import TrainerPublisher
from prime_rl.transport.transport_plan import TransportPlan
from prime_rl.utils.pathing import sync_wait_for_path
from prime_rl.utils.utils import get_broadcast_dir, get_step_path

NIXL_MX_READY_MARKER = "NCCL_READY"


def _get_qwen3_moe_spec_fns(hf_config):
    """Return (layer_specs_fn, non_layer_specs, is_dense_fn) for Qwen3MoE."""
    from prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe import (
        conversion_specs,
        is_dense_layer,
        non_layer_conversion_specs,
    )

    return (
        conversion_specs,
        non_layer_conversion_specs(),
        lambda i: is_dense_layer(hf_config, i),
    )


class NIXLMxWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via NIXL (zero-copy RDMA)."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLMxWeightBroadcastConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__(output_dir)
        self.config = config
        self.world = get_world()
        self.parallel_dims = parallel_dims
        self._multi_run_manager = get_multi_run_manager()
        self._publisher: TrainerPublisher | None = None
        self._plan: TransportPlan | None = None

        if parallel_dims.dp_replicate_enabled:
            self._is_primary = parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        else:
            self._is_primary = True

    def _lazy_init(self, model: nn.Module) -> None:
        """Build publisher + transport plan on first call (needs the live model)."""
        if self._publisher is not None:
            return

        hf_config = AutoConfig.from_pretrained(self.config.inference_model_name)
        default_conversion = select_default_conversion(self.config.inference_model_name)
        layer_specs_fn, non_layer_specs, is_dense_fn = _get_qwen3_moe_spec_fns(hf_config)

        state_dict = dict(model.named_parameters())
        state_dict.update({k: v for k, v in model.named_buffers() if k not in state_dict})

        client = MxClient(server_url=f"{self.config.host}:{self.config.port}")

        self._publisher = TrainerPublisher(
            client=client,
            rank=self.world.rank,
            peer_world_size=self.config.inference_world_size,
            inference_model_name=self.config.inference_model_name,
            default_conversion=default_conversion,
            base_dtype=hf_config.torch_dtype,
            layer_specs_fn=layer_specs_fn,
            non_layer_specs=non_layer_specs,
            is_dense_fn=is_dense_fn,
            num_layers=hf_config.num_hidden_layers,
            state_dict=state_dict,
            parallel_dims=self.parallel_dims,
        )
        self._publisher.publish()

        self._plan = TransportPlan(self._publisher)
        self._plan.negotiate(timeout=self.config.timeout)
        self._plan.setup_remote_agents()
        self._plan.prepare()

        self.logger.info(
            f"NIXL+MX init complete: {len(self._publisher.slots)} slots, "
            f"{len(self._plan.peers)} peers, {len(self._plan.writes)} writes"
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        if not self._is_primary:
            dist.barrier()
            return

        self._lazy_init(model)
        assert self._plan is not None and self._publisher is not None

        start = time.perf_counter()
        self.logger.debug("Starting NIXL+MX weight push")

        # Same handshake as NCCL: STABLE → orchestrator pauses inference →
        # orchestrator creates NCCL_READY marker → trainer pushes (inference
        # is safely idle). Without this, RDMA writes land in live serving
        # buffers and corrupt mid-request weights.
        notified_runs = self._compute_notified_runs()
        if self.world.is_master:
            self._notify_orchestrator(notified_runs)
        self._wait_for_ready(notified_runs)

        state_dict = dict(model.named_parameters())
        state_dict.update({k: v for k, v in model.named_buffers() if k not in state_dict})
        self._plan.push_once(state_dict)

        self._publisher.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
        self.logger.debug(f"NIXL+MX push done in {time.perf_counter() - start:.2f}s")

        if self.parallel_dims.dp_replicate_enabled:
            dist.barrier()

    def _compute_notified_runs(self) -> list[tuple[int, Path]]:
        notified: list[tuple[int, Path]] = []
        for idx in self._multi_run_manager.used_idxs:
            if not self._multi_run_manager.ready_to_update[idx]:
                continue
            save_dir = get_step_path(
                get_broadcast_dir(self._multi_run_manager.get_run_dir(idx)),
                self._multi_run_manager.progress[idx].step,
            )
            notified.append((idx, save_dir))
        return notified

    def _wait_for_ready(self, notified_runs: list[tuple[int, Path]]) -> None:
        """Wait for the orchestrator to pause inference and create the ready marker."""
        for _, save_dir in notified_runs:
            ready_file = save_dir / NIXL_MX_READY_MARKER
            self.logger.debug(f"Waiting for ready marker at {ready_file}")
            sync_wait_for_path(ready_file, interval=0.1, log_interval=10)

    def _notify_orchestrator(self, notified_runs: list[tuple[int, Path]]) -> None:
        for idx, save_dir in notified_runs:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "STABLE").touch()
            self._multi_run_manager.ready_to_update[idx] = False
