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

        state_dict = model.state_dict()

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
        self.logger.info("Published to MX. Starting negotiate.")

        self._plan = TransportPlan(self._publisher)
        self._plan.negotiate(timeout=self.config.timeout)
        self.logger.info(f"Negotiate done ({len(self._plan.peers)} peers)")
        self._plan.setup_remote_agents()
        self._plan.prepare()

        self.logger.info(
            f"NIXL+MX init complete: {len(self._publisher.slots)} slots, "
            f"{len(self._plan.peers)} peers, {len(self._plan.writes)} writes"
        )

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        if self._is_primary:
            self._lazy_init(model)
            assert self._plan is not None and self._publisher is not None
            self._publisher.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)

        if self.world.is_master:
            for idx in self._multi_run_manager.used_idxs:
                if self._multi_run_manager.ready_to_update[idx]:
                    self._multi_run_manager.ready_to_update[idx] = False
            self._publisher.rendezvous.wait_for_all_peers_ready(role="orchestrator", timeout=self.config.timeout)

        dist.barrier()

        if self._is_primary:
            start = time.perf_counter()
            self._plan.push_once(model.state_dict())
            self._publisher.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.logger.info(f"NIXL+MX push completed in {time.perf_counter() - start:.2f}s")

        dist.barrier()
