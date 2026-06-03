"""v2 trainer-side weight broadcast using the ModelExpress v2 fat clients.

This is the v2 of :class:`NIXLMxWeightBroadcast` (PR #2389), built on
:class:`MxV2TrainingPublisher` instead of the in-tree :class:`MxRendezvous`.
The data plane is unchanged — NIXL RDMA, GPU-direct, no CPU staging —
but the control-plane glue (heartbeat, freshest-per-rank dedup, same-rank
routing, compile_target metadata, multi-source slice picker, tree
fan-out) is graduated onto the published MX v2 surface.

The trainer-side conversion (FP8 packing, fusion, sharding into
``Sharded`` / ``Gathered`` / ``Expert`` slots) is *unchanged* from
PR #2389 — prime-rl still owns that kernel. What changes is **how the
already-converted bytes get published** (one ``publisher.publish()``
per step instead of a per-tensor ``post_write`` loop driven from the
trainer), and **what metadata rides along** (``compile_target`` +
``compile_metadata`` from the conversion registry + per-tensor MoE
expert ownership).

HSDP: when ``dp_replicate > 1`` only the primary replica (``dp_replicate
rank 0``) participates. Non-primary replicas hold bit-identical weights;
broadcasting a second copy would be pure waste.

See :file:`docs/proposals/post-pr2389-mx-v2.md` for the design rationale
and migration plan.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress.nemo_rl_v2 import MxV2TrainingPublisher, TrainerWorldLayout
from transformers import AutoConfig

from prime_rl.configs.trainer import MxV2WeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.models.conversions import select_default_conversion
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.transport.classic_cuda_pool import classic_cuda_alloc
from prime_rl.transport.nixl_agent import make_agent_name, pin_ucx_rail


class NIXLMxV2WeightBroadcast(WeightBroadcast):
    """v2 weight broadcast over NIXL + ModelExpress fat clients.

    Selectable from config via ``weight_broadcast.type = "mx_v2"``.
    Coexists with the existing ``"nixl_mx"`` path (PR #2389); no
    behavior of ``"nixl_mx"`` is affected by importing this module.

    Args:
        output_dir: training output directory (forwarded to base class).
        config: parsed :class:`MxV2WeightBroadcastConfig`.
        parallel_dims: ``ParallelDims`` instance describing the trainer's
            FSDP / TP / EP / DP layout — used to construct the
            ``TrainerWorldLayout`` carried in v2 metadata.
    """

    def __init__(
        self,
        output_dir: Path,
        config: MxV2WeightBroadcastConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__(output_dir)
        self.config = config
        self.world = get_world()
        self.parallel_dims = parallel_dims

        self.is_initialized = False
        self._publisher: MxV2TrainingPublisher | None = None
        self._model_slots: list[Any] | None = None
        self._conversion = None
        self._hf_config = None

        if self.is_primary_hsdp_rank:
            pin_ucx_rail(torch.cuda.current_device())

        self._multi_run_manager = get_multi_run_manager()

    # ------------------------------------------------------------------
    # HSDP gate — only rank 0 of dp_replicate publishes
    # ------------------------------------------------------------------

    @property
    def is_primary_hsdp_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build_world_layout(self) -> TrainerWorldLayout:
        """Translate prime-rl's ParallelDims into the MX v2 world layout."""
        return TrainerWorldLayout(
            fsdp_world_size=getattr(self.parallel_dims, "dp_shard_size", 1),
            tp_world_size=getattr(self.parallel_dims, "tp_size", 1),
            pp_world_size=getattr(self.parallel_dims, "pp_size", 1),
            ep_world_size=getattr(self.parallel_dims, "ep_size", 1),
        )

    def lazy_init(self, model: PreTrainedModelPrimeRL) -> None:
        """Build the v2 publisher + slot layout on first call.

        The model isn't available at ``__init__`` time (the WeightBroadcast
        instance is constructed before the trainer model is materialized),
        so slot construction and publisher initialization happen on the
        first ``broadcast_weights`` call.
        """
        if self.is_initialized:
            return

        self._hf_config = AutoConfig.from_pretrained(self.config.inference_model_name)
        self._conversion = select_default_conversion(self.config.inference_model_name)

        with classic_cuda_alloc():
            self._model_slots = model.build_slots(
                self.parallel_dims, self._conversion, self._hf_config.torch_dtype
            )

        # The v2 publisher owns the NIXL agent + MX client + heartbeat.
        # We pass our rank as ``worker_rank``; receivers with
        # ``same_rank_only=True`` (Phase 2 default) will only pull from
        # the trainer rank matching their own.
        world_layout = self._build_world_layout()
        self._publisher = MxV2TrainingPublisher(
            agent_name=make_agent_name("trainer", self.world.rank),
            device_id=torch.cuda.current_device(),
            mx_server_url=f"{self.config.host}:{self.config.port}",
            worker_rank=self.world.rank,
            world_layout=world_layout,
        )
        self._publisher.initialize(
            model_name=self.config.inference_model_name,
            dtype=str(self._hf_config.torch_dtype).replace("torch.", ""),
        )
        self.is_initialized = True
        # `select_default_conversion` may return either a registered conversion
        # object (with .compile_target + .compile_metadata) on the newer
        # conversion registry, OR a plain string ('bf16_cast', 'fp8_pack', ...)
        # on older registries. Use getattr so we degrade gracefully.
        conversion_target = getattr(self._conversion, "compile_target", str(self._conversion))
        self.logger.info(
            f"[mx_v2] publisher initialized: rank={self.world.rank} "
            f"layout={world_layout.encode()} "
            f"compile_target={conversion_target}"
        )

    # ------------------------------------------------------------------
    # Per-step broadcast
    # ------------------------------------------------------------------

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        """Publish version ``step`` of the converted weights.

        Per-step lifecycle:

        1. (HSDP) only the primary replica participates; others barrier.
        2. Fill the conversion slots from ``model.state_dict()`` —
           **same code path PR #2389 uses**, prime-rl owns the kernel.
        3. For each slot's buffers, call ``publisher.add_tensor(...)``
           tagged with the conversion's ``compile_target`` /
           ``compile_metadata`` (Phase 3) and any per-tensor MoE
           expert metadata.
        4. ``publisher.publish(version=step)`` + ``mark_ready()`` —
           catalog entry now visible to receivers polling for
           ``min_version=step``.
        5. Bump the heartbeat (the publisher's ``HeartbeatThread``
           runs in the background; nothing to do here).
        """
        if self.is_primary_hsdp_rank:
            self.lazy_init(model)

        if self.world.is_master:
            for idx in self._multi_run_manager.used_idxs:
                if self._multi_run_manager.ready_to_update[idx]:
                    self._multi_run_manager.ready_to_update[idx] = False

        dist.barrier()

        if not self.is_primary_hsdp_rank:
            # Non-primary HSDP replicas: bit-identical weights; nothing to publish.
            dist.barrier()
            return

        start = time.perf_counter()

        # 2. Fill slots from the live model state-dict via the conversion.
        #    This is where FP8 packing + fusion happens; same code path
        #    as PR #2389. We do NOT change the kernel.
        #    GatheredSlot's API takes only the state_dict; the conversion
        #    is baked into the slot at `from_spec` creation time.
        state_dict = model.state_dict()
        for slot in self._model_slots:
            slot.convert(state_dict)

        # 3. Register every slot tensor with the v2 publisher, tagged with
        #    compile_target + compile_metadata so receivers can refuse
        #    mismatched layouts at discovery (Phase 3).
        #    Falls back to the safe "hf_raw" default when:
        #      - publish_compile_target=False (caller opts out), or
        #      - the conversion is on an older registry without the
        #        compile_target/compile_metadata fields (graceful
        #        degradation; back-compat with PR #2389 conversions).
        if self.config.publish_compile_target:
            compile_target = getattr(self._conversion, "compile_target", "hf_raw")
            compile_metadata = getattr(self._conversion, "compile_metadata", None)
        else:
            compile_target = "hf_raw"
            compile_metadata = None

        n_tensors = 0
        for slot in self._model_slots:
            slot_is_expert = bool(getattr(slot, "is_expert", False))
            slot_expert_axis = int(getattr(slot, "expert_axis", 0))
            slot_owned_experts = tuple(getattr(slot, "owned_expert_ids", ()))
            for buf_key, tensor, _ in slot.buffers:
                self._publisher.add_tensor(
                    name=buf_key,
                    tensor=tensor,
                    is_expert=slot_is_expert,
                    expert_axis=slot_expert_axis,
                    owned_expert_ids=slot_owned_experts,
                    compile_target=compile_target,
                    compile_metadata=compile_metadata,
                )
                n_tensors += 1

        # 4. Publish + mark READY in one shot.
        mx_source_id = self._publisher.publish(version=step)
        self._publisher.mark_ready()

        elapsed = time.perf_counter() - start
        self.logger.info(
            f"[mx_v2] publish step={step} tensors={n_tensors} "
            f"compile_target={compile_target} mx_source_id={mx_source_id} "
            f"elapsed={elapsed:.3f}s"
        )

        dist.barrier()

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._publisher is not None:
            try:
                self._publisher.shutdown()
            finally:
                self._publisher = None
        self.is_initialized = False
