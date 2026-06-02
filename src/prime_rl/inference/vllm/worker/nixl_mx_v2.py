"""v2 inference-worker extension for ModelExpress weight refits.

The v2 of :class:`NIXLMxWeightUpdateWorker` (PR #2389), built on the
``MxWeightTransferEngine`` adapter from ModelExpress PR #349. The adapter
wraps every Phase 2/3/4 capability behind vLLM's :class:`WeightTransferEngine`
ABC (the same shape Anyscale's RDT PR `#43375
<https://github.com/vllm-project/vllm/pull/43375>`_ targets), so this
module is **maximally thin** — it just instantiates the engine and
plumbs vLLM's ``load_weights`` callback through.

Key differences from PR #2389:

- **Pull semantics, not push.** The trainer ``publish()``-es weights to
  the MX catalog; this worker calls ``engine.receive_weights(...)``
  which discovers + pulls. No pre-registered NIXL buffers on the
  inference side (the engine uses the scratch-buffer path internally).
- **Compile-target safety net (Phase 3b).** Optional
  ``compile_target_filter`` refuses sources whose tensors don't match
  the kernel layout this worker expects — BEFORE any RDMA cycle is
  spent. Set ``filter=None`` for back-compat (accept anything).
- **Mixed-TP path (Phase 4).** When ``target_tp_layout`` is set, the
  engine uses the multi-source slice picker; otherwise it uses the
  matched-TP single-source fast path.
- **Tree fan-out (TensorHub pattern).** When
  ``publish_self_as_replica=True`` in the engine's ``init_info``, the
  worker republishes itself as a source after each successful receive,
  so subsequent receivers can pull from peers instead of the trainer.

See :file:`docs/proposals/post-pr2389-mx-v2.md` for the design rationale
and migration plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import update_mla_absorbed_weights
from prime_rl.transport.nixl_agent import make_agent_name, pin_ucx_rail

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker  # type: ignore
else:
    Worker = object  # type: ignore

logger = init_logger("vllm.inference.vllm.worker_nixl_mx_v2")


class NIXLMxV2WeightUpdateWorker(Worker):
    """vLLM worker extension for the v2 (pull-mode) weight-refit path.

    Mounted via vLLM's ``worker_extension_cls`` plumbing — same hook
    PR #2389 uses for ``NIXLMxWeightUpdateWorker``. Two RPC endpoints:

    - :meth:`init_nixl_mx_v2` — called once at worker boot, sets up the
      :class:`MxWeightTransferEngine` for this rank.
    - :meth:`update_weights_via_mx_v2` — called per refit cycle by the
      orchestrator; engine discovers + pulls + feeds vLLM's
      ``load_weights``.
    """

    # ------------------------------------------------------------------
    # Model accessor (matches PR #2389)
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> Module:
        model_runner = self.model_runner
        model = (
            model_runner.model.runnable
            if hasattr(model_runner.model, "runnable")
            else model_runner.model
        )
        assert isinstance(model, Module)
        return model

    # ------------------------------------------------------------------
    # Init RPC
    # ------------------------------------------------------------------

    def init_nixl_mx_v2(
        self,
        host: str,
        port: int,
        rank_offset: int,
        *,
        publish_self_as_replica: bool = True,
        listen_port: int | None = None,
    ) -> None:
        """Build the :class:`MxWeightTransferEngine` for this worker.

        Args:
            host, port: ``modelexpress-server`` URL.
            rank_offset: orchestrator-assigned base rank for this pod;
                ``global_rank = rank_offset + self.device.index``.
            publish_self_as_replica: if True (default), after each
                successful receive this worker republishes itself as
                a source so newcomers can pull from it (tree fan-out).
            listen_port: optional explicit NIXL listen port; ``None``
                = auto.
        """
        from modelexpress.vllm_weight_transfer import MxInitInfo, MxWeightTransferEngine

        local_rank = self.device.index
        global_rank = rank_offset + local_rank
        inference_model_name = self.model_runner.model_config.model

        pin_ucx_rail(local_rank)

        self._engine = MxWeightTransferEngine(
            init_info=MxInitInfo(
                mx_server_url=f"{host}:{port}",
                model_name=inference_model_name,
                worker_rank=global_rank,
                agent_name=make_agent_name("inference", global_rank),
                device_id=local_rank,
                listen_port=listen_port,
                publish_self_as_replica=publish_self_as_replica,
            )
        )
        self._global_rank = global_rank
        logger.info(
            f"[mx_v2] init: rank={global_rank} model={inference_model_name} "
            f"publish_self_as_replica={publish_self_as_replica}"
        )

    # ------------------------------------------------------------------
    # Per-refit RPC
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights_via_mx_v2(
        self,
        step: int,
        *,
        compile_target_filter: list[str] | None = None,
        timeout_seconds: float = 300.0,
        same_rank_only: bool = True,
    ) -> dict[str, float | int | None]:
        """Pull version ``step`` of the weights from the catalog.

        Args:
            step: training-step counter; engine pulls sources with
                ``version >= step``.
            compile_target_filter: receiver-side Phase 3b filter.
                ``None`` (default) = back-compat, accept any layout.
                Set e.g. ``["cutlass_fp8"]`` or
                ``["cutlass_fp8", "hf_raw"]`` to refuse mismatches at
                discovery (no RDMA cycle spent on refusal).
            timeout_seconds: cap on the engine's per-receive RDMA wait.
            same_rank_only: enforce same-rank routing (required on
                GB200/EFA multi-NIC fabrics where rdma-0..3 are
                separate L3 subnets).

        Returns:
            Per-cycle metrics dict (bytes / Gbps / discovery_seconds /
            rdma_seconds) suitable for emission to dashboards.
        """
        from modelexpress.vllm_weight_transfer import MxUpdateInfo

        update_info = MxUpdateInfo(
            version=step,
            compile_target_filter=set(compile_target_filter) if compile_target_filter else None,
            target_tp_layout=None,  # matched-TP fast path; Phase 4 wire-up future
            timeout_seconds=timeout_seconds,
            same_rank_only=same_rank_only,
        )
        self._engine.receive_weights(update_info, load_weights=self._load_weights_batch)

        # Post-load housekeeping: same as PR #2389's path.
        torch.cuda.synchronize(self.device)
        update_mla_absorbed_weights(self.raw_model)

        # Surface the engine's metrics so the orchestrator / dashboards
        # can read per-cycle bandwidth + discovery latency without
        # parsing logs.
        stats = self._engine.last_transfer_stats
        metrics = {
            "step": step,
            "bytes_received": stats.bytes_received if stats else 0,
            "tensors_received": stats.tensors_received if stats else 0,
            "rdma_seconds": stats.elapsed_seconds if stats else 0.0,
            "bandwidth_gbps": stats.bandwidth_gbps if stats else 0.0,
            "discovery_seconds": self._engine.last_discovery_seconds,
            "source_worker_rank": stats.source_worker_rank if stats else None,
        }
        logger.info(
            f"[mx_v2] refit step={step} "
            f"bytes={metrics['bytes_received'] / 1e6:.1f}MB "
            f"rdma={metrics['rdma_seconds']:.3f}s "
            f"{metrics['bandwidth_gbps']:.1f}Gbps "
            f"from_rank={metrics['source_worker_rank']}"
        )
        return metrics

    # ------------------------------------------------------------------
    # vLLM load-weights bridge
    # ------------------------------------------------------------------

    def _load_weights_batch(self, batch: list[tuple[str, torch.Tensor]]) -> None:
        """Feed yielded ``(name, tensor)`` pairs through vLLM's load_weights.

        vLLM's :meth:`model.load_weights` handles HF→fused name remapping
        via ``stacked_params_mapping`` (e.g. ``q_proj|k_proj|v_proj →
        qkv_proj``), so this worker doesn't need to know about fusion —
        the engine yields HF-format names and vLLM does the rest.
        Matches the NemoRL v2 pattern + Anyscale's RDT pattern.
        """
        self.raw_model.load_weights(batch)
