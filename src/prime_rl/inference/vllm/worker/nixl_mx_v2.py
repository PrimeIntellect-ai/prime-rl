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

        # Cache the HF model config + parallel layout so the TT→HF
        # translator (`_translate_tt_to_hf`) can split fused tensors into
        # the per-tensor / per-expert names vLLM's `load_weights` expects.
        try:
            from transformers import AutoConfig
            hf = AutoConfig.from_pretrained(inference_model_name)
            mc = self.model_runner.model_config
            ep_size = getattr(mc, "ep_size", None) or getattr(
                mc, "data_parallel_size", 1
            )
            self._hf_config = {
                "model_type": getattr(hf, "model_type", ""),
                "num_attention_heads": getattr(hf, "num_attention_heads", 0),
                "num_kv_heads": getattr(hf, "num_key_value_heads", 0)
                or getattr(hf, "num_attention_heads", 0),
                "head_dim": getattr(hf, "head_dim", 0)
                or (
                    getattr(hf, "hidden_size", 0)
                    // max(1, getattr(hf, "num_attention_heads", 1))
                ),
                "num_experts": getattr(hf, "num_experts", 0)
                or getattr(hf, "num_local_experts", 0),
                "ep_size": int(ep_size or 1),
            }
        except Exception as e:  # noqa: BLE001 — never block engine init
            logger.warning(
                f"[mx_v2] HF config probe failed ({e!r}); TT→HF translator "
                f"will fall through to passthrough — non-MoE models only."
            )
            self._hf_config = None

        logger.info(
            f"[mx_v2] init: rank={global_rank} model={inference_model_name} "
            f"publish_self_as_replica={publish_self_as_replica} "
            f"hf_config={self._hf_config}"
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
        import time as _time

        from modelexpress.vllm_weight_transfer import MxUpdateInfo

        update_info = MxUpdateInfo(
            version=step,
            compile_target_filter=set(compile_target_filter) if compile_target_filter else None,
            target_tp_layout=None,  # matched-TP fast path; Phase 4 wire-up future
            timeout_seconds=timeout_seconds,
            same_rank_only=same_rank_only,
        )

        # Async-RL synchronization: orchestrator polls /update_weights_v2 with
        # step=N right after a training cycle, but the trainer publishes
        # version=N asynchronously (it has to finish optimizer.step + the
        # publisher's add_tensor loop). If the engine's discovery fires
        # before the trainer has marked version=N READY in the MX catalog,
        # `receive_weights` raises `no source matches filters`.
        #
        # Wrap the engine call in a bounded retry loop so the synchronization
        # gap is absorbed at the worker layer (no orchestrator changes needed
        # and the failure surface stays at this layer's known timeout).
        retry_deadline = _time.monotonic() + timeout_seconds
        backoff = 0.5
        attempts = 0
        last_err: Exception | None = None
        while True:
            attempts += 1
            try:
                self._engine.receive_weights(
                    update_info, load_weights=self._load_weights_batch
                )
                break
            except Exception as e:  # noqa: BLE001 — engine may raise plain RuntimeError
                msg = str(e)
                last_err = e
                # Only retry on "no source matches" / discovery-empty errors;
                # propagate any other (e.g. NIXL transport failure) immediately.
                transient = (
                    "no source matches" in msg
                    or "NoSourceMatchesFilterError" in msg
                    or "no matching source" in msg
                )
                if not transient or _time.monotonic() >= retry_deadline:
                    raise
                logger.info(
                    f"[mx_v2] receive_weights attempt #{attempts} for step={step}: "
                    f"transient miss ({msg[:80]!r}); retrying in {backoff:.1f}s"
                )
                _time.sleep(backoff)
                backoff = min(backoff * 1.6, 8.0)

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

        Translation pass: PrimeRL's trainer-side ``GatheredSlot`` emits
        tensors in TT-format (fused ``qkv_proj``, stacked-expert
        ``w13_weight``/``w2_weight``, ``mlp.router.gate`` prefix). vLLM's
        ``load_weights`` expects HF-checkpoint names + per-expert tensors
        so its ``stacked_params_mapping`` (QKV / gate-up) and
        ``expert_params_mapping`` (FusedMoE) can route them into the
        model's actual stacked params. We translate TT → HF here so the
        engine adapter (``MxWeightTransferEngine``) stays model-agnostic.

        The slot-side conversion specs that PrimeRL applies on the
        publisher side are the inverse of this translator — see
        ``prime_rl.trainer.models.qwen3_moe.converting_qwen3_moe``.
        """
        translated = self._translate_tt_to_hf(batch)
        if translated:
            self.raw_model.load_weights(translated)

    # ------------------------------------------------------------------
    # TT → HF translation
    # ------------------------------------------------------------------

    def _translate_tt_to_hf(
        self,
        batch: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Translate PrimeRL TT-format slot keys to HF checkpoint names.

        Currently supports Qwen3-MoE family (Qwen3MoeForCausalLM); other
        models pass through (most non-MoE PrimeRL models already match
        HF naming). To extend, add per-prefix unstacking logic.

        Layout assumption: the per-trainer-rank expert subset matches the
        per-inference-rank EP subset (i.e. ``trainer.ep == inference.EP``),
        so local-expert index lines up with global expert ID via
        ``my_rank * num_local + local_id``. Cross-EP slicing (Phase 4
        mixed-TP / multi-source picker) is the follow-up that lifts this
        constraint.
        """
        cfg = self._hf_config
        if cfg is None or cfg.get("model_type") not in {"qwen3_moe", "qwen3"}:
            return batch  # passthrough for unsupported models

        q_size = cfg["num_attention_heads"] * cfg["head_dim"]
        kv_size = cfg["num_kv_heads"] * cfg["head_dim"]
        num_experts = cfg.get("num_experts", 0)
        ep_size = max(1, cfg.get("ep_size", 1))
        num_local_experts = num_experts // ep_size if num_experts else 0
        my_rank = self._global_rank % ep_size if ep_size > 1 else 0

        out: list[tuple[str, torch.Tensor]] = []
        for name, tensor in batch:
            # ── QKV split (fused → q/k/v) ───────────────────────────────
            if name.endswith(".self_attn.qkv_proj.weight"):
                prefix = name.removesuffix(".self_attn.qkv_proj.weight")
                expected = q_size + 2 * kv_size
                assert tensor.shape[0] == expected, (
                    f"qkv_proj rows {tensor.shape[0]} != "
                    f"q({q_size})+k({kv_size})+v({kv_size})={expected}"
                )
                out.append((f"{prefix}.self_attn.q_proj.weight", tensor[:q_size]))
                out.append((f"{prefix}.self_attn.k_proj.weight", tensor[q_size : q_size + kv_size]))
                out.append((f"{prefix}.self_attn.v_proj.weight", tensor[q_size + kv_size :]))

            # ── Dense MLP gate/up split (future-proof, no-op on Qwen3-30B-A3B)
            elif name.endswith(".mlp.gate_up_proj.weight"):
                prefix = name.removesuffix(".mlp.gate_up_proj.weight")
                mid = tensor.shape[0] // 2
                out.append((f"{prefix}.mlp.gate_proj.weight", tensor[:mid]))
                out.append((f"{prefix}.mlp.up_proj.weight", tensor[mid:]))

            # ── Router rename (TT prefix → HF) ──────────────────────────
            elif name.endswith(".mlp.router.gate.weight"):
                prefix = name.removesuffix(".mlp.router.gate.weight")
                out.append((f"{prefix}.mlp.gate.weight", tensor))

            # ── MoE w13 (fused gate+up, stacked across local experts) ───
            elif name.endswith(".mlp.experts.w13_weight"):
                prefix = name.removesuffix(".mlp.experts.w13_weight")
                if tensor.ndim != 3:
                    out.append((name, tensor))
                    continue
                n_local, fused_dim, _ = tensor.shape
                moe_dim = fused_dim // 2
                for j in range(n_local):
                    global_id = my_rank * num_local_experts + j
                    out.append(
                        (
                            f"{prefix}.mlp.experts.{global_id}.gate_proj.weight",
                            tensor[j, :moe_dim].contiguous(),
                        )
                    )
                    out.append(
                        (
                            f"{prefix}.mlp.experts.{global_id}.up_proj.weight",
                            tensor[j, moe_dim:].contiguous(),
                        )
                    )

            # ── MoE w2 (down, stacked across local experts) ─────────────
            elif name.endswith(".mlp.experts.w2_weight"):
                prefix = name.removesuffix(".mlp.experts.w2_weight")
                if tensor.ndim != 3:
                    out.append((name, tensor))
                    continue
                n_local = tensor.shape[0]
                for j in range(n_local):
                    global_id = my_rank * num_local_experts + j
                    out.append(
                        (
                            f"{prefix}.mlp.experts.{global_id}.down_proj.weight",
                            tensor[j].contiguous(),
                        )
                    )

            # ── Passthrough: norms, o_proj, q/k_norm, embed, lm_head ────
            else:
                out.append((name, tensor))

        return out
