"""vLLM worker extension that receives weight updates over NIXL.

Registers every vLLM parameter + weight-scale buffer with NIXL and publishes
per-slot chunked descriptors that match the trainer's write layout. Uses two
``all_gather_obj`` rounds so the inference side has the trainer's layout
before it has to build the chunked descriptors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map, update_mla_absorbed_weights
from prime_rl.trainer.models.slots import LayoutEntry
from prime_rl.utils.nixl_transfer import NixlAgentWrapper, make_agent_name

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker  # noqa: F401

    Worker = Worker  # type: ignore
else:
    Worker = object  # type: ignore

logger = init_logger("vllm.inference.vllm.worker_nixl")


class NIXLWeightUpdateWorker(Worker):
    """vLLM worker extension for in-place weight updates over NIXL."""

    def init_nixl_transfer(
        self,
        host: str,
        port: int,
        rank_offset: int,
        trainer_world_size: int,
        inference_world_size: int,
        timeout: int,
        backends: list[str],
    ) -> None:
        local_rank = self.device.index
        global_rank = trainer_world_size + rank_offset + local_rank
        full_world_size = trainer_world_size + inference_world_size

        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        self._model = model

        self._agent = NixlAgentWrapper(
            name=make_agent_name("inference", global_rank),
            local_rank=local_rank,
            backends=backends,
        )

        named_tensors: dict[str, torch.Tensor] = {}
        # Register each vLLM target tensor as a single NIXL region (one MR on the
        # NIC). Sub-range xfer descriptors at post time resolve to this MR's rkey —
        # overlapping per-chunk MRs confuse mlx5 rkey lookup and trigger local
        # protection errors on WRITE landing.
        for name, param in model.named_parameters():
            t = param.data.contiguous()
            named_tensors[name] = t
            self._agent.register_tensor(t)
        for name, buf in model.named_buffers():
            if name in named_tensors or not name.endswith("_weight_scale_inv"):
                continue
            t = buf.contiguous()
            named_tensors[name] = t
            self._agent.register_tensor(t)

        expert_map = {k: v.cpu().tolist() for k, v in build_expert_map(model).items()}

        self._spg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=global_rank,
            world_size=full_world_size,
            store_timeout=timeout,
        )

        # Round 1 — pull trainer's flat LayoutEntry list. Agent metadata is
        # deferred to round 2 so the metadata trainer sees already includes
        # every chunk registration.
        round1 = self._spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "expert_map": expert_map,
            }
        )
        layout_entries: list[LayoutEntry] = round1[0]["layout_entries"]

        # For each slot, publish a LIST of serialized 1-entry xfer dlists, each
        # covering one chunk's sub-range within the (already-registered) full
        # tensor. Trainer picks the right index per write.
        descriptors: dict[str, list[bytes]] = {}

        def _publish_chunks(chunks: list[torch.Tensor]) -> list[bytes]:
            out: list[bytes] = []
            for c in chunks:
                dlist = self._agent._agent.get_xfer_descs(
                    [(c.data_ptr(), c.numel() * c.element_size(), c.get_device())],
                    mem_type="cuda",
                )
                out.append(self._agent.serialize_descs(dlist))
            return out

        for entry in layout_entries:
            full = named_tensors[entry.inference_name]
            subview = full.narrow(0, entry.offset_rows, entry.rows)
            sub_rows = entry.rows // entry.num_chunks
            chunks = [subview.narrow(0, i * sub_rows, sub_rows) for i in range(entry.num_chunks)]
            descriptors[entry.slot_key] = _publish_chunks(chunks)

        for name, tensor in named_tensors.items():
            if ".mlp.experts." not in name:
                continue
            moe_prefix = name.rsplit(".", 1)[0]
            num_local = len(expert_map[moe_prefix])
            chunks = [tensor.narrow(0, e, 1) for e in range(num_local)]
            descriptors[name] = _publish_chunks(chunks)

        # Round 2 — publish agent metadata (now carrying all chunk registrations),
        # serialized descriptors, and expert map. Trainer picks these up.
        round2 = self._spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "agent_name": self._agent.name,
                "agent_metadata": self._agent.get_metadata(),
                "descriptors": descriptors,
                "expert_map": expert_map,
            }
        )
        for peer in round2[:trainer_world_size]:
            self._agent.add_remote(peer["agent_metadata"])
        logger.info(
            f"NIXL transfer ready: rank={global_rank} descriptors={len(descriptors)} trainers={trainer_world_size}"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        if not hasattr(self, "_spg"):
            raise RuntimeError("NIXL transfer not initialized — call /init_nixl_transfer first")
        self._spg.barrier()

        # Diagnostic: mirror the trainer-side SIG anchors.
        #   G (bf16 gather)   : input_layernorm.weight
        #   F (fp8 gather)    : self_attn.kv_b_proj.weight + scale
        #   E (fp8 expert)    : mlp.experts.w13_weight expert[0]
        named_params = dict(self._model.named_parameters())
        named_bufs = dict(self._model.named_buffers())

        def _lookup(name):
            """vLLM stores FP8 scales sometimes as params, sometimes as buffers."""
            if name in named_params:
                return named_params[name].data, "param"
            if name in named_bufs:
                return named_bufs[name].data, "buf"
            return None, "MISSING"

        g_name = "model.layers.3.input_layernorm.weight"
        f_name = "model.layers.3.self_attn.kv_b_proj.weight"
        f_scale_name = "model.layers.3.self_attn.kv_b_proj.weight_scale_inv"
        e_name = "model.layers.3.mlp.experts.w13_weight"
        e_scale_name = "model.layers.3.mlp.experts.w13_weight_scale_inv"
        e_prefix = "model.layers.3.mlp.experts"
        from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map
        expert_map = build_expert_map(self._model)

        fused_qkv_name = "model.layers.3.self_attn.fused_qkv_a_proj.weight"
        fused_qkv_scale_name = "model.layers.3.self_attn.fused_qkv_a_proj.weight_scale_inv"
        # F_q covers inference fused_qkv_a_proj[0:2048] → matches q_a_proj (row 0..2047).
        # F_kv covers inference fused_qkv_a_proj[2048:2624] → matches kv_a_proj_with_mqa (row 0..575).
        fq_t, fq_loc = _lookup(fused_qkv_name)
        fq_s, fq_s_loc = _lookup(fused_qkv_scale_name)
        if fq_t is not None:
            q_w_bytes = fq_t[:2048].contiguous().view(torch.uint8).to(torch.int64).sum().item()
            kv_w_bytes = fq_t[2048:2624].contiguous().view(torch.uint8).to(torch.int64).sum().item()
            q_s_sum = fq_s[:16].to(torch.float64).sum().item() if fq_s is not None else -1.0
            kv_s_sum = fq_s[16:21].to(torch.float64).sum().item() if fq_s is not None else -1.0
            logger.info(
                f"[nixl SIG inference] anchor=F_q loc={fq_loc}/{fq_s_loc} key={fused_qkv_name}[:2048] "
                f"w_bytes={q_w_bytes} w_shape={tuple(fq_t[:2048].shape)} scale={q_s_sum:.8f}"
            )
            logger.info(
                f"[nixl SIG inference] anchor=F_kv loc={fq_loc}/{fq_s_loc} key={fused_qkv_name}[2048:2624] "
                f"w_bytes={kv_w_bytes} w_shape={tuple(fq_t[2048:2624].shape)} scale={kv_s_sum:.8f}"
            )
        e_t, e_loc = _lookup(e_name)
        e_s, e_s_loc = _lookup(e_scale_name)
        if e_t is not None and e_prefix in expert_map:
            owned = expert_map[e_prefix].cpu().tolist()
            # Log global experts 0..3 that this worker owns (if any).
            for global_id in (0, 1, 2, 3):
                if global_id in owned:
                    local_idx = owned.index(global_id)
                    w_bytes = e_t[local_idx].view(torch.uint8).to(torch.int64).sum().item()
                    s_sum = e_s[local_idx].to(torch.float64).sum().item() if e_s is not None else -1.0
                    logger.info(
                        f"[nixl SIG inference] anchor=E[E{global_id}] loc={e_loc}/{e_s_loc} "
                        f"key={e_name}[local={local_idx}] "
                        f"w_bytes={w_bytes} scale={s_sum:.8f}"
                    )

        update_mla_absorbed_weights(self._model)
