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
        g_name = "model.layers.3.input_layernorm.weight"
        f_name = "model.layers.3.self_attn.kv_b_proj.weight"
        f_scale_name = "model.layers.3.self_attn.kv_b_proj.weight_scale_inv"
        e_name = "model.layers.3.mlp.experts.w13_weight"
        e_scale_name = "model.layers.3.mlp.experts.w13_weight_scale_inv"
        e_prefix = "model.layers.3.mlp.experts"
        # Import locally to avoid circular init on older worker vLLM builds.
        from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map
        expert_map = build_expert_map(self._model)

        if g_name in named_params:
            s = named_params[g_name].data.to(torch.float64).sum().item()
            logger.info(f"[nixl SIG inference] anchor=G key={g_name} sum={s:.8f}")
        if f_name in named_params:
            w = named_params[f_name].data.view(torch.uint8).to(torch.int64).sum().item()
            sc = named_bufs[f_scale_name].data.to(torch.float64).sum().item() if f_scale_name in named_bufs else 0.0
            logger.info(f"[nixl SIG inference] anchor=F key={f_name} w_bytes={w} scale={sc:.8f}")
        if e_name in named_params and e_prefix in expert_map:
            owned = expert_map[e_prefix].cpu().tolist()
            if 0 in owned:
                local_idx = owned.index(0)
                w0 = named_params[e_name].data[local_idx].view(torch.uint8).to(torch.int64).sum().item()
                sc0 = named_bufs[e_scale_name].data[local_idx].to(torch.float64).sum().item() if e_scale_name in named_bufs else 0.0
                logger.info(f"[nixl SIG inference] anchor=E key={e_name}[E0] w_bytes={w0} scale={sc0:.8f}")

        update_mla_absorbed_weights(self._model)
