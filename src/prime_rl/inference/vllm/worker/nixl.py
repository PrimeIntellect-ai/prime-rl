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

        logger.info(
            f"Initializing NIXL transfer: local_rank={local_rank} rank_offset={rank_offset} "
            f"global_rank={global_rank} trainer_ws={trainer_world_size} inference_ws={inference_world_size}"
        )

        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        self._model = model

        logger.info(f"[nixl inference rank={global_rank}] creating NIXL agent")
        self._agent = NixlAgentWrapper(
            name=make_agent_name("inference", global_rank),
            local_rank=local_rank,
            backends=backends,
        )
        logger.info(f"[nixl inference rank={global_rank}] agent created; registering tensors")

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
        logger.info(
            f"[nixl inference rank={global_rank}] gathered {len(named_tensors)} tensor refs; "
            f"SPG created; round 1 all_gather_obj"
        )

        # Round 1 — pull trainer's layout. Agent metadata is deferred to round 2
        # so the metadata trainer sees already includes every chunk registration.
        round1 = self._spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "expert_map": expert_map,
            }
        )
        trainer_infos = round1[:trainer_world_size]
        layout: dict = trainer_infos[0]["non_expert_layout"]

        logger.info(f"[nixl inference rank={global_rank}] round 1 done; building chunked descriptors")

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

        for layer_layout in layout.values():
            for slot_key, info in layer_layout.items():
                full = named_tensors[info["inference_name"]]
                subview = full.narrow(0, info["offset_rows"], info["rows"])
                n_chunks = trainer_world_size if info["handling"] == "per_shard" else 1
                sub_rows = info["rows"] // n_chunks
                chunks = [subview.narrow(0, i * sub_rows, sub_rows) for i in range(n_chunks)]
                descriptors[slot_key] = _publish_chunks(chunks)

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
            f"NIXL transfer ready: rank={global_rank} published {len(descriptors)} slot descriptors, "
            f"added {trainer_world_size} trainer peers"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        if not hasattr(self, "_spg"):
            raise RuntimeError("NIXL transfer not initialized — call /init_nixl_transfer first")
        rank = getattr(self._agent, "name", "?")
        logger.info(f"[nixl inference {rank}] entering SPG barrier in update_weights_from_path")
        import time
        t0 = time.perf_counter()
        self._spg.barrier()
        logger.info(f"[nixl inference {rank}] left SPG barrier in {(time.perf_counter() - t0) * 1e3:.1f}ms")
        update_mla_absorbed_weights(self._model)
        logger.info(f"[nixl inference {rank}] update_mla_absorbed_weights done")
