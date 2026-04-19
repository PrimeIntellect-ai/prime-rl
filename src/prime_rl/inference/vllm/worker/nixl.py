"""vLLM worker extension that receives weight updates over NIXL.

Registers every vLLM parameter + weight-scale buffer with NIXL and publishes
``(ptr, nbytes, device)`` per tensor. The trainer builds its own remote
descriptors at the right byte offsets — this side doesn't need to know the
per-source layout, so one ``all_gather_obj`` round is enough.
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

        self._agent = NixlAgentWrapper(
            name=make_agent_name("inference", global_rank),
            local_rank=local_rank,
            backends=backends,
        )

        tensor_ptrs: dict[str, tuple[int, int, int]] = {}

        def _register(name: str, tensor: torch.Tensor) -> None:
            contig = tensor.contiguous()
            self._agent.register_tensor(contig)
            tensor_ptrs[name] = (contig.data_ptr(), contig.numel() * contig.element_size(), contig.get_device())

        for name, param in model.named_parameters():
            _register(name, param.data)
        for name, buf in model.named_buffers():
            if name in tensor_ptrs or not name.endswith("_weight_scale_inv"):
                continue
            _register(name, buf)

        expert_map = {k: v.cpu().tolist() for k, v in build_expert_map(model).items()}

        self._spg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=global_rank,
            world_size=full_world_size,
            store_timeout=timeout,
        )
        gathered = self._spg.all_gather_obj(
            {
                "role": "inference",
                "global_rank": global_rank,
                "agent_name": self._agent.name,
                "agent_metadata": self._agent.get_metadata(),
                "tensor_ptrs": tensor_ptrs,
                "expert_map": expert_map,
            }
        )
        for peer in gathered[:trainer_world_size]:
            self._agent.add_remote(peer["agent_metadata"])

        logger.info(
            f"NIXL transfer ready: registered {len(tensor_ptrs)} tensors, "
            f"added {trainer_world_size} trainer peers"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        if not hasattr(self, "_spg"):
            raise RuntimeError("NIXL transfer not initialized — call /init_nixl_transfer first")
        self._spg.barrier()
        update_mla_absorbed_weights(self._model)
