"""vLLM worker extension that receives weight updates over NIXL.

Counterpart to :mod:`prime_rl.trainer.rl.broadcast.nixl`. The inference side
registers parameter memory directly with NIXL (zero-copy RDMA target),
publishes its expert-ownership map per FusedMoE module, and sits on a
single process-group barrier per sync while the trainer posts writes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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


def _iter_transfer_targets(model: Module):
    """Yield (name, tensor) for every parameter + weight-scale buffer we want to
    receive from the trainer.

    vLLM stores FP8 scales as buffers (``w13_weight_scale_inv`` / ``w2_weight_scale_inv``),
    not parameters, so ``named_parameters()`` alone is insufficient.
    """
    seen: set[str] = set()
    for name, param in model.named_parameters():
        seen.add(name)
        yield name, param.data
    for name, buf in model.named_buffers():
        if name in seen:
            continue
        # Only ship weight scales — other buffers (rotary embeddings, caches) are not
        # synchronized from the trainer.
        if name.endswith("_weight_scale_inv"):
            yield name, buf


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
        """Register local parameter memory and rendezvous with the trainer."""
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

        # Register every receivable tensor and record its serialized descriptor so the
        # trainer can deserialize it on the other side.
        descriptors: dict[str, bytes] = {}
        for name, tensor in _iter_transfer_targets(model):
            desc = self._agent.register_tensor(tensor.contiguous())
            descriptors[name] = self._agent.serialize_descs(desc)

        # Expert ownership per FusedMoE module. Re-use the existing helper — each entry
        # is a tensor of global expert indices that this worker holds (sorted by local slot).
        expert_map = {k: v.cpu().tolist() for k, v in build_expert_map(model).items()}

        self._spg = StatelessProcessGroup.create(
            host=host,
            port=port,
            rank=global_rank,
            world_size=full_world_size,
            store_timeout=timeout,
        )

        my_info = {
            "role": "inference",
            "global_rank": global_rank,
            "agent_name": self._agent.name,
            "agent_metadata": self._agent.get_metadata(),
            "descriptors": descriptors,
            "expert_map": expert_map,
        }
        all_info: list[dict[str, Any]] = self._spg.all_gather_obj(my_info)

        # Add every trainer agent so future WRITEs from them can land here.
        for peer in all_info[:trainer_world_size]:
            self._agent.add_remote(peer["agent_metadata"])

        logger.info(
            f"NIXL transfer ready: registered {len(descriptors)} tensors, "
            f"added {trainer_world_size} trainer peers"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        """Receive one round of NIXL writes and repost-process the model.

        The actual data movement is driven entirely by the trainer: writes land
        directly in the already-registered parameter memory. We only need to
        wait on the end-of-sync barrier and recompute MLA absorbed weights.
        """
        if not hasattr(self, "_spg"):
            raise RuntimeError("NIXL transfer not initialized — call /init_nixl_transfer first")
        logger.debug("Waiting for NIXL end-of-sync barrier")
        self._spg.barrier()
        logger.debug("NIXL writes complete, running postprocess")
        update_mla_absorbed_weights(self._model)
