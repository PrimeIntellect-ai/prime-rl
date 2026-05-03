"""vLLM worker extension that receives weight updates over NIXL + Model Express.

Registers every vLLM parameter and ``_weight_scale_inv`` buffer with NIXL,
publishes the agent's metadata and tensor descriptors through MX, then
flips to ``READY``. The trainer posts RDMA WRITEs directly into these
registered buffers; by the time the orchestrator calls
:meth:`update_weights_from_path`, the writes have already landed — the
worker just needs to flush GPUDirect RDMA writes and run post-load
housekeeping (e.g. recompute MLA absorbed weights).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Module
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map, update_mla_absorbed_weights
from prime_rl.transport.inference_receiver import InferenceReceiver
from prime_rl.transport.nixl_agent import pin_ucx_rail

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker  # type: ignore
else:
    Worker = object  # type: ignore

logger = init_logger("vllm.inference.vllm.worker_nixl_mx")


class NIXLMxWeightUpdateWorker(Worker):
    """vLLM worker extension for in-place weight updates over NIXL + MX."""

    def init_nixl_mx(self, host: str, port: int, rank_offset: int) -> None:
        local_rank = self.device.index
        global_rank = rank_offset + local_rank

        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        self._model = model

        live_tensors: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if not param.is_contiguous():
                raise RuntimeError(f"non-contiguous param {name} cannot be NIXL-registered")
            live_tensors[name] = param.data
        for name, buf in model.named_buffers():
            if name in live_tensors or not name.endswith("_weight_scale_inv"):
                continue
            if not buf.is_contiguous():
                raise RuntimeError(f"non-contiguous buffer {name} cannot be NIXL-registered")
            live_tensors[name] = buf

        expert_map = {k: v.cpu().tolist() for k, v in build_expert_map(model).items()}

        inference_model_name = model_runner.model_config.model

        pin_ucx_rail(local_rank)

        from modelexpress.client import MxClient

        client = MxClient(server_url=f"{host}:{port}")
        self._receiver = InferenceReceiver(
            client=client,
            rank=global_rank,
            peer_world_size=0,
            inference_model_name=inference_model_name,
            live_tensors=live_tensors,
            expert_map=expert_map,
        )
        self._receiver.publish()
        self._receiver.mark_ready()

        logger.info(
            f"NIXL+MX init: rank={global_rank} tensors={len(live_tensors)} "
            f"experts={sum(len(v) for v in expert_map.values())} model={inference_model_name}"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        """Block until the trainer's RDMA push completes, then flush and recompute.

        The trainer flips MX status from INITIALIZING → READY after
        ``push_once`` completes. We poll for that transition here so the
        orchestrator's HTTP call blocks until the data has actually
        landed — without this, the orchestrator would resume inference
        on partially-written buffers.
        """
        self._receiver.rendezvous.wait_for_all_peers_ready(timeout=1200)
        torch.cuda.synchronize(self.device)
        update_mla_absorbed_weights(self._model)
        logger.info("Weight update applied (NIXL+MX)")
