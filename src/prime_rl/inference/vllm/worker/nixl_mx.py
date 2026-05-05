from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from torch.nn import Module
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import build_expert_map, update_mla_absorbed_weights
from prime_rl.transport.mx_rendezvous import MxRendezvous
from prime_rl.transport.nixl_agent import NixlAgentWrapper, make_agent_name, pin_ucx_rail
from prime_rl.transport.wire import RendezvousPayload

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker  # type: ignore
else:
    Worker = object  # type: ignore

logger = init_logger("vllm.inference.vllm.worker_nixl_mx")


class NIXLMxWeightUpdateWorker(Worker):
    """vLLM worker extension for in-place weight updates over NIXL + MX."""

    @property
    def raw_model(self) -> Module:
        model_runner = self.model_runner
        model = model_runner.model.runnable if hasattr(model_runner.model, "runnable") else model_runner.model
        assert isinstance(model, Module)
        return model

    def register_tensors_with_nixl(self, model: Module) -> None:
        self._descriptors: list[p2p_pb2.TensorDescriptor] = []
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

        for name, tensor in live_tensors.items():
            self.agent.register_tensor(tensor)
            self._descriptors.append(self.agent.make_tensor_descriptor(name, tensor))

    def init_nixl_mx(self, host: str, port: int, rank_offset: int) -> None:
        local_rank = self.device.index
        global_rank = rank_offset + local_rank
        inference_model_name = self.model_runner.model_config.model

        pin_ucx_rail(local_rank)
        self.agent = NixlAgentWrapper(name=make_agent_name("inference", global_rank))
        self.rendezvous = MxRendezvous(
            client=MxClient(server_url=f"{host}:{port}"),
            role="inference",
            rank=global_rank,
            peer_world_size=0,
            model_name=inference_model_name,
        )

        expert_map = {k: v.cpu().tolist() for k, v in build_expert_map(self.raw_model).items()}

        self.register_tensors_with_nixl(self.raw_model)
        payload = RendezvousPayload(
            agent_metadata=self.agent.get_metadata(),
            agent_name=self.agent.name,
            expert_map=expert_map,
        )
        self.rendezvous.publish(
            nixl_metadata=msgspec.msgpack.encode(payload),
            tensors=self._descriptors,
        )
        self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)

        logger.info(
            f"NIXL+MX init: rank={global_rank} tensors={len(self._descriptors)} "
            f"experts={sum(len(v) for v in expert_map.values())} model={inference_model_name}"
        )

    @torch.no_grad()
    def update_weights_from_path(self, weight_dir: str | None = None) -> None:
        """Block until the trainer's RDMA push completes, then recompute the MLA absorbed weights and return, orchestrator can then call `/resume`"""
        self.rendezvous.wait_for_all_peers_ready(timeout=1200)
        torch.cuda.synchronize(self.device)
        update_mla_absorbed_weights(self.raw_model)
        logger.info("Weight update applied (NIXL+MX)")
