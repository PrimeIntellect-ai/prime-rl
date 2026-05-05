"""Broadcast weights into the inference engine via NIXL + Model Express.

Thin lifecycle wrapper around :class:`TrainerPublisher` +
:class:`TransportPlan`. Slot allocation and the MX rendezvous are deferred
to the first :meth:`broadcast_weights` call because the trainer model is
not available at ``setup_weight_broadcast`` time.

HSDP: when ``dp_replicate > 1`` only the primary replica (``dp_replicate
rank 0``) participates. Non-primary replicas hold bit-identical weights
so a second copy over the wire would be pure waste; they barrier-sync
to stay in lockstep.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import msgspec
import torch
import torch.distributed as dist
import torch.nn as nn
from modelexpress import MxClient, p2p_pb2
from transformers import AutoConfig

from prime_rl.configs.trainer import NIXLMxWeightBroadcastConfig
from prime_rl.trainer.models import PreTrainedModelPrimeRL
from prime_rl.trainer.models.conversions import select_default_conversion
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.trainer.rl.broadcast.base import WeightBroadcast
from prime_rl.trainer.runs import get_multi_run_manager
from prime_rl.trainer.utils import get_world
from prime_rl.transport.classic_cuda_pool import classic_cuda_alloc
from prime_rl.transport.mx_rendezvous import MxRendezvous
from prime_rl.transport.nixl_agent import NixlAgentWrapper, make_agent_name, pin_ucx_rail
from prime_rl.transport.transport_plan import TransportPlan
from prime_rl.transport.wire import RendezvousPayload


class NIXLMxWeightBroadcast(WeightBroadcast):
    """Broadcast weights into the inference engine via NIXL (zero-copy RDMA)."""

    def __init__(
        self,
        output_dir: Path,
        config: NIXLMxWeightBroadcastConfig,
        parallel_dims: ParallelDims,
    ) -> None:
        super().__init__(output_dir)
        self.config = config
        self.world = get_world()
        self.parallel_dims = parallel_dims

        if self.is_primary_hsdp_rank:
            pin_ucx_rail(torch.cuda.current_device())
            self.nixl_agent = NixlAgentWrapper(name=make_agent_name("trainer", self.world.rank))

        self.is_initialized = False

        self._multi_run_manager = get_multi_run_manager()
        self._flush_every = 100

    @property
    def is_primary_hsdp_rank(self) -> bool:
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh("dp_replicate").get_local_rank() == 0
        else:
            return True

    def register_slot_buffers_with_nixl(self) -> None:
        for slot in self.model_slots:
            for _, tensor, _ in slot.buffers:
                self.nixl_agent.register_tensor(tensor)

    def publish_metadata(self) -> None:
        """This method creates a list of tensor descriptors for each slot buffer and publishes them to the rendezvous."""
        descriptors: list[p2p_pb2.TensorDescriptor] = []
        for slot in self.model_slots:
            for buf_key, tensor, _ in slot.buffers:
                descriptors.append(self.nixl_agent.make_tensor_descriptor(buf_key, tensor))

        layout = []
        for slot in self.model_slots:
            layout.extend(slot.layout_payload())

        payload = RendezvousPayload(
            agent_metadata=self.nixl_agent.get_metadata(),
            agent_name=self.nixl_agent.name,
            layout=layout,
        )
        self.rendezvous.publish(
            nixl_metadata=msgspec.msgpack.encode(payload),
            tensors=descriptors,
        )

    def get_worker_metadata(self) -> list[p2p_pb2.WorkerMetadata]:
        peer_refs = self.rendezvous.wait_for_peers(
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=self.config.timeout,
            poll_interval=1.0,
        )

        return [self.rendezvous.fetch_peer(ref) for ref in peer_refs]

    def lazy_init(self, model: PreTrainedModelPrimeRL) -> None:
        """Build publisher + transport plan on first call (needs the live model)."""
        if self.is_initialized:
            return

        hf_config = AutoConfig.from_pretrained(self.config.inference_model_name)
        default_conversion = select_default_conversion(self.config.inference_model_name)

        with classic_cuda_alloc():
            self.model_slots = model.build_slots(self.parallel_dims, default_conversion, hf_config.torch_dtype)

        self.rendezvous = MxRendezvous(
            client=MxClient(server_url=f"{self.config.host}:{self.config.port}"),
            role="trainer",
            rank=self.world.rank,
            peer_world_size=self.config.inference_world_size,
            model_name=self.config.inference_model_name,
        )
        self.register_slot_buffers_with_nixl()
        self.publish_metadata()

        self.transport_plan = TransportPlan(
            agent=self.nixl_agent, peer_metadata=self.get_worker_metadata(), slots=self.model_slots
        )

        self.is_initialized = True

    def drain(self, handles: list[tuple[Any, str]]) -> None:
        if not handles:
            return
        for h, tag in handles:
            self.nixl_agent.wait(h, context=tag)

    @torch.no_grad()
    def broadcast_weights(self, model: nn.Module, step: int) -> None:
        if self.is_primary_hsdp_rank:
            # Try to initialize the transport plan if we haven't already, signal the orchestrator that we are ready to push by setting the status to INITIALIZING
            self.lazy_init(model)
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_INITIALIZING)

        if self.world.is_master:
            for idx in self._multi_run_manager.used_idxs:
                if self._multi_run_manager.ready_to_update[idx]:
                    self._multi_run_manager.ready_to_update[idx] = False
            self.rendezvous.wait_for_all_peers_ready(role="orchestrator", timeout=self.config.timeout)

        dist.barrier()

        if self.is_primary_hsdp_rank:
            start = time.perf_counter()

            handles = []
            for local_prep, remote_prep, entry in self.transport_plan.prepare_writes(
                model.state_dict(), self.model_slots
            ):
                # Post the write to the NIXL agent
                handle = self.nixl_agent.post_write(
                    local_prep=local_prep,
                    local_idx=entry.local_chunk_idx,
                    remote_prep=remote_prep,
                    remote_idx=entry.remote_chunk_idx,
                )
                handles.append((handle, entry.tag))
                if len(handles) % self._flush_every == 0:
                    self.drain(handles)
                    handles.clear()

            self.drain(handles)

            # Signal the orchestrator that we are ready to push by setting the status to READY
            self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
            self.logger.debug(f"NIXL+MX push completed in {time.perf_counter() - start:.2f}s")

        dist.barrier()
