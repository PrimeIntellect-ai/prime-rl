"""Inference-side per-rank receiver: register vLLM live params with NIXL,
publish the agent metadata + tensor descriptors + ``expert_map`` through
Model Express, then mark itself ``READY``.

Setup-time only — does not perform any transfer. The receiver doesn't
allocate buffers and doesn't apply conversions; it pins whatever vLLM
already loaded.

Diverges from upstream:
* Standalone class, not a vLLM worker extension. The vLLM worker plugin
  (later step) becomes a thin wrapper that iterates ``named_parameters``
  and ``_weight_scale_inv`` buffers, calls ``build_expert_map`` on the
  vLLM model, then constructs an :class:`InferenceReceiver` with those
  inputs.
* Single publish per side instead of upstream's two SPG rounds. The
  trainer's pass-2 chunked-descriptor exchange isn't needed:
  empirically NIXL's xfer dlist is a pure ``(addr, size, dev)`` carrier
  with no agent-specific state — bytes from ``get_serialized_descs`` are
  byte-identical whether constructed by the local or remote agent, and
  ``prep_xfer_dlist`` accepts a locally-constructed dlist for peer
  addresses. So the trainer chunks remote dlists locally at RDMA-prep
  time using each peer's tensor base addresses (from
  :attr:`p2p_pb2.WorkerMetadata.tensors`) plus its own layout. The
  ``READY`` lifecycle flag on MX (:meth:`MxRendezvous.set_status`)
  carries the synchronization signal that pass 2 was implicitly
  providing.
"""

from __future__ import annotations

import msgspec
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from torch import Tensor

from prime_rl.transport.mx_rendezvous import MxRendezvous
from prime_rl.transport.nixl_agent import NixlAgentWrapper, make_agent_name
from prime_rl.transport.wire import RendezvousPayload


class InferenceReceiver:
    """One inference rank's NIXL registrations + MX rendezvous handle."""

    def __init__(
        self,
        *,
        client: MxClient,
        rank: int,
        peer_world_size: int,
        inference_model_name: str,
        live_tensors: dict[str, Tensor],
        expert_map: dict[str, list[int]],
        quantization: str = "",
        expert_parallel_size: int = 0,
    ) -> None:
        self.expert_map = expert_map
        self.live_tensors = live_tensors
        self.agent = NixlAgentWrapper(name=make_agent_name("inference", rank))
        self._descriptors: list[p2p_pb2.TensorDescriptor] = []
        for name, tensor in live_tensors.items():
            self.agent.register_tensor(tensor)
            self._descriptors.append(self.agent.make_tensor_descriptor(name, tensor))
        self.rendezvous = MxRendezvous(
            client=client,
            role="inference",
            rank=rank,
            peer_world_size=peer_world_size,
            model_name=inference_model_name,
            expert_parallel_size=expert_parallel_size,
            quantization=quantization,
        )

    def publish(self) -> str:
        """Push agent metadata + tensor descriptors + expert_map through MX. Returns ``mx_source_id``."""
        payload = RendezvousPayload(
            agent_metadata=self.agent.get_metadata(),
            agent_name=self.agent.name,
            expert_map=self.expert_map,
        )
        return self.rendezvous.publish(
            nixl_metadata=msgspec.msgpack.encode(payload),
            tensors=self._descriptors,
        )

    def mark_ready(self) -> None:
        """Flip lifecycle to ``READY`` so the trainer's
        :meth:`MxRendezvous.wait_for_peers` (status-filtered) unblocks.
        """
        self.rendezvous.set_status(p2p_pb2.SOURCE_STATUS_READY)
