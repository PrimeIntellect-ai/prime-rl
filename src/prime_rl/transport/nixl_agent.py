"""Thin wrapper around the NIXL agent for prime-rl.

Exposes the minimum surface needed to *publish* a worker's NIXL identity
through Model Express:

* :class:`NixlAgentWrapper` — pin a tensor's memory for RDMA, get the
  agent's serialized metadata blob, build :class:`TensorDescriptor`
  protos referencing the registered memory.
* :func:`make_agent_name` — deterministic naming so peers can address
  each other after rendezvous.

The actual transport (``post_write``, ``wait``, prepped dlists,
``add_remote_agent``) lives with the trainer publisher / inference
receiver — kept out of here so this module stays small and isolated.

``nixl_cu13`` is imported lazily so the module loads on machines without
NIXL installed; only construction of :class:`NixlAgentWrapper` requires it.
"""

from __future__ import annotations

import socket
from typing import Sequence

from modelexpress import p2p_pb2
from torch import Tensor


class NixlAgentWrapper:
    """One per process. Owns a NIXL agent and its registered memory."""

    def __init__(self, name: str, backends: Sequence[str] = ("UCX",)) -> None:
        from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore

        self.name = name
        self.backends: list[str] = list(backends)
        self._agent = nixl_agent(name, nixl_agent_config(backends=self.backends))

    def register_tensor(self, tensor: Tensor) -> None:
        """Pin the tensor's device memory for RDMA. Idempotent per tensor."""
        self._agent.register_memory(tensor, backends=self.backends)

    def get_metadata(self) -> bytes:
        """Serialized agent metadata. A peer feeds these bytes into
        ``add_remote_agent`` to address this agent.
        """
        return self._agent.get_agent_metadata()

    def make_tensor_descriptor(self, name: str, tensor: Tensor) -> p2p_pb2.TensorDescriptor:
        """Build a :class:`p2p_pb2.TensorDescriptor` pointing at this agent's
        registered memory for ``tensor``. Caller must have already passed
        ``tensor`` through :meth:`register_tensor`.
        """
        return p2p_pb2.TensorDescriptor(
            name=name,
            addr=tensor.data_ptr(),
            size=tensor.numel() * tensor.element_size(),
            device_id=tensor.device.index if tensor.device.type == "cuda" else 0,
            dtype=str(tensor.dtype).removeprefix("torch."),
        )


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"
