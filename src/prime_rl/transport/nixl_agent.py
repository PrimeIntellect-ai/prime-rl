"""Thin wrapper around the NIXL agent for prime-rl.

Covers the agent lifecycle (register tensors, get serialized metadata,
build :class:`p2p_pb2.TensorDescriptor` protos for MX) and the RDMA
primitives used by :class:`prime_rl.transport.transport_plan.TransportPlan`
(``add_remote_agent``, sub-range ``prep_xfer_dlist``, WRITE post, busy-wait).

``nixl_cu13`` is imported lazily so the module loads on machines without
NIXL installed; only construction of :class:`NixlAgentWrapper` requires it.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Sequence

from modelexpress import p2p_pb2
from torch import Tensor


class NixlAgentWrapper:
    """One per process. Owns a NIXL agent and its registered memory."""

    def __init__(self, name: str, backends: Sequence[str] = ("UCX",)) -> None:
        from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore

        self.name = name
        self.backends: list[str] = list(backends)
        self._agent = nixl_agent(name, nixl_agent_config(backends=self.backends))

    # --- registration / metadata -------------------------------------------- #

    def register_tensor(self, tensor: Tensor) -> None:
        """Pin the tensor's device memory for RDMA. Idempotent per tensor."""
        self._agent.register_memory(tensor, backends=self.backends)

    def get_metadata(self) -> bytes:
        """Serialized agent metadata. A peer feeds these bytes into
        :meth:`add_remote_agent` to address this agent.
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

    # --- transport primitives ---------------------------------------------- #

    def add_remote_agent(self, peer_metadata: bytes) -> str:
        """Import a peer's serialized agent metadata. Returns the peer's agent name."""
        return self._agent.add_remote_agent(peer_metadata)

    def prep_local(self, descs: Sequence[tuple[int, int, int]]) -> Any:
        """Prepare a local-side dlist (no peer binding).

        Each entry is a ``(addr, size, device_id)`` triple within memory
        already registered on this agent.
        """
        return self._agent.prep_xfer_dlist(
            agent_name="", xfer_list=list(descs), mem_type="cuda", backends=self.backends
        )

    def prep_remote(self, peer_name: str, descs: Sequence[tuple[int, int, int]]) -> Any:
        """Prepare a remote-side dlist bound to ``peer_name``.

        ``peer_name`` must have been imported via :meth:`add_remote_agent`;
        each entry's ``(addr, size, device_id)`` must fall within an MR the
        peer registered.
        """
        return self._agent.prep_xfer_dlist(
            agent_name=peer_name, xfer_list=list(descs), mem_type="cuda", backends=self.backends
        )

    def post_write(self, *, local_prep: Any, local_idx: int, remote_prep: Any, remote_idx: int) -> Any:
        """Post a single WRITE: local chunk ``local_idx`` → remote chunk ``remote_idx``."""
        handle = self._agent.make_prepped_xfer(
            operation="WRITE",
            local_xfer_side=local_prep,
            local_indices=[local_idx],
            remote_xfer_side=remote_prep,
            remote_indices=[remote_idx],
            backends=self.backends,
        )
        state = self._agent.transfer(handle)
        if state in ("ERR", "ERROR", "FAIL"):
            raise RuntimeError(f"nixl WRITE post returned state {state}")
        return handle

    def wait(self, handle: Any, *, context: str = "") -> None:
        """Busy-poll a transfer handle to completion. Raises on error states."""
        while True:
            state = self._agent.check_xfer_state(handle)
            if state in ("DONE", "SUCCESS"):
                self._agent.release_xfer_handle(handle)
                return
            if state in ("ERR", "ERROR", "FAIL"):
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"nixl transfer ended state={state} context={context!r}")
            time.sleep(0.0005)


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"
