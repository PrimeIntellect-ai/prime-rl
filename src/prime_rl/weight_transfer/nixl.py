"""Thin wrapper around the NIXL agent for prime-rl weight transfer.

Covers the agent lifecycle (register memory, serialized metadata exchange)
and the RDMA primitives used by the NIXL weight broadcast: import a remote
agent, prepare descriptor lists, post batched WRITEs, busy-wait completion.

``nixl`` is imported lazily so the module loads on machines without NIXL
installed; only construction of :class:`NixlAgent` requires it.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Sequence

from torch import Tensor

# (addr, num_bytes, device_id) within memory registered on the owning agent.
MemDesc = tuple[int, int, int]


class NixlAgent:
    """One per process. Owns a NIXL agent and its registered memory."""

    def __init__(self, name: str, backends: Sequence[str] = ("UCX",)) -> None:
        try:
            from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore
        except ImportError:
            from nixl._api import nixl_agent, nixl_agent_config  # type: ignore

        self.name = name
        self.backends: list[str] = list(backends)
        self._agent = nixl_agent(name, nixl_agent_config(backends=self.backends))

    # --- registration / metadata -------------------------------------------- #

    def register_tensor(self, tensor: Tensor) -> None:
        """Pin the tensor's device memory for RDMA. Idempotent per tensor."""
        self._agent.register_memory(tensor, backends=self.backends)

    def get_metadata(self) -> bytes:
        """Serialized agent metadata. A peer feeds these bytes into
        :meth:`add_remote_agent` to address this agent."""
        return self._agent.get_agent_metadata()

    # --- transport primitives ---------------------------------------------- #

    def add_remote_agent(self, peer_metadata: bytes) -> str:
        """Import a peer's serialized agent metadata. Returns the peer's agent name."""
        return self._agent.add_remote_agent(peer_metadata)

    def make_connection(self, peer_name: str) -> None:
        """Eagerly establish the UCX connection to a peer.

        Without this, the first WRITE to each peer includes the full UCX
        endpoint creation + RDMA handshake overhead (~seconds per peer).
        """
        self._agent.make_connection(peer_name)

    def prep_local(self, descs: Sequence[MemDesc]) -> Any:
        """Prepare a local-side descriptor list (no peer binding)."""
        return self._agent.prep_xfer_dlist(
            agent_name="", xfer_list=list(descs), mem_type="cuda", backends=self.backends
        )

    def prep_remote(self, peer_name: str, descs: Sequence[MemDesc]) -> Any:
        """Prepare a remote-side descriptor list bound to ``peer_name``.

        ``peer_name`` must have been imported via :meth:`add_remote_agent`;
        each entry must fall within a memory region the peer registered.
        """
        return self._agent.prep_xfer_dlist(
            agent_name=peer_name, xfer_list=list(descs), mem_type="cuda", backends=self.backends
        )

    def post_read(
        self,
        local_prep: Any,
        local_idxs: Sequence[int],
        remote_prep: Any,
        remote_idxs: Sequence[int],
    ) -> Any:
        """Post one batched READ pulling remote descriptors into matching local ones."""
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local_prep,
            local_indices=list(local_idxs),
            remote_xfer_side=remote_prep,
            remote_indices=list(remote_idxs),
            backends=self.backends,
        )
        state = self._agent.transfer(handle)
        if state in ("ERR", "ERROR", "FAIL"):
            raise RuntimeError(f"nixl READ post returned state {state}")
        return handle

    def wait(self, handle: Any, context: str = "") -> None:
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


def set_ucx_env_defaults() -> None:
    """Set UCX transport defaults for GPUDirect RDMA WRITEs.

    ``setdefault`` only — values exported by the SLURM templates (which own
    NIC selection via ``UCX_NET_DEVICES``) always win. Call once per process
    before constructing :class:`NixlAgent`.
    """
    os.environ.setdefault("UCX_TLS", "rc_mlx5,ud,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "put_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "8192")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
