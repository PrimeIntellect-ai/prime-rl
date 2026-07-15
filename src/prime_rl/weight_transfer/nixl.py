"""Small NIXL adapter used by the pull-based weight-transfer backend."""

from __future__ import annotations

import os
import socket
import time
from typing import Any, Sequence

from torch import Tensor

MemoryDescriptor = tuple[int, int, int]


class NixlAgent:
    def __init__(self, name: str, backends: Sequence[str] = ("UCX",)) -> None:
        from nixl._api import nixl_agent, nixl_agent_config

        self.name = name
        self.backends = tuple(backends)
        self._agent = nixl_agent(name, nixl_agent_config(backends=list(self.backends)))

    def register_tensor(self, tensor: Tensor) -> None:
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError(
                f"NIXL source tensors must be contiguous CUDA tensors, got device={tensor.device}, "
                f"shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}"
            )
        self._agent.register_memory(tensor, backends=list(self.backends))

    def metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_remote_agent(self, metadata: bytes) -> str:
        return self._agent.add_remote_agent(metadata)

    def connect(self, remote_name: str) -> None:
        self._agent.make_connection(remote_name)

    def prepare_local(self, descriptors: Sequence[MemoryDescriptor]) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name="",
            xfer_list=list(descriptors),
            mem_type="cuda",
            backends=list(self.backends),
        )

    def prepare_remote(self, remote_name: str, descriptors: Sequence[MemoryDescriptor]) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name=remote_name,
            xfer_list=list(descriptors),
            mem_type="cuda",
            backends=list(self.backends),
        )

    def read(
        self,
        local_descriptors: Any,
        local_indices: Sequence[int],
        remote_descriptors: Any,
        remote_indices: Sequence[int],
    ) -> Any:
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local_descriptors,
            local_indices=list(local_indices),
            remote_xfer_side=remote_descriptors,
            remote_indices=list(remote_indices),
            backends=list(self.backends),
        )
        state = self._agent.transfer(handle)
        if state in {"ERR", "ERROR", "FAIL"}:
            raise RuntimeError(f"NIXL READ submission failed with state {state}")
        return handle

    def wait(self, handle: Any, context: str = "", timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            state = self._agent.check_xfer_state(handle)
            if state in {"DONE", "SUCCESS"}:
                self._agent.release_xfer_handle(handle)
                return
            if state in {"ERR", "ERROR", "FAIL"}:
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"NIXL transfer failed with state {state}: {context}")
            if deadline is not None and time.monotonic() >= deadline:
                self._agent.release_xfer_handle(handle)
                raise TimeoutError(f"NIXL transfer timed out after {timeout}s: {context}")
            time.sleep(0.0005)


def agent_name(role: str, rank: int, session_id: str) -> str:
    return f"prime-rl-{session_id}-{role}-{socket.gethostname()}-{rank}"


def configure_ucx(local_device: int) -> None:
    """Apply topology-aware NIC pinning without overriding operator settings."""

    from modelexpress.ucx_utils import apply_nic_pin_for_device

    apply_nic_pin_for_device(local_device)
    os.environ.setdefault("UCX_TLS", "rc_mlx5,ud,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "put_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "8192")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
