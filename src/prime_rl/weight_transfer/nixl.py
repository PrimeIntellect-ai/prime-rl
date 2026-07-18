"""Small NIXL adapter used by the trainer and vLLM workers."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from torch import Tensor

MemDesc = tuple[int, int, int]


@dataclass
class NixlRead:
    handle: Any
    state: str | None = None


class NixlAgent:
    def __init__(self, name: str, backends: Sequence[str] = ("UCX",)) -> None:
        try:
            from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore[import-not-found]
        except ImportError:
            from nixl._api import nixl_agent, nixl_agent_config  # type: ignore[import-not-found]

        self.name = name
        self.backends = list(backends)
        self._agent = nixl_agent(name, nixl_agent_config(backends=self.backends))

    def register_tensor(self, tensor: Tensor) -> None:
        self._agent.register_memory(tensor, backends=self.backends)

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_remote_agent(self, metadata: bytes) -> str:
        return self._agent.add_remote_agent(metadata)

    def make_connection(self, peer_name: str) -> None:
        self._agent.make_connection(peer_name)

    def prep_local(self, descs: Sequence[MemDesc]) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name="", xfer_list=list(descs), mem_type="cuda", backends=self.backends
        )

    def prep_remote(self, peer_name: str, descs: Sequence[MemDesc]) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name=peer_name, xfer_list=list(descs), mem_type="cuda", backends=self.backends
        )

    def prepare_read(self, local: Any, indices: Sequence[int], remote: Any) -> NixlRead:
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local,
            local_indices=list(indices),
            remote_xfer_side=remote,
            remote_indices=list(indices),
            backends=self.backends,
        )
        return NixlRead(handle)

    def post_read(self, read: NixlRead) -> None:
        if read.state == "INVALID":
            raise RuntimeError("NIXL READ cannot be reposted after a transfer failure")
        try:
            read.state = self._agent.transfer(read.handle)
            if read.state in ("ERR", "ERROR", "FAIL"):
                state = read.state
                raise RuntimeError(f"NIXL READ post failed with state {state}")
        except Exception:
            self._retire_read(read)
            raise

    def wait(
        self,
        read: NixlRead,
        context: str = "",
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            while True:
                if cancelled is not None and cancelled():
                    raise RuntimeError(f"NIXL transfer cancelled, context={context!r}")
                if read.state not in ("DONE", "SUCCESS"):
                    read.state = self._agent.check_xfer_state(read.handle)
                if read.state in ("DONE", "SUCCESS"):
                    return
                if read.state in ("ERR", "ERROR", "FAIL"):
                    raise RuntimeError(f"NIXL transfer failed with state={read.state}, context={context!r}")
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"NIXL transfer timed out after {timeout}s, context={context!r}")
                time.sleep(0.0005)
        except Exception:
            self._retire_read(read)
            raise

    def _retire_read(self, read: NixlRead) -> None:
        read.state = "INVALID"
        try:
            self._agent.release_xfer_handle(read.handle)
        except Exception:
            pass


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"


def set_ucx_env_defaults() -> None:
    os.environ.setdefault("UCX_TLS", "rc_x,rc,dc_x,dc,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "get_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "0")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
