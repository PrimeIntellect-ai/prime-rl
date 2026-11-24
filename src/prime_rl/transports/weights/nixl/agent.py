"""Small NIXL adapter used by the trainer and vLLM workers."""

from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from torch import Tensor

MemDesc = tuple[int, int, int]


@dataclass(slots=True)
class PreparedRead:
    """Reusable READ request with descriptor lists kept alive for its lifetime."""

    local_descriptors: Any
    remote_descriptors: Any
    handle: Any


def group_notification(group_index: int, generation: int) -> str:
    return f"{group_index:08x}:{generation:016x}"


def policy_notification(step: int, event: str) -> str:
    return f"policy:{step:016x}:{event}"


class NixlAgent:
    def __init__(self, name: str) -> None:
        try:
            from nixl_cu13._api import nixl_agent, nixl_agent_config  # type: ignore[import-not-found]
        except ImportError:
            from nixl._api import nixl_agent, nixl_agent_config  # type: ignore[import-not-found]

        self.name = name
        self._agent = nixl_agent(name, nixl_agent_config(backends=["UCX"]))
        self._notifications: dict[str, list[bytes]] = {}

    def register_tensor(self, tensor: Tensor) -> None:
        self._agent.register_memory(tensor, backends=["UCX"])

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_remote_agent(self, metadata: bytes) -> NixlPeer:
        return NixlPeer(
            local_agent=self,
            remote_agent_name=self._agent.add_remote_agent(metadata).decode(),
        )

    def make_connection(self, peer: NixlPeer) -> None:
        self._agent.make_connection(peer.remote_agent_name)

    def prepare_xfer_dlist(self, descs: Sequence[MemDesc], peer: NixlPeer | None = None) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name=peer.remote_agent_name if peer is not None else "",
            xfer_list=list(descs),
            mem_type="VRAM",
            backends=["UCX"],
        )

    def prepare_read(self, local: Any, indices: Sequence[int], remote: Any) -> PreparedRead:
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local,
            local_indices=list(indices),
            remote_xfer_side=remote,
            remote_indices=list(indices),
            backends=["UCX"],
        )
        return PreparedRead(
            local_descriptors=local,
            remote_descriptors=remote,
            handle=handle,
        )

    def post_read(self, read: PreparedRead, notification: str) -> None:
        state = self._agent.transfer(read.handle, notification.encode())
        if state in ("ERR", "ERROR", "FAIL"):
            raise RuntimeError(f"NIXL READ post failed with state {state}")

    def send_notification(self, peer: NixlPeer, notification: str) -> None:
        self._agent.send_notif(peer.remote_agent_name, notification)

    def wait_for_notification(
        self,
        peers: Sequence[NixlPeer],
        notification: str,
        *,
        timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        pending = {peer.remote_agent_name for peer in peers}
        encoded_notification = notification.encode()
        deadline = time.monotonic() + timeout
        while pending:
            if cancelled is not None and cancelled():
                raise RuntimeError("NIXL notification wait cancelled")
            for sender, messages in self._agent.get_new_notifs(backends=["UCX"]).items():
                self._notifications.setdefault(sender, []).extend(messages)
            for sender in list(pending):
                messages = self._notifications.get(sender, [])
                if encoded_notification in messages:
                    messages.remove(encoded_notification)
                    pending.remove(sender)
            if pending and time.monotonic() >= deadline:
                raise TimeoutError(f"NIXL notification wait timed out after {timeout}s, missing={sorted(pending)}")
            if pending:
                time.sleep(0.0005)

    def wait(
        self,
        read: PreparedRead,
        context: str = "",
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                self._agent.release_xfer_handle(read.handle)
                raise RuntimeError(f"NIXL transfer cancelled, context={context!r}")
            state = self._agent.check_xfer_state(read.handle)
            if state in ("DONE", "SUCCESS"):
                return
            if state in ("ERR", "ERROR", "FAIL"):
                self._agent.release_xfer_handle(read.handle)
                raise RuntimeError(f"NIXL transfer failed with state={state}, context={context!r}")
            if deadline is not None and time.monotonic() >= deadline:
                self._agent.release_xfer_handle(read.handle)
                raise TimeoutError(f"NIXL transfer timed out after {timeout}s, context={context!r}")
            time.sleep(0.0005)


@dataclass(frozen=True, slots=True)
class NixlPeer:
    local_agent: NixlAgent
    remote_agent_name: str


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"


def set_ucx_env_defaults(device_index: int) -> None:
    if "UCX_NET_DEVICES" not in os.environ:
        from prime_rl.utils.rdma import nearest_rdma_port

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_device = str(device_index)
        if visible_devices:
            visible_device = visible_devices.split(",")[device_index].strip()
        os.environ["UCX_NET_DEVICES"] = nearest_rdma_port(visible_device)
        os.environ.setdefault("UCX_MAX_RNDV_RAILS", "1")
        os.environ.setdefault("UCX_MAX_RMA_RAILS", "1")
    os.environ.setdefault("UCX_TLS", "rc_x,rc,dc_x,dc,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "get_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "0")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
