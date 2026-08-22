"""Small NIXL adapter used by the trainer and vLLM workers."""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from torch import Tensor

MemDesc = tuple[int, int, int]


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

    def register_tensor(self, tensor: Tensor) -> None:
        self._agent.register_memory(tensor, backends=["UCX"])

    def get_metadata(self) -> bytes:
        return self._agent.get_agent_metadata()

    def add_remote_agent(self, metadata: bytes) -> str:
        return self._agent.add_remote_agent(metadata).decode()

    def make_connection(self, peer_name: str) -> None:
        self._agent.make_connection(peer_name)

    def prepare_xfer_dlist(self, descs: Sequence[MemDesc], agent_name: str | None = None) -> Any:
        return self._agent.prep_xfer_dlist(
            agent_name=agent_name or "",
            xfer_list=list(descs),
            mem_type="cuda",
            backends=["UCX"],
        )

    def post_read(self, local: Any, indices: Sequence[int], remote: Any) -> Any:
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local,
            local_indices=list(indices),
            remote_xfer_side=remote,
            remote_indices=list(indices),
            backends=["UCX"],
        )
        state = self._agent.transfer(handle)
        if state in ("ERR", "ERROR", "FAIL"):
            raise RuntimeError(f"NIXL READ post failed with state {state}")
        return handle

    def send_notification(self, peer_name: str, notification: str) -> None:
        self._agent.send_notif(peer_name, notification)

    def wait_for_notification(
        self,
        peer_names: Sequence[str],
        notification: str,
        *,
        timeout: float,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        pending = set(peer_names)
        deadline = time.monotonic() + timeout
        while pending:
            if cancelled is not None and cancelled():
                raise RuntimeError("NIXL notification wait cancelled")
            for sender in list(pending):
                if self._agent.check_remote_xfer_done(
                    sender,
                    notification.encode(),
                    backends=["UCX"],
                    tag_is_prefix=False,
                ):
                    pending.remove(sender)
            if pending and time.monotonic() >= deadline:
                raise TimeoutError(f"NIXL notification wait timed out after {timeout}s, missing={sorted(pending)}")
            if pending:
                time.sleep(0.0005)

    def wait(
        self,
        handle: Any,
        context: str = "",
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if cancelled is not None and cancelled():
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"NIXL transfer cancelled, context={context!r}")
            state = self._agent.check_xfer_state(handle)
            if state in ("DONE", "SUCCESS"):
                self._agent.release_xfer_handle(handle)
                return
            if state in ("ERR", "ERROR", "FAIL"):
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"NIXL transfer failed with state={state}, context={context!r}")
            if deadline is not None and time.monotonic() >= deadline:
                self._agent.release_xfer_handle(handle)
                raise TimeoutError(f"NIXL transfer timed out after {timeout}s, context={context!r}")
            time.sleep(0.0005)


def make_agent_name(role: str, global_rank: int) -> str:
    return f"{role}-{socket.gethostname()}-r{global_rank}"


def set_ucx_env_defaults(device_index: int) -> None:
    if "UCX_NET_DEVICES" not in os.environ:
        import pynvml

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_device = str(device_index)
        if visible_devices:
            visible_device = visible_devices.split(",")[device_index].strip()

        pynvml.nvmlInit()
        try:
            if visible_device.isdecimal():
                handle = pynvml.nvmlDeviceGetHandleByIndex(int(visible_device))
            elif visible_device.startswith("GPU-"):
                handle = pynvml.nvmlDeviceGetHandleByUUID(visible_device.encode())
            else:
                raise RuntimeError(f"Unsupported CUDA_VISIBLE_DEVICES entry for NIXL: {visible_device}")
            bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
        finally:
            pynvml.nvmlShutdown()

        if isinstance(bus_id, bytes):
            bus_id = bus_id.decode()
        domain, bus, device = bus_id.rsplit(":", 2)
        gpu_path = Path("/sys/bus/pci/devices") / f"{int(domain, 16):04x}:{bus.lower()}:{device.lower()}"
        if not gpu_path.exists():
            raise RuntimeError(f"GPU PCI device is missing from sysfs: {gpu_path}")

        rdma_ports: list[tuple[str, Path]] = []
        for hca_path in sorted(Path("/sys/class/infiniband").glob("*")):
            for port_path in sorted((hca_path / "ports").glob("*")):
                if (port_path / "link_layer").read_text().strip() != "InfiniBand":
                    continue
                if "ACTIVE" not in (port_path / "state").read_text():
                    continue
                rdma_ports.append((f"{hca_path.name}:{port_path.name}", (hca_path / "device").resolve()))
        if not rdma_ports:
            raise RuntimeError("NIXL requires an active InfiniBand port")

        gpu_parts = gpu_path.resolve().parts
        ports_by_distance: list[tuple[int, str]] = []
        for port, device_path in rdma_ports:
            device_parts = device_path.parts
            common_parts = 0
            for gpu_part, device_part in zip(gpu_parts, device_parts):
                if gpu_part != device_part:
                    break
                common_parts += 1
            distance = len(gpu_parts) + len(device_parts) - 2 * common_parts
            ports_by_distance.append((distance, port))

        os.environ["UCX_NET_DEVICES"] = min(ports_by_distance)[1]
        os.environ.setdefault("UCX_MAX_RNDV_RAILS", "1")
        os.environ.setdefault("UCX_MAX_RMA_RAILS", "1")
    os.environ.setdefault("UCX_TLS", "rc_x,rc,dc_x,dc,cuda_copy")
    os.environ.setdefault("UCX_IB_GPU_DIRECT_RDMA", "y")
    os.environ.setdefault("UCX_RNDV_SCHEME", "get_zcopy")
    os.environ.setdefault("UCX_RNDV_THRESH", "0")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    os.environ.setdefault("UCX_WARN_UNUSED_ENV_VARS", "n")
