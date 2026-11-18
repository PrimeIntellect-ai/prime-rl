"""RDMA port selection by PCIe distance to a GPU.

GPUDirect RDMA is fastest, and safest, between a GPU and the NIC behind the same PCIe switch. Traffic routed to a
NIC on the other CPU socket crosses the inter-socket link, and sustained transfer load that way can stall the GPU.

    python -m prime_rl.utils.rdma <cuda device index or GPU-UUID>

prints the active InfiniBand port nearest that GPU (e.g. ``mlx5_4:1``), for ``UCX_NET_DEVICES``.
"""

import sys
from pathlib import Path

SYSFS_INFINIBAND = Path("/sys/class/infiniband")
SYSFS_PCI = Path("/sys/bus/pci/devices")


def gpu_pci_path(device: str) -> Path:
    """Resolved sysfs PCI path of a GPU given as an NVML index or a ``GPU-`` UUID."""
    import pynvml

    pynvml.nvmlInit()
    try:
        if device.isdecimal():
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(device))
        elif device.startswith("GPU-"):
            handle = pynvml.nvmlDeviceGetHandleByUUID(device.encode())
        else:
            raise RuntimeError(f"Unsupported GPU identifier: {device}")
        bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId
    finally:
        pynvml.nvmlShutdown()

    if isinstance(bus_id, bytes):
        bus_id = bus_id.decode()
    domain, bus, slot = bus_id.rsplit(":", 2)
    path = SYSFS_PCI / f"{int(domain, 16):04x}:{bus.lower()}:{slot.lower()}"
    if not path.exists():
        raise RuntimeError(f"GPU PCI device is missing from sysfs: {path}")
    return path.resolve()


def active_rdma_ports(link_layer: str = "InfiniBand") -> list[tuple[str, Path]]:
    """Active ``<hca>:<port>`` RDMA ports of the given link layer, with their resolved PCI device paths."""
    ports: list[tuple[str, Path]] = []
    for hca_path in sorted(SYSFS_INFINIBAND.glob("*")):
        for port_path in sorted((hca_path / "ports").glob("*")):
            try:
                if (port_path / "link_layer").read_text().strip() != link_layer:
                    continue
                if "ACTIVE" not in (port_path / "state").read_text():
                    continue
            except OSError:
                continue
            ports.append((f"{hca_path.name}:{port_path.name}", (hca_path / "device").resolve()))
    return ports


def pci_distance(a: Path, b: Path) -> int:
    """Hops between two devices in the sysfs PCI tree (through their deepest common bridge)."""
    common = 0
    for part_a, part_b in zip(a.parts, b.parts):
        if part_a != part_b:
            break
        common += 1
    return len(a.parts) + len(b.parts) - 2 * common


def nearest_rdma_port(device: str, link_layer: str = "InfiniBand") -> str:
    """The active RDMA port nearest the GPU in the PCIe tree."""
    ports = active_rdma_ports(link_layer)
    if not ports:
        raise RuntimeError(f"No active {link_layer} RDMA port")
    gpu = gpu_pci_path(device)
    return min((pci_distance(gpu, path), port) for port, path in ports)[1]


if __name__ == "__main__":
    print(nearest_rdma_port(sys.argv[1]))
