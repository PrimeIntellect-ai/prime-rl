from pathlib import Path

import pytest

from prime_rl.utils import rdma


def _fake_sysfs(tmp_path: Path, ports: dict[str, tuple[str, str, str]]) -> Path:
    """ports: hca name -> (PCI path of the NIC under tmp_path/pci, link layer, state)."""
    root = tmp_path / "infiniband"
    for hca, (pci, link_layer, state) in ports.items():
        device = tmp_path / "pci" / pci
        device.mkdir(parents=True, exist_ok=True)
        port = root / hca / "ports" / "1"
        port.mkdir(parents=True)
        (port / "link_layer").write_text(link_layer + "\n")
        (port / "state").write_text(state + "\n")
        (root / hca / "device").symlink_to(device)
    return root


@pytest.fixture
def gpu_behind_switch_b(tmp_path, monkeypatch):
    gpu = tmp_path / "pci" / "root0" / "switch_b" / "gpu"
    gpu.mkdir(parents=True)
    monkeypatch.setattr(rdma, "gpu_pci_path", lambda device: gpu.resolve())


def test_nearest_port_is_behind_the_gpus_pcie_switch(tmp_path, monkeypatch, gpu_behind_switch_b):
    root = _fake_sysfs(
        tmp_path,
        {
            "mlx5_0": ("root0/switch_a/nic", "InfiniBand", "4: ACTIVE"),
            "mlx5_1": ("root0/switch_b/nic", "InfiniBand", "4: ACTIVE"),
            "mlx5_2": ("root1/switch_c/nic", "InfiniBand", "4: ACTIVE"),
            "mlx5_3": ("root0/switch_b/nic_down", "InfiniBand", "1: DOWN"),
        },
    )
    monkeypatch.setattr(rdma, "SYSFS_INFINIBAND", root)
    assert rdma.nearest_rdma_port("0") == "mlx5_1:1"


def test_roce_ports_are_used_without_infiniband(tmp_path, monkeypatch, gpu_behind_switch_b):
    root = _fake_sysfs(
        tmp_path,
        {
            "mlx5_0": ("root0/switch_a/nic", "Ethernet", "4: ACTIVE"),
            "mlx5_1": ("root0/switch_b/nic", "Ethernet", "4: ACTIVE"),
        },
    )
    monkeypatch.setattr(rdma, "SYSFS_INFINIBAND", root)
    assert rdma.nearest_rdma_port("0") == "mlx5_1:1"


def test_infiniband_is_preferred_over_roce(tmp_path, monkeypatch, gpu_behind_switch_b):
    root = _fake_sysfs(
        tmp_path,
        {
            "mlx5_0": ("root0/switch_a/nic", "InfiniBand", "4: ACTIVE"),
            "mlx5_1": ("root0/switch_b/nic", "Ethernet", "4: ACTIVE"),
        },
    )
    monkeypatch.setattr(rdma, "SYSFS_INFINIBAND", root)
    assert rdma.nearest_rdma_port("0") == "mlx5_0:1"


def test_no_active_port_raises(tmp_path, monkeypatch, gpu_behind_switch_b):
    root = _fake_sysfs(tmp_path, {"mlx5_0": ("root0/switch_b/nic", "InfiniBand", "1: DOWN")})
    monkeypatch.setattr(rdma, "SYSFS_INFINIBAND", root)
    with pytest.raises(RuntimeError, match="No active"):
        rdma.nearest_rdma_port("0")
