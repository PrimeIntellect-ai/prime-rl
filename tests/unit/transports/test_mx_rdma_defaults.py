"""ModelExpress RDMA default tests."""

import os
from pathlib import Path
from runpy import run_path

import pytest

_MODULE = run_path(Path(__file__).parents[3] / "src" / "prime_rl" / "transports" / "weights" / "mx_rdma.py")
apply_rdma_defaults = _MODULE["apply_rdma_defaults"]
DEFAULT_MIN_GBPS = _MODULE["_DEFAULT_MIN_GBPS"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("MX_RDMA_NIC_PIN", "MX_RESHARD_MIN_GBPS", "UCX_NET_DEVICES"):
        monkeypatch.delenv(key, raising=False)


def test_defaults_enable_rail_spreading():
    apply_rdma_defaults()

    assert os.environ["MX_RDMA_NIC_PIN"] == "auto"
    assert os.environ["MX_RESHARD_MIN_GBPS"] == DEFAULT_MIN_GBPS


def test_operator_overrides_are_preserved(monkeypatch):
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "mlx5_4:1")
    monkeypatch.setenv("MX_RESHARD_MIN_GBPS", "5")

    apply_rdma_defaults()

    assert os.environ["MX_RDMA_NIC_PIN"] == "mlx5_4:1"
    assert os.environ["MX_RESHARD_MIN_GBPS"] == "5"


def test_ucx_device_list_suppresses_default_pin(monkeypatch):
    monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")

    apply_rdma_defaults()

    assert "MX_RDMA_NIC_PIN" not in os.environ
    assert os.environ["UCX_NET_DEVICES"] == "mlx5_0:1"
    assert os.environ["MX_RESHARD_MIN_GBPS"] == DEFAULT_MIN_GBPS


def test_explicit_pin_wins_over_ucx_device_list(monkeypatch):
    monkeypatch.setenv("UCX_NET_DEVICES", "mlx5_0:1")
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "auto")

    apply_rdma_defaults()

    assert os.environ["MX_RDMA_NIC_PIN"] == "auto"


def test_empty_pin_override_is_preserved(monkeypatch):
    monkeypatch.setenv("MX_RDMA_NIC_PIN", "")

    apply_rdma_defaults()

    assert os.environ["MX_RDMA_NIC_PIN"] == ""


def test_calling_twice_is_stable():
    apply_rdma_defaults()
    apply_rdma_defaults()

    assert os.environ["MX_RDMA_NIC_PIN"] == "auto"
