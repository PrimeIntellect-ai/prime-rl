"""RDMA defaults for ModelExpress receivers."""

from __future__ import annotations

import os

_NIC_PIN = "MX_RDMA_NIC_PIN"
_UCX_NET_DEVICES = "UCX_NET_DEVICES"
_MIN_GBPS = "MX_RESHARD_MIN_GBPS"
_DEFAULT_MIN_GBPS = "25"


def apply_rdma_defaults() -> None:
    """Set ModelExpress rail and throughput defaults."""
    # Enabling automatic pinning would override an explicit UCX device list.
    if _NIC_PIN not in os.environ and _UCX_NET_DEVICES not in os.environ:
        os.environ[_NIC_PIN] = "auto"
    os.environ.setdefault(_MIN_GBPS, _DEFAULT_MIN_GBPS)
