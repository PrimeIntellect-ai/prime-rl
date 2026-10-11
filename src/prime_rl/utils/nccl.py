import os
from collections.abc import Iterator

import pynvml
import torch

from prime_rl.utils.logger import get_logger


def disable_nccl_p2p_if_unavailable() -> None:
    """Disable NCCL P2P/SHM transports when GPUs lack NVLink interconnect.

    In environments without NVLink (e.g. PCIe-only or VMs with GPU passthrough),
    NCCL's P2P and SHM transports can fail because they rely on CUDA IPC which
    requires peer access. Disabling them forces NCCL to use socket-based transport.

    Uses pynvml to check physical GPU topology, which works regardless of
    CUDA_VISIBLE_DEVICES restrictions on the current process.
    """
    pynvml.nvmlInit()
    try:
        n = pynvml.nvmlDeviceGetCount()
        if n < 2:
            return

        for i in range(n):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            for link in range(18):
                try:
                    if pynvml.nvmlDeviceGetNvLinkState(handle, link):
                        return  # NVLink found, P2P should work
                except pynvml.NVMLError:
                    break

        # No NVLink found between any GPUs — disable P2P/SHM if not already set
        if "NCCL_P2P_DISABLE" not in os.environ or "NCCL_SHM_DISABLE" not in os.environ:
            os.environ["NCCL_P2P_DISABLE"] = "1"
            os.environ["NCCL_SHM_DISABLE"] = "1"
            get_logger().warning(
                "No NVLink detected, disabling NCCL P2P and SHM transports. "
                "Override by setting NCCL_P2P_DISABLE=0 and NCCL_SHM_DISABLE=0 explicitly."
            )
    finally:
        pynvml.nvmlShutdown()


def iter_tensor_buckets(
    tensor_info: list[tuple[str, torch.Size, int]],
    dtype: torch.dtype,
    max_bytes: int = 512 * 1024**2,
) -> Iterator[list[tuple[str, torch.Size, int]]]:
    """Group whole tensors without exceeding the budget, except for a single oversized tensor."""
    itemsize = dtype.itemsize
    bucket = []
    bucket_bytes = 0
    for info in tensor_info:
        size = info[2] * itemsize
        if bucket and bucket_bytes + size > max_bytes:
            yield bucket
            bucket = []
            bucket_bytes = 0
        bucket.append(info)
        bucket_bytes += size
    if bucket:
        yield bucket
