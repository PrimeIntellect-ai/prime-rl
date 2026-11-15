import dataclasses
import json
import math
import os
from collections.abc import Iterator
from typing import Any

import pynvml
import torch
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId
from vllm.distributed.utils import StatelessProcessGroup

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


def encode_tensor_metadata(metadata: dict[torch.dtype, list[tuple[str, torch.Size, int]]]) -> bytes:
    """Serialize a broadcast's tensor metadata as JSON: ``[[dtype name, [[key, shape], ...]], ...]``."""
    return json.dumps(
        [
            [str(dtype).removeprefix("torch."), [[key, list(shape)] for key, shape, _ in tensor_info]]
            for dtype, tensor_info in metadata.items()
        ]
    ).encode()


def decode_tensor_metadata(data: bytes) -> dict[torch.dtype, list[tuple[str, torch.Size, int]]]:
    """Parse ``encode_tensor_metadata`` output, deriving each tensor's element count from its shape."""
    metadata = {}
    for dtype_name, tensor_info in json.loads(data):
        dtype = getattr(torch, dtype_name)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Invalid dtype in weight metadata: {dtype_name}")
        metadata[dtype] = [(key, torch.Size(shape), math.prod(shape)) for key, shape in tensor_info]
    return metadata


@dataclasses.dataclass
class UniqueIdProcessGroup(StatelessProcessGroup):
    """A process group limited to exchanging the NCCL unique id as raw bytes."""

    def broadcast_obj(self, obj: Any, src: int) -> ncclUniqueId:
        key = f"nccl_unique_id/{src}"
        if self.rank == src:
            if not isinstance(obj, ncclUniqueId):
                raise TypeError(f"Can only broadcast an NCCL unique id, got {type(obj).__name__}")
            self.store.set(key, bytes(obj))
            return obj
        return ncclUniqueId.from_buffer_copy(self.store.get(key))

    def send_obj(self, obj: Any, dst: int) -> None:
        raise NotImplementedError("UniqueIdProcessGroup only broadcasts the NCCL unique id")

    def recv_obj(self, src: int) -> Any:
        raise NotImplementedError("UniqueIdProcessGroup only broadcasts the NCCL unique id")

    def all_gather_obj(self, obj: Any) -> list[Any]:
        raise NotImplementedError("UniqueIdProcessGroup only broadcasts the NCCL unique id")


def create_nccl_communicator(
    host: str, port: int, rank: int, world_size: int, device: int | str | torch.device, timeout: int
) -> PyNcclCommunicator:
    """Rendezvous at `host:port` through a TCPStore hosted by rank 0 and build an NCCL communicator over it."""
    store = StatelessProcessGroup.create(
        host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout
    ).store
    group = UniqueIdProcessGroup(rank=rank, world_size=world_size, store=store)
    return PyNcclCommunicator(group, device=device)
