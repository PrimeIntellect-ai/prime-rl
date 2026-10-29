"""NCCL weight broadcasts with bounded waits and recoverable peer failures."""

import ctypes
import os
import time

import torch
from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclUniqueId,
)
from vllm.distributed.utils import StatelessProcessGroup


class _Config(ctypes.Structure):
    # NCCL's versioned 2.17 ABI; newer libraries fill absent fields with defaults.
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("magic", ctypes.c_uint),
        ("version", ctypes.c_uint),
        ("blocking", ctypes.c_int),
        ("cgaClusterSize", ctypes.c_int),
        ("minCTAs", ctypes.c_int),
        ("maxCTAs", ctypes.c_int),
        ("netName", ctypes.c_char_p),
    ]


class WeightBroadcastCommunicator:
    """Single-threaded communicator, separate from inference/model collectives.

    Each broadcast waits for both NCCL launch and CUDA completion before returning.
    This keeps callers out of uninterruptible tensor.item()/copy waits after a peer
    dies. Abort runs on the same thread as the NCCL calls, never concurrently.
    """

    def __init__(self, group: StatelessProcessGroup, device: int | str | torch.device, timeout: float):
        if timeout <= 0:
            raise ValueError("NCCL operation timeout must be positive")
        if os.environ.get("NCCL_COMM_BLOCKING") not in (None, "0"):
            raise ValueError("Recoverable weight transfer requires NCCL_COMM_BLOCKING to be unset or 0")
        self.device = torch.device(f"cuda:{device}" if isinstance(device, int) else device)
        self.rank = group.rank
        self.timeout = timeout
        self.nccl = NCCLLibrary()
        self.comm = ncclComm_t()
        init = self.nccl.lib.ncclCommInitRankConfig
        init.restype = ctypes.c_int
        init.argtypes = [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int, ctypes.POINTER(_Config)]
        self._async_error = self.nccl.lib.ncclCommGetAsyncError
        self._async_error.restype = ctypes.c_int
        self._async_error.argtypes = [ncclComm_t, ctypes.POINTER(ctypes.c_int)]
        self._broadcast = self.nccl.lib.ncclBroadcast
        self._broadcast.restype = ctypes.c_int
        self._broadcast.argtypes = [
            buffer_type,
            buffer_type,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ncclComm_t,
            cudaStream_t,
        ]
        unique_id = group.broadcast_obj(self.nccl.ncclGetUniqueId() if self.rank == 0 else None, src=0)
        undef = -(2**31)
        config = _Config(ctypes.sizeof(_Config), 0xCAFEBEEF, 21700, 0, undef, undef, undef, None)
        with torch.cuda.device(self.device):
            try:
                deadline = time.monotonic() + timeout
                self._wait_launch(
                    init(ctypes.byref(self.comm), group.world_size, unique_id, self.rank, config), deadline
                )
            except BaseException:
                self.close()
                raise

    def _check(self, result: int) -> None:
        if result not in (0, 7):  # ncclSuccess / ncclInProgress
            self.nccl.NCCL_CHECK(result)

    def _state(self) -> int:
        state = ctypes.c_int()
        self.nccl.NCCL_CHECK(self._async_error(self.comm, ctypes.byref(state)))
        self._check(state.value)
        return state.value

    def _poll(self, deadline: float) -> None:
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError(f"NCCL weight transfer did not complete within {self.timeout}s")
        # Most metadata broadcasts finish in microseconds. Do not impose a
        # scheduler sleep on each one; yield during longer transfers/waits.
        if now >= deadline - self.timeout + 0.001:
            time.sleep(0.0001)

    def _wait_launch(self, result: int, deadline: float) -> None:
        self._check(result)
        while result == 7:
            result = self._state()
            if result == 7:
                self._poll(deadline)

    def broadcast(self, tensor: torch.Tensor, src: int) -> None:
        if not self.comm:
            raise RuntimeError("NCCL weight communicator is closed; reconnect before receiving weights")
        if tensor.device != self.device or not tensor.is_contiguous():
            raise ValueError("Weight broadcast requires a contiguous tensor on the communicator's device")
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            deadline = time.monotonic() + self.timeout
            try:
                result = self._broadcast(
                    buffer_type(tensor.data_ptr()),
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    ncclDataTypeEnum.from_torch(tensor.dtype),
                    src,
                    self.comm,
                    cudaStream_t(stream.cuda_stream),
                )
                self._wait_launch(result, deadline)
                while not stream.query():
                    self._state()
                    self._poll(deadline)
                self._state()
            except BaseException:
                self.close()
                raise

    def close(self) -> None:
        if self.comm:
            with torch.cuda.device(self.device):
                self.nccl.ncclCommAbort(self.comm)
            self.comm = ncclComm_t()
