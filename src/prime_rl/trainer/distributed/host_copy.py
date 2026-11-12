"""A device-to-host copy in two compile-friendly ops: `host_copy_start` queues the copy into pinned memory and returns
at once, `host_copy_wait` waits for that copy alone. Work queued between them keeps the GPU busy while the host waits,
where a blocking ``.cpu()`` would wait for everything queued before it and leave the GPU idle until the host queues
more."""

import torch
from torch import Tensor

# The event recorded after each pending copy (None for a copy of a host tensor), by the address of its buffer.
_PENDING: dict[int, torch.cuda.Event | None] = {}


@torch.library.custom_op("prime_rl::host_copy_start", mutates_args=())
def host_copy_start(x: Tensor) -> Tensor:
    """``x`` copied into a pinned host tensor, asynchronously: read it only through `host_copy_wait`."""
    if not x.is_cuda:
        host = x.clone()
        _PENDING[host.data_ptr()] = None
        return host
    host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
    host.copy_(x, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    _PENDING[host.data_ptr()] = event
    return host


@host_copy_start.register_fake
def _(x: Tensor) -> Tensor:
    return torch.empty(x.shape, dtype=x.dtype, device="cpu")


@torch.library.custom_op("prime_rl::host_copy_wait", mutates_args=())
def host_copy_wait(host: Tensor) -> Tensor:
    """The values of a `host_copy_start` copy, once it has landed."""
    event = _PENDING.pop(host.data_ptr())
    if event is not None:
        event.synchronize()
    return host.clone()


@host_copy_wait.register_fake
def _(host: Tensor) -> Tensor:
    return torch.empty_like(host)
