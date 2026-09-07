"""Symmetric-memory buffer allocation for the fused dispatch/combine prototype kernels."""

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@dataclass
class PeerBuffer:
    """One symmetric tensor plus every peer's raw pointer into their copy of it."""

    local: torch.Tensor
    peer_ptrs: torch.Tensor  # (ep_size,) int64 data pointers, indexed by peer rank

    def zero_(self) -> None:
        self.local.zero_()


def _make_peer_buffer(
    group: dist.ProcessGroup, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> PeerBuffer:
    ep_size = group.size()
    t = symm_mem.empty(*shape, dtype=dtype, device=device)
    t.zero_()
    handle = symm_mem.rendezvous(t, group)
    peer_bufs = [handle.get_buffer(r, t.shape, t.dtype) for r in range(ep_size)]
    peer_ptrs = torch.tensor([b.data_ptr() for b in peer_bufs], dtype=torch.int64, device=device)
    return PeerBuffer(local=t, peer_ptrs=peer_ptrs)


@dataclass
class CometMoEBuffers:
    group: dist.ProcessGroup
    ep_size: int
    hidden_dim: int
    dispatch_capacity: int
    combine_capacity: int

    # Dispatch direction: routed tokens land here, grouped/aligned by local expert.
    dispatch_hidden: PeerBuffer  # (dispatch_capacity, hidden_dim), dtype = model dtype
    dispatch_flags: PeerBuffer  # (dispatch_capacity // block_m,), int32 arrival flags, one per tile

    # Combine direction: expert outputs are sent back to the origin rank here.
    combine_hidden: PeerBuffer  # (combine_capacity, hidden_dim)
    combine_flags: PeerBuffer  # (combine_capacity // block_m,), int32 arrival flags, one per tile

    _barrier_handle: Any  # `_SymmetricMemory` handle from `dispatch_hidden`'s rendezvous; see `barrier`

    def reset(self) -> None:
        self.dispatch_hidden.zero_()
        self.dispatch_flags.zero_()
        self.combine_hidden.zero_()
        self.combine_flags.zero_()

    def barrier(self) -> None:
        """A device-side barrier via `dispatch_hidden`'s symmetric-memory signal pad -- *not*
        `dist.barrier()`. Both give the same cross-rank ordering guarantee, but `dist.barrier()`
        is a host-blocking NCCL collective, and was measured to cost several milliseconds *per
        call* in some environments (idle-GPU/driver-reinit overhead after any host-blocking
        sync); this one dispatches entirely on-device (measured ~0.24ms overhead in that same
        environment, matching plain kernel-launch cost) since `_SymmetricMemory.barrier` spins on
        a signal pad rather than blocking the host thread on an NCCL op.
        """
        self._barrier_handle.barrier(0)


def init_comet_moe_buffers(
    group: dist.ProcessGroup,
    *,
    hidden_dim: int,
    dispatch_capacity: int,
    combine_capacity: int,
    block_m: int,
    dtype: torch.dtype,
    device: torch.device,
) -> CometMoEBuffers:
    """Allocate the symmetric buffers for one MoE layer.

    Capacities are static (allocated once, reused every step) — real training would size them
    from a worst-case token/expert-imbalance bound (mirroring `receive_capacity` /
    `num_max_tokens_per_rank` in torchtitan's MinimalAsyncEP) and assert on overflow rather than
    silently truncating; this prototype takes the caller's word for it. Both capacities must be
    multiples of `block_m`: one arrival flag covers exactly one BLOCK_M-row tile.
    """
    backend = symm_mem.get_backend(device)
    if backend != "CUDA":
        raise RuntimeError(f"comet_moe requires the CUDA symmetric-memory backend, got {backend}.")
    if dispatch_capacity % block_m or combine_capacity % block_m:
        raise ValueError(f"capacities must be multiples of block_m={block_m}.")

    ep_size = group.size()
    dispatch_hidden_t = symm_mem.empty(dispatch_capacity, hidden_dim, dtype=dtype, device=device)
    dispatch_hidden_t.zero_()
    dispatch_hidden_handle = symm_mem.rendezvous(dispatch_hidden_t, group)
    dispatch_hidden = PeerBuffer(
        local=dispatch_hidden_t,
        peer_ptrs=torch.tensor(
            [dispatch_hidden_handle.get_buffer(r, dispatch_hidden_t.shape, dtype).data_ptr() for r in range(ep_size)],
            dtype=torch.int64,
            device=device,
        ),
    )

    return CometMoEBuffers(
        group=group,
        ep_size=ep_size,
        hidden_dim=hidden_dim,
        dispatch_capacity=dispatch_capacity,
        combine_capacity=combine_capacity,
        dispatch_hidden=dispatch_hidden,
        dispatch_flags=_make_peer_buffer(group, (dispatch_capacity // block_m,), torch.int32, device),
        combine_hidden=_make_peer_buffer(group, (combine_capacity, hidden_dim), dtype, device),
        combine_flags=_make_peer_buffer(group, (combine_capacity // block_m,), torch.int32, device),
        _barrier_handle=dispatch_hidden_handle,
    )
