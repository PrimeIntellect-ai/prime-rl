"""One fused kernel that applies a sparse weight delta to the live params in place.

Each change is a checkpoint tensor's flat index and a raw-bit difference. The kernel finds its tensor (a binary
search over the delta's entries), maps the index through that tensor's writes (where the engine's loader copied
each block of the checkpoint tensor, recorded once per ``CheckpointIndexMap``), and adds the difference to every
param element it was written to, wrapping in the bits dtype. Indices are unique within a delta, so the
read-modify-writes need no sort or atomics. It also returns each change's contribution to the position-weighted
checksum of its tensor, ``(index + 1) * (new - old)``, so the running checksums move with the changes alone.
"""

import torch
import triton
import triton.language as tl

# Fields of one write: first row, rows, first column, columns, param, offset, row stride, column stride, and where its
# row and column offset tables start in the maps buffer (-1: the stride applies)
WRITE_FIELDS = 10

_TL_BITS = {torch.uint8: tl.uint8, torch.int16: tl.int16, torch.int32: tl.int32, torch.int64: tl.int64}


# Triton otherwise compiles a variant per value class of these (== 1, divisible by 16, pointer alignment), which a new
# delta can hit inside a weight-update pause: one variant per bits dtype serves every delta.
_RUNTIME_ARGS = (
    "indices_ptr",
    "deltas_ptr",
    "contributions_ptr",
    "entry_ends_ptr",
    "entry_names_ptr",
    "writes_ptr",
    "num_cols_ptr",
    "param_ptrs_ptr",
    "maps_ptr",
    "num_changes",
    "num_entries",
    "search_steps",
)


@triton.jit(do_not_specialize=_RUNTIME_ARGS)
def _sparse_apply_kernel(
    indices_ptr,
    deltas_ptr,
    contributions_ptr,
    entry_ends_ptr,
    entry_names_ptr,
    writes_ptr,
    num_cols_ptr,
    param_ptrs_ptr,
    maps_ptr,
    num_changes,
    num_entries,
    search_steps,
    MAX_WRITES: tl.constexpr,
    TRACK: tl.constexpr,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_changes
    index = tl.load(indices_ptr + offsets, mask=mask, other=0)
    delta = tl.load(deltas_ptr + offsets, mask=mask, other=0)

    # The entry of each change: the first whose cumulative end exceeds its position
    low = tl.zeros([BLOCK], dtype=tl.int64)
    high = tl.full([BLOCK], num_entries, dtype=tl.int64)
    for _ in range(search_steps):
        active = low < high
        middle = (low + high) // 2
        end = tl.load(entry_ends_ptr + middle, mask=mask & active, other=0)
        right = active & (end <= offsets)
        low = tl.where(right, middle + 1, low)
        high = tl.where(active & (end > offsets), middle, high)

    name = tl.load(entry_names_ptr + low, mask=mask, other=-1)
    valid = mask & (name >= 0)
    num_cols = tl.load(num_cols_ptr + name, mask=valid, other=1)
    row = index // num_cols
    col = index % num_cols
    total = tl.zeros([BLOCK], dtype=tl.int64)
    for k in tl.static_range(MAX_WRITES):
        write = writes_ptr + (name * MAX_WRITES + k) * 10
        row0 = tl.load(write + 0, mask=valid, other=0)
        rows = tl.load(write + 1, mask=valid, other=0)
        col0 = tl.load(write + 2, mask=valid, other=0)
        cols = tl.load(write + 3, mask=valid, other=0)
        r = row - row0
        c = col - col0
        hit = valid & (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
        param = tl.load(write + 4, mask=hit, other=0)
        offset = tl.load(write + 5, mask=hit, other=0)
        row_stride = tl.load(write + 6, mask=hit, other=0)
        col_stride = tl.load(write + 7, mask=hit, other=0)
        row_table = tl.load(write + 8, mask=hit, other=-1)
        col_table = tl.load(write + 9, mask=hit, other=-1)
        row_part = tl.where(
            row_table >= 0, tl.load(maps_ptr + row_table + r, mask=hit & (row_table >= 0), other=0), r * row_stride
        )
        col_part = tl.where(
            col_table >= 0, tl.load(maps_ptr + col_table + c, mask=hit & (col_table >= 0), other=0), c * col_stride
        )
        base = tl.load(param_ptrs_ptr + param, mask=hit, other=0).to(tl.pointer_type(BITS))
        target = base + (offset + row_part + col_part)
        old = tl.load(target, mask=hit, other=0)
        new = old + delta
        tl.store(target, new, mask=hit)
        if TRACK:
            total += tl.where(hit, (new.to(tl.int64) - old.to(tl.int64)) * (index + 1), 0)
    if TRACK:
        tl.store(contributions_ptr + offsets, total, mask=mask)


def sparse_apply(
    indices: torch.Tensor,
    deltas: torch.Tensor,
    entry_ends: torch.Tensor,
    entry_names: torch.Tensor,
    writes: torch.Tensor,
    num_cols: torch.Tensor,
    param_ptrs: torch.Tensor,
    maps: torch.Tensor,
    track: bool,
    block: int = 1024,
) -> torch.Tensor | None:
    """Apply ``deltas`` (raw-bit differences in one bits dtype) at the checkpoint ``indices`` of consecutive
    entries ending at ``entry_ends`` (entry tensor ids ``entry_names``, -1 for tensors this worker does not hold).
    ``writes[name]`` are the tensor's writes, ``WRITE_FIELDS`` each, padded with zero-row writes; ``maps`` holds their
    row and column offset tables. Returns each
    change's checksum contribution when ``track``."""
    num_changes, num_entries = indices.numel(), entry_ends.numel()
    contributions = torch.empty(num_changes if track else 1, dtype=torch.int64, device=indices.device)
    if num_changes == 0:
        return contributions[:0] if track else None
    grid = (triton.cdiv(num_changes, block),)
    _sparse_apply_kernel[grid](
        indices,
        deltas,
        contributions,
        entry_ends,
        entry_names,
        writes,
        num_cols,
        param_ptrs,
        maps,
        num_changes,
        num_entries,
        num_entries.bit_length() + 1,
        MAX_WRITES=writes.shape[1],
        TRACK=track,
        BITS=_TL_BITS[deltas.dtype],
        BLOCK=block,
    )
    return contributions if track else None


def warm_up(writes: torch.Tensor, num_cols: torch.Tensor, param_ptrs: torch.Tensor, maps: torch.Tensor) -> None:
    """Compile the kernel for every bits dtype before it runs inside a weight-update pause: one change of a tensor
    this worker does not hold, so nothing is written."""
    device = writes.device
    for bits in _TL_BITS:
        for track in (False, True):
            sparse_apply(
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.zeros(1, dtype=bits, device=device),
                torch.ones(1, dtype=torch.int64, device=device),
                torch.full((1,), -1, dtype=torch.int64, device=device),
                writes,
                num_cols,
                param_ptrs,
                maps,
                track,
            )
