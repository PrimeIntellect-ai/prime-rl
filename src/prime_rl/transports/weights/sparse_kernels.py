"""Triton kernels that diff a parameter update in place: no host sync, no workspace beyond the old values."""

from typing import NamedTuple

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 4096


@triton.jit
def _load(
    old_base, old_offsets, addresses, lengths, block_pieces, block_starts, NEW_FP32: tl.constexpr, SAME: tl.constexpr
):
    pid = tl.program_id(0)
    piece = tl.load(block_pieces + pid)
    offsets = tl.load(block_starts + pid) + tl.arange(0, 4096)
    mask = offsets < tl.load(lengths + piece)
    if NEW_FP32:
        new = tl.load(tl.load(addresses + piece).to(tl.pointer_type(tl.float32)) + offsets, mask=mask, other=0)
        new = new.to(tl.bfloat16).to(tl.int16, bitcast=True)
    else:
        new = tl.load(tl.load(addresses + piece).to(tl.pointer_type(tl.int16)) + offsets, mask=mask, other=0)
    if SAME:
        old = new
    else:
        old = tl.load(old_base + tl.load(old_offsets + piece) + offsets, mask=mask, other=0)
    return pid, piece, offsets, mask, old, new


@triton.jit
def _count_kernel(
    old_base,
    old_offsets,
    addresses,
    lengths,
    firsts,
    segments,
    block_pieces,
    block_starts,
    counts,
    sums,
    NEW_FP32: tl.constexpr,
    SAME: tl.constexpr,
):
    pid, piece, offsets, mask, old, new = _load(
        old_base, old_offsets, addresses, lengths, block_pieces, block_starts, NEW_FP32, SAME
    )
    tl.store(counts + pid, tl.sum(((old != new) & mask).to(tl.int32)))
    weights = tl.load(firsts + piece) + offsets + 1
    tl.atomic_add(sums + tl.load(segments + piece), tl.sum(tl.where(mask, weights * new.to(tl.int64), 0)))


@triton.jit(do_not_specialize=["skip", "capacity"])
def _write_kernel(
    old_base,
    old_offsets,
    addresses,
    lengths,
    firsts,
    block_pieces,
    block_starts,
    ends,
    indices,
    values,
    skip,
    capacity,
    NEW_FP32: tl.constexpr,
):
    pid, piece, offsets, mask, old, new = _load(
        old_base, old_offsets, addresses, lengths, block_pieces, block_starts, NEW_FP32, False
    )
    changed = ((old != new) & mask).to(tl.int32)
    position = tl.load(ends + pid) - tl.sum(changed) + tl.cumsum(changed, 0) - changed - skip
    store = (changed != 0) & (position >= 0) & (position < capacity)
    tl.store(indices + position, (tl.load(firsts + piece) + offsets).to(tl.int32), mask=store)
    delta = new - old
    tl.store(values + position, (delta << 1) ^ (delta >> 15), mask=store)


class Pieces(NamedTuple):
    """Contiguous runs of bf16 wire bits, each within one HF tensor: ``names``, the first block of each, and the
    device tables of the kernels."""

    names: list[str]
    first_blocks: np.ndarray
    new_fp32: bool
    segments: torch.Tensor
    tables: tuple[torch.Tensor, ...]


def build_pieces(rows: list[tuple[str, int, int, int, int, int]], new_fp32: bool, device) -> Pieces:
    """``rows`` of ``(hf_name, offset in the old values, address of the new values, length, HF flat index of the
    first element, segment)``."""
    names, old_offsets, addresses, lengths, firsts, segments = zip(*rows)
    num_blocks = [-(-length // BLOCK) for length in lengths]
    first_blocks = np.cumsum([0, *num_blocks[:-1]])
    block_pieces = np.repeat(np.arange(len(rows), dtype=np.int32), num_blocks)
    block_starts = (np.arange(len(block_pieces)) - first_blocks[block_pieces]) * BLOCK

    def table(values, dtype=torch.int64):
        return torch.tensor(np.asarray(values), dtype=dtype).to(device)

    segments = table(segments)
    tables = (table(old_offsets), table(addresses), table(lengths), table(firsts), segments)
    return Pieces(
        list(names), first_blocks, new_fp32, segments, (*tables, table(block_pieces, torch.int32), table(block_starts))
    )


def count_changes(pieces: Pieces, old: torch.Tensor | None, counts: torch.Tensor, sums: torch.Tensor) -> None:
    """Changed elements per block into ``counts``, and add each piece's ``bits_checksum`` of the new values to
    ``sums[segment]``; without ``old``, only the checksums."""
    old_offsets, addresses, lengths, firsts, segments, block_pieces, block_starts = pieces.tables
    _count_kernel[(len(block_pieces),)](
        addresses if old is None else old,
        old_offsets,
        addresses,
        lengths,
        firsts,
        segments,
        block_pieces,
        block_starts,
        counts,
        sums,
        NEW_FP32=pieces.new_fp32,
        SAME=old is None,
    )


def write_changes(
    pieces: Pieces, old: torch.Tensor, ends: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, skip: int
) -> None:
    """Changes ``skip`` to ``skip + len(indices)`` (``ends``: inclusive cumulative block counts): their HF flat
    indices and ``zigzag`` bit differences."""
    old_offsets, addresses, lengths, firsts, _, block_pieces, block_starts = pieces.tables
    _write_kernel[(len(block_pieces),)](
        old,
        old_offsets,
        addresses,
        lengths,
        firsts,
        block_pieces,
        block_starts,
        ends,
        indices,
        values,
        skip,
        len(indices),
        NEW_FP32=pieces.new_fp32,
    )
