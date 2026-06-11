"""Op chains: the slice language produced by the lazy-tensor bake.

During the bake, vLLM's weight loaders run against zero-storage
:class:`~prime_rl.weight_transfer.lazy.LazyWeight` placeholders and each
allowlisted view/slice op is recorded as an :data:`OpSpec`. The worker then
*resolves* every recorded chain into a strided region of the trainer's
published source tensor, so the slice can be pulled with raw RDMA READs —
no replay, no staging, no conversion.

Ops are restricted to a closed allowlist of pure view / shape ops so a chain
can never compute on (and silently corrupt) weight data. Anything outside the
allowlist raises during the bake — loud failure over wrong bytes.
"""

from __future__ import annotations

from typing import Any

import torch

# A single recorded op: (op_name, positional_args, kwargs).
OpSpec = tuple[str, tuple[Any, ...], dict[str, Any]]
OpChain = tuple[OpSpec, ...]

# torch.Tensor methods that map to pure-view / shape-only operations. Every
# entry maps `torch.Tensor.fn` to the name used to reach the method via
# getattr on resolution. Anything that escapes this set (arithmetic,
# .to/.float, .item, .data, bool-mask indexing) raises during the bake.
SUPPORTED_OPS: dict[Any, str] = {
    torch.Tensor.narrow: "narrow",
    torch.Tensor.view: "view",
    torch.Tensor.reshape: "reshape",
    torch.Tensor.__getitem__: "__getitem__",
    torch.Tensor.unsqueeze: "unsqueeze",
    torch.Tensor.squeeze: "squeeze",
    torch.Tensor.transpose: "transpose",
    torch.Tensor.t: "t",
    torch.Tensor.permute: "permute",
    torch.Tensor.flatten: "flatten",
    torch.Tensor.contiguous: "contiguous",
    torch.Tensor.chunk: "chunk",
}
SUPPORTED_OP_NAMES = frozenset(SUPPORTED_OPS.values())

# Bound on RDMA descriptors for a single recorded copy. A region whose
# innermost dimension is non-contiguous decomposes into per-element runs;
# this cap turns that pathology into a loud error instead of a transfer
# plan with millions of descriptors.
MAX_RUNS_PER_COPY = 1 << 16


class UnsupportedOpError(NotImplementedError):
    """A weight loader used an op (or op argument) outside the allowlist."""


def apply_chain(tensor: torch.Tensor, ops: OpChain) -> torch.Tensor:
    """Apply a recorded op chain to a tensor.

    Multi-return ops (``chunk``) yield a tuple; the recorded chain always
    follows them with an integer ``__getitem__``, which plain ``getattr``
    dispatch handles on the tuple as well.
    """
    result: Any = tensor
    for name, args, kwargs in ops:
        if name not in SUPPORTED_OP_NAMES:
            raise UnsupportedOpError(f"op {name!r} is not in the supported op set {sorted(SUPPORTED_OP_NAMES)}")
        result = getattr(result, name)(*args, **kwargs)
    if not isinstance(result, torch.Tensor):
        raise UnsupportedOpError(f"op chain produced a non-tensor ({type(result).__name__})")
    return result


def resolve_chain_region(
    shape: tuple[int, ...], dtype: torch.dtype, ops: OpChain
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Resolve an op chain to a strided region of its root tensor.

    Simulates the chain on a meta tensor and verifies the result is still a
    *view* of the root (shares its zero-byte storage). Returns
    ``(storage_offset_elems, shape, stride)`` of the resulting view. A chain
    that breaks view-ness (e.g. ``contiguous()`` on a non-contiguous
    intermediate, which would allocate) cannot be expressed as raw RDMA over
    the source tensor and raises.
    """
    root = torch.empty(shape, dtype=dtype, device="meta")
    result = apply_chain(root, ops)
    if result is not root and result._base is not root:
        raise UnsupportedOpError(
            f"op chain {ops!r} on shape {shape} does not resolve to a view of the source tensor "
            "(an op materialized a copy); this loader cannot be served by RDMA pulls"
        )
    return result.storage_offset(), tuple(result.shape), tuple(result.stride())


def region_elem_runs(
    offset_elems: int,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Decompose a strided region into ``(elem_offset, num_elems)`` runs.

    Runs are emitted in C iteration order of the region, i.e. the element
    order of a ``.contiguous()`` materialization of the same logical tensor.
    Two regions of one ``copy_`` (source and destination) therefore produce
    byte streams that correspond element by element. Offsets are in element
    units relative to the root tensor's storage origin, addresses left to the
    caller (the source tensor is sharded; the destination is a live view).
    """
    numel = 1
    for s in shape:
        numel *= s
    if numel == 0:
        return []
    dims = [(s, st) for s, st in zip(shape, stride) if s != 1]
    if any(st < 0 for _, st in dims):
        raise UnsupportedOpError("negative strides are not supported for RDMA regions")
    if not dims:
        return [(offset_elems, 1)]

    # Fold the contiguous suffix of dims into one run.
    run_elems = 1
    k = len(dims)
    while k > 0 and dims[k - 1][1] == run_elems:
        run_elems *= dims[k - 1][0]
        k -= 1
    outer = dims[:k]

    num_runs = 1
    for size, _ in outer:
        num_runs *= size
    if num_runs > MAX_RUNS_PER_COPY:
        raise UnsupportedOpError(
            f"region of shape {shape} / stride {stride} decomposes into {num_runs} runs (max {MAX_RUNS_PER_COPY})"
        )

    runs: list[tuple[int, int]] = []

    def emit(dim: int, offset: int) -> None:
        if dim == len(outer):
            runs.append((offset, run_elems))
            return
        size, stride_ = outer[dim]
        for i in range(size):
            emit(dim + 1, offset + i * stride_)

    emit(0, offset_elems)
    return runs


def tensor_runs(view: torch.Tensor) -> list[tuple[int, int]]:
    """``(device_addr, num_bytes)`` runs for a live tensor view, in C order."""
    esize = view.element_size()
    base = view.data_ptr()
    return [(base + off * esize, n * esize) for off, n in region_elem_runs(0, tuple(view.shape), tuple(view.stride()))]


def match_runs(src_runs: list[tuple[int, int]], dst_runs: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Zip two run decompositions of the same byte stream into transfer pairs.

    Returns ``(src_addr, dst_addr, num_bytes)`` triples, splitting runs at
    each other's boundaries. Both inputs must cover the same total length.
    """
    pairs: list[tuple[int, int, int]] = []
    i = j = 0
    src_off = dst_off = 0
    while i < len(src_runs) and j < len(dst_runs):
        length = min(src_runs[i][1] - src_off, dst_runs[j][1] - dst_off)
        pairs.append((src_runs[i][0] + src_off, dst_runs[j][0] + dst_off, length))
        src_off += length
        dst_off += length
        if src_off == src_runs[i][1]:
            i += 1
            src_off = 0
        if dst_off == dst_runs[j][1]:
            j += 1
            dst_off = 0
    if i != len(src_runs) or j != len(dst_runs):
        raise ValueError(f"run length mismatch: src={sum(n for _, n in src_runs)}B dst={sum(n for _, n in dst_runs)}B")
    return pairs
