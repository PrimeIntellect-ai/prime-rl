"""Zero-storage lazy weight placeholders for the NIXL bake.

Mimics the lazy-tensor mechanism of vLLM PR #43375 (sharded RDT weight
transfer), reduced to the one mode prime-rl needs: a *recording dry run*
against the live model. ``model.load_weights`` is driven once per worker
with one :class:`LazyWeight` per HF checkpoint tensor; vLLM's own loaders
(fused QKV, merged gate/up, FusedMoE expert routing) call their usual
``narrow``/``view``/``__getitem__``/... on the placeholder — each op is
appended to the placeholder's chain — and finally ``copy_`` it into a view
of a real parameter. That ``copy_`` is the recording sink: it captures the
source op chain and the destination view, and moves no data.

Because destination params keep their live storage (weights are updated
in place via RDMA, never re-materialized), no meta-device tricks and no
layerwise-reload plumbing are needed: the recorded destination views point
straight at the memory NIXL writes into.

Any op outside the allowlist (arithmetic, ``.to``/``.float``, ``.item``,
``.data``, bool-mask indexing) raises :class:`UnsupportedOpError` — a loud
failure instead of silently transferring the wrong bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from prime_rl.weight_transfer.chains import SUPPORTED_OPS, OpChain, OpSpec, UnsupportedOpError


@dataclass
class RecordedCopy:
    """One recorded ``copy_``: fetch ``apply_chain(hf[src_name], ops)`` and
    write it into the real-storage view ``dst``."""

    src_name: str
    ops: OpChain
    dst: torch.Tensor


@dataclass
class BakeRecorder:
    """Collects every ``copy_`` issued during one dry-run ``load_weights`` pass."""

    copies: list[RecordedCopy] = field(default_factory=list)


class LazyWeight(torch.Tensor):
    """Zero-storage tensor that records how a weight slice is consumed.

    Built via ``_make_wrapper_subclass`` so ``.shape``/``.dtype``/``.device``/
    ``.size()``/``.dim()`` work without allocating storage. Every supported op
    returns a child ``LazyWeight`` with the op appended to its chain; ``copy_``
    with a lazy source records a :class:`RecordedCopy` on the shared
    :class:`BakeRecorder` and performs no data movement.
    """

    _name: str
    _ops: OpChain
    _recorder: BakeRecorder

    @staticmethod
    def __new__(
        cls,
        name: str,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        recorder: BakeRecorder,
        ops: OpChain = (),
    ) -> "LazyWeight":
        t = torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=False,
        )
        t._name = name
        t._ops = tuple(ops)
        t._recorder = recorder
        return t

    def __repr__(self) -> str:  # noqa: D105 — repr must not touch storage
        return f"LazyWeight(name={self._name!r}, shape={tuple(self.shape)}, dtype={self.dtype}, ops={self._ops!r})"

    def _make_child(self, new_shape: torch.Size, new_dtype: torch.dtype, *new_ops: OpSpec) -> "LazyWeight":
        return LazyWeight(
            name=self._name,
            shape=new_shape,
            dtype=new_dtype,
            device=self.device,
            recorder=self._recorder,
            ops=self._ops + new_ops,
        )

    def _meta(self) -> torch.Tensor:
        """A meta tensor of this lazy's shape/dtype, used to infer post-op
        shapes via PyTorch itself instead of reimplementing shape rules."""
        return torch.empty(self.shape, dtype=self.dtype, device="meta")

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}

        # copy_ with a lazy source is the recording sink.
        if func is torch.Tensor.copy_:
            dst = args[0]
            src = args[1] if len(args) > 1 else kwargs.get("src")
            if isinstance(src, cls):
                if isinstance(dst, cls):
                    raise UnsupportedOpError(f"copy_ between two lazy weights ({src._name!r} -> {dst._name!r})")
                if tuple(dst.shape) != tuple(src.shape):
                    raise UnsupportedOpError(
                        f"copy_ shape mismatch for {src._name!r}: src {tuple(src.shape)} vs dst {tuple(dst.shape)}"
                    )
                src._recorder.copies.append(RecordedCopy(src_name=src._name, ops=src._ops, dst=dst))
                return dst

        # Allowlisted view/slice/shape ops: append to the chain, return a child.
        op_name = SUPPORTED_OPS.get(func)
        if op_name is not None:
            self_ = args[0]
            if isinstance(self_, cls):
                return cls._intercept(self_, func, op_name, tuple(args[1:]), kwargs)

        # Everything else falls through. Pure metadata reads (.shape, .size(),
        # .dim(), .numel()) never reach dispatch because the wrapper subclass
        # stored them at construction; ops that need data land in
        # __torch_dispatch__ below and raise.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def _intercept(cls, self_: "LazyWeight", func: Callable, op_name: str, args: tuple, kwargs: dict):
        """Append the op and return a child (or tuple of children for chunk)."""
        meta = self_._meta()
        with torch._C.DisableTorchFunctionSubclass():
            meta_result = func(meta, *args, **kwargs)

        base_op: OpSpec = (op_name, args, dict(kwargs))

        if isinstance(meta_result, torch.Tensor):
            return self_._make_child(meta_result.shape, meta_result.dtype, base_op)

        # Multi-return ops (chunk): one child per output, each chain followed
        # by an integer __getitem__ so the replay can index the tuple result.
        if isinstance(meta_result, (tuple, list)):
            return tuple(
                self_._make_child(m.shape, m.dtype, base_op, ("__getitem__", (i,), {}))
                for i, m in enumerate(meta_result)
            )

        raise UnsupportedOpError(f"op {op_name!r} returned a non-tensor ({type(meta_result).__name__}); cannot defer")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for value in list(args) + list(kwargs.values()):
            if isinstance(value, cls):
                raise UnsupportedOpError(
                    f"unsupported op {func} reached __torch_dispatch__ on lazy weight "
                    f"{value._name!r} (chain={value._ops!r}). Supported ops: "
                    f"{sorted(set(SUPPORTED_OPS.values()))}, plus copy_ as the sink. "
                    "Loaders that need .to(), .float(), .item(), arithmetic, bool-mask "
                    "indexing, or .data access are not supported by the NIXL weight broadcast."
                )
        return func(*args, **kwargs)
