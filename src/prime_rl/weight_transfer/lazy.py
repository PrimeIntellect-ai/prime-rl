"""Zero-storage lazy weight placeholders for the NIXL bake.

Mimics the lazy-tensor mechanism of vLLM PR #43375 (sharded RDT weight
transfer): a *recording dry run* through vLLM's own layerwise-reload path.
``model.load_weights`` is driven once with one :class:`LazyWeight` per HF
checkpoint tensor; vLLM's loaders (fused QKV, merged gate/up, FusedMoE expert
routing) call their usual ``narrow``/``view``/``__getitem__``/... on the
placeholder — each op is appended to the placeholder's chain — and finally
``copy_`` it into a view of a (meta) parameter. That ``copy_`` is the
recording sink: it captures the source op chain plus the destination's owning
``(module, param_name)`` and its ``offset/shape/stride``, and moves no data.

The destination is recorded against the param layout that exists at *load*
time — i.e. before ``process_weights_after_loading``. For an online-fp8 model
that layout is bf16, so per sync the worker materializes bf16 params, fills
them with the pulled slices, and re-runs ``process_weights_after_loading`` to
re-quantize to fp8 — exactly as a normal vLLM weight reload.

Any op outside the allowlist (arithmetic, ``.to``/``.float``, ``.item``,
``.data``, bool-mask indexing) raises :class:`UnsupportedOpError` — a loud
failure instead of silently transferring the wrong bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from prime_rl.weight_transfer.chains import SUPPORTED_OPS, OpChain, OpSpec, UnsupportedOpError


@dataclass
class RecordedCopy:
    """One recorded ``copy_``: the source slice ``apply_chain(src_name, ops)``
    lands in ``param_name`` of ``layer`` at the strided destination
    ``(offset, shape, stride)`` (captured from the meta destination view, valid
    without storage)."""

    src_name: str
    ops: OpChain
    layer: Any
    param_name: str
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


@dataclass
class BakeRecorder:
    """Collects every ``copy_`` issued during one dry-run ``load_weights`` pass.

    ``current`` is the ``(module, param_name)`` stamp the engine sets around
    each param's loader so the lazy ``copy_`` can attribute its destination; a
    ``copy_`` with no stamp is left unattributed (its group falls back)."""

    copies: list[RecordedCopy] = field(default_factory=list)
    current: "tuple[Any, str] | None" = None


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
                current = src._recorder.current
                if current is not None:
                    layer, param_name = current
                    src._recorder.copies.append(
                        RecordedCopy(
                            src_name=src._name,
                            ops=src._ops,
                            layer=layer,
                            param_name=param_name,
                            offset=dst.storage_offset(),
                            shape=tuple(dst.shape),
                            stride=tuple(dst.stride()),
                        )
                    )
                # Fire a meta copy_ so layerwise's load-numel counter still
                # advances (otherwise the layer never reaches "fully loaded").
                meta_src = torch.empty(src.shape, dtype=src.dtype, device="meta")
                with torch._C.DisableTorchFunctionSubclass():
                    return dst.copy_(meta_src)

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
