"""FSDP2 tensor subclasses that make the unshard/reshard lifecycle a weight-preparation cache.

Adapted from torchtitan's ``torchtitan/quantization/_fsdp_tensor.py``, with the flat schema taken
from the keys of the dict ``prepare`` returns rather than from an operands dataclass's fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes, return_and_correct_aliasing

PrepareFn = Callable[[torch.Tensor], dict[str, torch.Tensor]]

FSDP_SHARDED_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.c10d.scatter_.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.alias.default,
    # Unreachable today, since configure_moe_runtime forbids num_experts % ep and every rank joins
    # the mesh; kept as insurance and probably deletable.
    torch.ops.aten.new_empty.default,
    torch.ops.aten.constant_pad_nd.default,
}

FSDP_UNSHARDED_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
}

FSDP_UNSHARDED_FACTORY_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros_like.default,
}


# Instrumentation only: lets the tests and the A/B runs assert how often prepare actually ran.
class PrepareCallCounter:
    def __init__(self) -> None:
        self.count = 0

    def increment(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


PREPARE_CALLS = PrepareCallCounter()


def validate_prepared(prepared: Any, prepare_fn: PrepareFn) -> dict[str, torch.Tensor]:
    if not isinstance(prepared, dict):
        raise TypeError(f"{prepare_fn} must return a dict of tensors, got {type(prepared).__name__}.")
    if not prepared:
        raise ValueError(f"{prepare_fn} returned no tensors.")
    for name, tensor in prepared.items():
        if not isinstance(name, str):
            raise TypeError(f"{prepare_fn} returned a non-string key {name!r}.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{prepare_fn} returned a non-tensor value for {name!r}.")
    storages = {tensor.untyped_storage().data_ptr() for tensor in prepared.values()}
    if len(storages) != len(prepared):
        raise ValueError(
            f"{prepare_fn} returned entries aliasing the same storage; FSDP owns each entry's "
            "storage and would free it twice. Derive such a view inside the op instead."
        )
    return prepared


def run_prepare(prepare_fn: PrepareFn, logical_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    PREPARE_CALLS.increment()
    return validate_prepared(prepare_fn(logical_tensor), prepare_fn)


class PreparedTensorBase(torch.Tensor):
    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            kwargs.get("logical_size", tensor.size()),
            strides=kwargs.get("logical_stride", tensor.stride()),
            storage_offset=kwargs.get("logical_storage_offset", tensor.storage_offset()),
            dtype=kwargs.get("logical_dtype", tensor.dtype),
            layout=tensor.layout,
            device=kwargs.get("logical_device", tensor.device),
            pin_memory=tensor.is_pinned(),
            requires_grad=kwargs.get("logical_requires_grad", tensor.requires_grad),
        )


class ShardedPreparedTensor(PreparedTensorBase):
    """The persistent parameter: the high-precision shard plus the FSDP all-gather hooks."""

    def __init__(
        self,
        tensor: torch.Tensor,
        prepare_fn: PrepareFn,
        release_all_gather_outputs: bool = True,
        **logical_metadata: Any,
    ) -> None:
        self._tensor = tensor
        self._prepare_fn = prepare_fn
        self._release_all_gather_outputs = release_all_gather_outputs

    @property
    def prepare_fn(self) -> PrepareFn:
        return self._prepare_fn

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        """Named and shaped for the upstream hook added in pytorch#194114.

        FSDP calls this itself once the torch pin has that commit; until then ``_post_all_gather``
        reads it and frees the buffer by hand.
        """
        return self._release_all_gather_outputs

    def __repr__(self) -> str:
        return f"ShardedPreparedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    def __tensor_flatten__(self):
        return ["_tensor"], (self._prepare_fn, self._release_all_gather_outputs)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        prepare_fn, release_all_gather_outputs = metadata
        return cls(inner_tensors["_tensor"], prepare_fn, release_all_gather_outputs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        template = None
        preserve_wrapper = func in FSDP_SHARDED_OPS

        def unwrap(tensor: ShardedPreparedTensor) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif preserve_wrapper and tensor._prepare_fn is not template._prepare_fn:
                raise RuntimeError("FSDP operation mixed sharded tensors with different prepare callables")
            return tensor._tensor

        output = func(
            *pytree.tree_map_only(cls, unwrap, args or ()),
            **pytree.tree_map_only(cls, unwrap, kwargs or {}),
        )
        if not preserve_wrapper:
            return output
        assert template is not None
        prepare_fn = template._prepare_fn
        release = template._release_all_gather_outputs
        return pytree.tree_map_only(torch.Tensor, lambda t: cls(t, prepare_fn, release), output)

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        sharded_dims = [
            dim
            for dim, (local, logical) in enumerate(zip(self._tensor.shape, outer_size, strict=True))
            if local != logical
        ]
        if len(sharded_dims) > 1:
            raise RuntimeError(
                f"FSDP sharded more than one dimension: local {tuple(self._tensor.shape)} "
                f"against logical {tuple(outer_size)}"
            )
        if sharded_dims and sharded_dims[0] != 0:
            raise NotImplementedError(
                f"Prepared weights support sharding dimension 0 only, but this parameter of shape "
                f"{tuple(outer_size)} is sharded on dimension {sharded_dims[0]}."
            )
        dtype = mp_policy.param_dtype or self._tensor.dtype
        padded_rows = math.ceil(outer_size[0] / mesh.size())
        if self._tensor.size(0) != padded_rows:
            source = self._tensor.new_zeros((padded_rows, *self._tensor.shape[1:]), dtype=dtype)
            source.narrow(0, 0, self._tensor.size(0)).copy_(self._tensor)
        else:
            source = self._tensor.to(dtype)
        return (source,), outer_size

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        # Under activation checkpointing this hook fires inside the recompute region, where the
        # first unshard allocates and every later one copies, so a recorded op sequence taken with
        # the ambient dispatch modes on would not line up with the forward's.
        with _disable_current_modes():
            return self._post_all_gather(all_gather_outputs, metadata, out)

    def _post_all_gather(self, all_gather_outputs, metadata, out):
        (gathered,) = all_gather_outputs
        logical_tensor = gathered
        # An unevenly sharded parameter gathers padding rows past the logical size, which would
        # reach prepare as real data.
        if metadata is not None and logical_tensor.size(0) != metadata[0]:
            logical_tensor = logical_tensor.narrow(0, 0, metadata[0])

        if out is None:
            with torch.no_grad():
                prepared = run_prepare(self._prepare_fn, logical_tensor)
            unsharded = UnshardedPreparedTensor(logical_tensor, prepared)
            self._release_gather_buffer(gathered)
            return unsharded, tuple(prepared.values())

        target = out._local_tensor if isinstance(out, DTensor) else out
        if not isinstance(target, UnshardedPreparedTensor):
            raise RuntimeError(f"FSDP refill target is not an UnshardedPreparedTensor: {type(target).__name__}")
        existing = target.prepared_storage
        managed = tuple(existing.values())
        with (
            torch.no_grad(),
            # A refill must not invalidate the version checks of tensors an op saved for backward.
            torch.autograd._unsafe_preserve_version_counter(managed),
        ):
            refilled = run_prepare(self._prepare_fn, logical_tensor)
            if refilled.keys() != existing.keys():
                raise RuntimeError(
                    f"prepare returned keys {sorted(refilled)} on refill but {sorted(existing)} on the "
                    "first unshard; the schema must be constant for a parameter."
                )
            for name, tensor in refilled.items():
                existing[name].copy_(tensor)
        self._release_gather_buffer(gathered)

    def _release_gather_buffer(self, gathered: torch.Tensor) -> None:
        if self.fsdp_should_release_all_gather_outputs_after_post_all_gather():
            gathered.untyped_storage().resize_(0)


class UnshardedPreparedTensor(PreparedTensorBase):
    """What an op sees between unshard and reshard: the prepared tensors, no high-precision storage."""

    def __init__(
        self,
        metadata_source: torch.Tensor,
        prepared: dict[str, torch.Tensor],
        **logical_metadata: Any,
    ) -> None:
        self._prepared = prepared
        # __tensor_flatten__ reports inner tensors by attribute name and the subclass machinery
        # fetches them with a plain getattr, so each entry has to exist as an attribute too.
        for name, tensor in prepared.items():
            setattr(self, f"_prepared_{name}", tensor)

    @property
    def prepared(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(self._prepared)

    @property
    def prepared_storage(self) -> dict[str, torch.Tensor]:
        return self._prepared

    def __repr__(self) -> str:
        return (
            f"UnshardedPreparedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, prepared={sorted(self._prepared)})"
        )

    def __tensor_flatten__(self):
        names = tuple(self._prepared)
        return [f"_prepared_{name}" for name in names], (names, self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        names, dtype = metadata
        prepared = {name: inner_tensors[f"_prepared_{name}"] for name in names}
        return UnshardedPreparedTensor(
            next(iter(prepared.values())),
            prepared,
            logical_size=outer_size,
            logical_stride=outer_stride,
            logical_dtype=dtype,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        template = None

        def unwrap(tensor: UnshardedPreparedTensor) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif tensor._prepared is not template._prepared:
                raise RuntimeError("FSDP operation mixed unsharded tensors from different unshards")
            # No high-precision storage to hand the op; a meta tensor carries what view ops need.
            return torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device="meta",
                requires_grad=tensor.requires_grad,
            )

        def wrap_view(tensor: torch.Tensor):
            assert template is not None
            prepared = template._prepared
            layout_source = next(iter(prepared.values()))
            return UnshardedPreparedTensor(
                layout_source,
                prepared,
                logical_size=tensor.size(),
                logical_stride=tensor.stride(),
                logical_storage_offset=tensor.storage_offset(),
                logical_dtype=template.dtype,
                logical_device=template.device,
                logical_requires_grad=tensor.requires_grad,
            )

        original_args, original_kwargs = args, kwargs or {}
        args, kwargs = pytree.tree_map_only(cls, unwrap, (original_args, original_kwargs))
        assert template is not None
        if func in FSDP_UNSHARDED_FACTORY_OPS:
            kwargs["device"] = template.device
            return func(*args, **kwargs)
        if func not in FSDP_UNSHARDED_VIEW_OPS:
            raise RuntimeError(f"{func} attempted to read a storage-free UnshardedPreparedTensor")
        wrapped = pytree.tree_map_only(torch.Tensor, wrap_view, func(*args, **kwargs))
        return return_and_correct_aliasing(func, original_args, original_kwargs, wrapped)


def install_prepared_weights(
    module: nn.Module,
    prepare_fns: Mapping[str, PrepareFn],
    *,
    release_all_gather_outputs: bool = True,
) -> None:
    """Wrap the named parameters of ``module`` in place, before ``fully_shard``."""
    if not release_all_gather_outputs:
        # TODO: add support
        # release=False means the op still needs the gathered high-precision weight, which we cannot
        # honor: UnshardedPreparedTensor drops it, and that is what makes releasing safe.
        # Supporting it means retaining the gathered tensor and exposing it next to .prepared, but
        # never listing it in __tensor_flatten__, since FSDP already owns that storage.
        raise NotImplementedError(
            "release_all_gather_outputs=False is not supported: the unsharded wrapper keeps no "
            "high-precision storage, so an op has no way to read the gathered weight."
        )
    for name, prepare_fn in prepare_fns.items():
        parameter = getattr(module, name, None)
        if not isinstance(parameter, nn.Parameter):
            raise ValueError(f"{type(module).__name__} has no parameter {name!r} to prepare.")
        if isinstance(parameter.data, ShardedPreparedTensor):
            raise ValueError(f"{type(module).__name__}.{name} is already wrapped for preparation.")
        module.register_parameter(
            name,
            nn.Parameter(
                ShardedPreparedTensor(parameter.data, prepare_fn, release_all_gather_outputs),
                requires_grad=parameter.requires_grad,
            ),
        )


def prepared_or_none(weight: torch.Tensor) -> Mapping[str, torch.Tensor] | None:
    local = weight.to_local() if isinstance(weight, DTensor) else weight
    if isinstance(local, UnshardedPreparedTensor):
        return local.prepared
    if isinstance(local, ShardedPreparedTensor):
        raise RuntimeError("An op read a ShardedPreparedTensor, so it ran outside its weights' unshard scope.")
    return None
