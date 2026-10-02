"""FSDP2 tensor subclasses that make the unshard/reshard lifecycle a weight-preparation cache.

Adapted from torchtitan's ``torchtitan/quantization/_fsdp_tensor.py``, with the flat schema taken
from the keys of the dict ``prepare`` returns rather than from an operands dataclass's fields.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch
from torch import nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.tensor import DTensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes, return_and_correct_aliasing

Prepared = dict[str, torch.Tensor]
Out = Prepared | None
ShardBlocking = tuple[int | None, ...]


@runtime_checkable
class PrepareOp(Protocol):
    """Derives a whole weight's prepared tensors, filling ``out``'s tensors in place when given."""

    def prepare(self, weight: torch.Tensor, *, out: Out = None) -> Prepared: ...


@runtime_checkable
class ShardedPrepareOp(Protocol):
    """Prepares a weight's shard before the all-gather, completing the gathered set afterwards.

    ``shard_blocking`` gives, per weight axis, the extent over which the recipe shares a scale, and
    ``None`` means the recipe reduces across that whole axis.
    """

    shard_blocking: ShardBlocking

    def prepare_shard(self, shard: torch.Tensor, *, out: Out = None) -> Prepared: ...

    def complete_gathered(self, wire: Mapping[str, torch.Tensor], *, out: Out = None) -> Prepared: ...


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


def validate_prepared(prepared: Any, prepare: Callable[..., Prepared]) -> Prepared:
    if not isinstance(prepared, dict):
        raise TypeError(f"{prepare} must return a dict of tensors, got {type(prepared).__name__}.")
    if not prepared:
        raise ValueError(f"{prepare} returned no tensors.")
    for name, tensor in prepared.items():
        if not isinstance(name, str):
            raise TypeError(f"{prepare} returned a non-string key {name!r}.")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{prepare} returned a non-tensor value for {name!r}.")
        dense = (
            tensor.is_contiguous()
            and tensor.storage_offset() == 0
            and tensor.untyped_storage().size() == tensor.numel() * tensor.itemsize
        )
        if not dense:
            raise ValueError(
                f"{prepare} returned a non-dense entry for {name!r}; FSDP's alloc_storage sizes "
                "each entry's storage from numel * itemsize, so anything larger is truncated on the "
                "next unshard. Return a contiguous tensor owning its whole storage."
            )
    storages = {tensor.untyped_storage().data_ptr() for tensor in prepared.values()}
    if len(storages) != len(prepared):
        raise ValueError(
            f"{prepare} returned entries aliasing the same storage; FSDP owns each entry's "
            "storage and would free it twice. Derive such a view inside the op instead."
        )
    return prepared


def prepare_and_validate(
    prepare: Callable[..., Prepared],
    source: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    out: Out = None,
) -> Prepared:
    if out is None:
        return validate_prepared(prepare(source), prepare)
    expected = tuple(out.values())
    filled = prepare(source, out=out)
    actual = tuple(out.values())
    same_tensors = len(actual) == len(expected) and all(
        entry is original for entry, original in zip(actual, expected, strict=True)
    )
    if filled is not out or not same_tensors:
        raise RuntimeError(
            f"{prepare} replaced the tensors it was given instead of filling them; FSDP keeps the "
            "originals, so the op would read stale data. Write into out's tensors and return out."
        )
    return out


def run_prepare(
    prepare: Callable[..., Prepared],
    source: torch.Tensor,
    *,
    out: Out = None,
) -> Prepared:
    PREPARE_CALLS.increment()
    return prepare_and_validate(prepare, source, out=out)


class UnshardMetadata(NamedTuple):
    outer_size: torch.Size
    wire_keys: tuple[str, ...]


def check_sharding_is_preparable(
    blocking: ShardBlocking,
    outer_size: torch.Size,
    sharded_dim: int,
    mesh_size: int,
    name: str,
) -> None:
    """Raise unless ``name``'s shard can be prepared without seeing the rest of the weight."""
    logical = outer_size[sharded_dim]
    if logical % mesh_size != 0:
        raise ValueError(
            f"{name} is sharded unevenly: dimension {sharded_dim} has size {logical}, which the "
            f"{mesh_size}-way FSDP mesh does not divide. FSDP requires every all-gather input on a "
            "rank holding padding to have the padded sharded size, which a set of differently "
            "shaped prepared tensors can never satisfy, so preparation must wait for the gather."
        )
    extent = blocking[sharded_dim]
    if extent is None:
        raise ValueError(
            f"{name} reduces across dimension {sharded_dim}, which FSDP shards {mesh_size} ways, so "
            "one shard does not hold everything its scales are computed from."
        )
    shard_extent = logical // mesh_size
    if shard_extent % extent != 0:
        raise ValueError(
            f"{name}'s shard boundary cuts a quantization tile: dimension {sharded_dim} of size "
            f"{logical} gives each of the {mesh_size} ranks {shard_extent}, which the recipe's "
            f"blocking {extent} does not divide."
        )


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
        op: PrepareOp | ShardedPrepareOp,
        **logical_metadata: Any,
    ) -> None:
        self._tensor = tensor
        self._op = op

    @property
    def op(self) -> PrepareOp | ShardedPrepareOp:
        return self._op

    def fsdp_should_release_all_gather_outputs_after_post_all_gather(self) -> bool:
        """Named and shaped for the upstream hook added in pytorch#194114.

        FSDP calls this itself once the torch pin has that commit; until then ``_prepare_gathered``
        reads it and frees the buffer by hand. A ``ShardedPrepareOp``'s gathered tensors are the
        cache itself, so they have to outlive the hook.
        """
        return not isinstance(self._op, ShardedPrepareOp)

    def __repr__(self) -> str:
        return f"ShardedPreparedTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    def __tensor_flatten__(self):
        return ["_tensor"], (self._op,)

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
        (op,) = metadata
        return cls(inner_tensors["_tensor"], op)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        template = None
        preserve_wrapper = func in FSDP_SHARDED_OPS

        def unwrap(tensor: ShardedPreparedTensor) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif preserve_wrapper and tensor._op is not template._op:
                raise RuntimeError("FSDP operation mixed sharded tensors prepared by different ops")
            return tensor._tensor

        output = func(
            *pytree.tree_map_only(cls, unwrap, args or ()),
            **pytree.tree_map_only(cls, unwrap, kwargs or {}),
        )
        if not preserve_wrapper:
            return output
        assert template is not None
        prepare_op = template._op
        return pytree.tree_map_only(torch.Tensor, lambda t: cls(t, prepare_op), output)

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
        shard = self._tensor.to(dtype)
        if isinstance(self._op, ShardedPrepareOp):
            name = type(module).__name__
            check_sharding_is_preparable(
                self._op.shard_blocking,
                outer_size,
                sharded_dim=0,
                mesh_size=mesh.size(),
                name=name,
            )
            wire = run_prepare(self._op.prepare_shard, shard)
            if mesh.size() == 1 and len(wire) > 1:
                raise ValueError(
                    f"{name} prepares {len(wire)} tensors before the all-gather, but its FSDP mesh "
                    "has size 1, where torch copies all_gather_inputs[0] into a single output and "
                    "drops every later input. Prepare after the gather at this geometry."
                )
            return tuple(wire.values()), UnshardMetadata(outer_size, tuple(wire))
        padded_rows = math.ceil(outer_size[0] / mesh.size())
        if shard.size(0) != padded_rows:
            source = shard.new_zeros((padded_rows, *shard.shape[1:]))
            source.narrow(0, 0, shard.size(0)).copy_(shard)
        else:
            source = shard
        return (source,), UnshardMetadata(outer_size, ())

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        # Under activation checkpointing this hook fires inside the recompute region, where the
        # first unshard allocates and every later one refills in place, so the two record different
        # op sequences and a sequence taken with the ambient dispatch modes on would not line up.
        with _disable_current_modes():
            if metadata.wire_keys:
                return self._complete_gathered(all_gather_outputs, metadata, param_dtype, out)
            return self._prepare_gathered(all_gather_outputs, metadata, out)

    def _prepare_gathered(self, all_gather_outputs, metadata, out):
        (gathered,) = all_gather_outputs
        logical_tensor = gathered
        # An unevenly sharded parameter gathers padding rows past the logical size, which would
        # reach prepare as real data.
        if logical_tensor.size(0) != metadata.outer_size[0]:
            logical_tensor = logical_tensor.narrow(0, 0, metadata.outer_size[0])

        if out is None:
            with torch.no_grad():
                prepared = run_prepare(self._op.prepare, logical_tensor)
            unsharded = UnshardedPreparedTensor(logical_tensor, prepared)
            self._release_gather_buffer(gathered)
            return unsharded, tuple(prepared.values())

        self._refill(run_prepare, self._op.prepare, logical_tensor, out)
        self._release_gather_buffer(gathered)

    def _complete_gathered(self, all_gather_outputs, metadata, param_dtype, out):
        wire = dict(zip(metadata.wire_keys, all_gather_outputs, strict=True))
        if out is not None:
            self._refill(prepare_and_validate, self._op.complete_gathered, wire, out)
            return
        with torch.no_grad():
            prepared = prepare_and_validate(self._op.complete_gathered, wire)
        prepared_ids = {id(tensor) for tensor in prepared.values()}
        dropped = [name for name, tensor in wire.items() if id(tensor) not in prepared_ids]
        if dropped:
            raise RuntimeError(
                f"{self._op} dropped the gathered tensors {dropped} in complete_gathered; FSDP "
                "transports them on every unshard, so nothing would ever read them. Return each "
                "gathered tensor, or stop sending it over the wire."
            )
        unsharded = UnshardedPreparedTensor(
            next(iter(prepared.values())),
            prepared,
            logical_size=metadata.outer_size,
            logical_stride=make_contiguous_strides_for(metadata.outer_size),
            logical_dtype=param_dtype,
        )
        gathered_ids = {id(tensor) for tensor in all_gather_outputs}
        # FSDP allocates and frees the gathered tensors itself, so naming one here manages it twice.
        derived = tuple(tensor for tensor in prepared.values() if id(tensor) not in gathered_ids)
        return unsharded, derived

    def _refill(self, runner, prepare, source, out) -> None:
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
            runner(prepare, source, out=existing)

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
            setattr(self, f"prepared_{name}", tensor)

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
        return [f"prepared_{name}" for name in names], (names, self.dtype)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        names, dtype = metadata
        prepared = {name: inner_tensors[f"prepared_{name}"] for name in names}
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


def unwrapped_master_shard(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` with any preparation wrapper replaced by a plain alias of the master shard."""
    if isinstance(tensor, DTensor):
        local = tensor._local_tensor
        if not isinstance(local, ShardedPreparedTensor):
            return tensor
        return DTensor(local._tensor.detach(), tensor._spec, requires_grad=tensor.requires_grad)
    if isinstance(tensor, ShardedPreparedTensor):
        return tensor._tensor.detach()
    return tensor


def unwrap_prepared_state_dict_entries(module, state_dict, prefix, local_metadata) -> None:
    """Keep the preparation wrapper out of the state dict, which is what reaches a checkpoint.

    Torch stamps an attribute onto every hook it registers, so this has to be a plain function.
    """
    for key in list(state_dict):
        if key.startswith(prefix):
            state_dict[key] = unwrapped_master_shard(state_dict[key])


def install_prepared_weights(module: nn.Module, ops: Mapping[str, PrepareOp | ShardedPrepareOp]) -> None:
    """Wrap the named parameters of ``module`` in place, before ``fully_shard``."""
    for name, op in ops.items():
        if not isinstance(op, (PrepareOp, ShardedPrepareOp)):
            raise TypeError(
                f"{type(module).__name__}.{name} was given {op!r}, which is neither a PrepareOp "
                "(a prepare method) nor a ShardedPrepareOp (shard_blocking, prepare_shard and "
                "complete_gathered)."
            )
        parameter = getattr(module, name, None)
        if not isinstance(parameter, nn.Parameter):
            raise ValueError(f"{type(module).__name__} has no parameter {name!r} to prepare.")
        if isinstance(parameter.data, ShardedPreparedTensor):
            raise ValueError(f"{type(module).__name__}.{name} is already wrapped for preparation.")
        module.register_parameter(
            name,
            nn.Parameter(
                ShardedPreparedTensor(parameter.data, op),
                requires_grad=parameter.requires_grad,
            ),
        )
    module.register_state_dict_post_hook(unwrap_prepared_state_dict_entries)


def unsharded_prepared_or_none(weight: torch.Tensor) -> UnshardedPreparedTensor | None:
    """``weight``'s unsharded wrapper, whose prepared tensors an op reads as ``prepared_<name>``."""
    local = weight.to_local() if isinstance(weight, DTensor) else weight
    if isinstance(local, UnshardedPreparedTensor):
        return local
    if isinstance(local, ShardedPreparedTensor):
        raise RuntimeError("An op read a ShardedPreparedTensor, so it ran outside its weights' unshard scope.")
    return None

