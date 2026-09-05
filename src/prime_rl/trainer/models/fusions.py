from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.distributed.tensor import Shard

from prime_rl.utils.logger import get_logger
from prime_rl.utils.weights import resolve_fqn

# Replaces some of a module's parameters with a packed one. Advertised by modules
# through their ``supported_fusions``.
RuntimeFusion = Callable[[nn.Module], None]


class PackedParameterSpec:
    """One physical parameter packing several logical parameters along a dimension.

    Training allocates, computes on, and optimizes the packed parameter. The logical
    parameters keep their canonical names in checkpoints. They alias the packed storage
    unless the parameter is a DTensor sharded along the packing dimension, where a split
    has to redistribute and yields copies instead.

    ``name`` and ``logical_names`` are relative to the module the packing is registered
    on, and may name parameters of its children. ``dim`` is a non-negative dimension
    index into the packed parameter.
    """

    def __init__(self, name: str, logical_names: Sequence[str], sizes: Sequence[int], dim: int):
        self.name = name
        self.logical_names = tuple(logical_names)
        self.sizes = tuple(sizes)
        self.dim = dim

    def split_into_logical_views(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The logical parameters, in ``logical_names`` order, as views of ``tensor``."""
        return tensor.split(self.sizes, dim=self.dim)

    def pack_logical_tensors(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(tensors), dim=self.dim)

    def muon_matrix_partitions(self, parameter: torch.Tensor) -> tuple[int, ...] | None:
        """Muon's partition sizes, or None when the packing is not on the output dimension."""
        return self.sizes if self.dim == parameter.ndim - 2 else None


def register_packed_parameter_state_dict_hooks(module: nn.Module, spec: PackedParameterSpec) -> None:
    """Record a packed parameter and keep the module's state dict in canonical form.

    The hooks below trade the packed entry for its canonical ones on the way out and
    back again on the way in, so checkpoints never see the packed layout. Torch stamps
    an attribute onto every hook it registers, so these have to be plain functions.
    """

    def split_packed_entry_into_logical_entries(module, state_dict, prefix, local_metadata) -> None:
        views = spec.split_into_logical_views(state_dict.pop(f"{prefix}{spec.name}"))
        for logical_name, view in zip(spec.logical_names, views):
            state_dict[f"{prefix}{logical_name}"] = view

    def join_logical_entries_into_packed_entry(module, state_dict, prefix, *load_state_dict_args) -> None:
        state_dict[f"{prefix}{spec.name}"] = spec.pack_logical_tensors(
            [state_dict.pop(f"{prefix}{logical_name}") for logical_name in spec.logical_names]
        )

    module.packed_parameter_specs = (*getattr(module, "packed_parameter_specs", ()), spec)
    module.register_state_dict_post_hook(split_packed_entry_into_logical_entries)
    module.register_load_state_dict_pre_hook(join_logical_entries_into_packed_entry)


def fuse_gate_up_projections(module: nn.Module) -> None:
    """Pack a gated expert's gate and up projections into one grouped-GEMM weight.

    The packed weight is ``[num_experts, 2 * hidden_dim, dim]``, so the forward path is a
    single grouped GEMM whose transpose and chunk are views.
    """
    spec = PackedParameterSpec("gate_up_proj", ("gate_proj", "up_proj"), (module.hidden_dim, module.hidden_dim), dim=1)
    module.gate_up_proj = nn.Parameter(
        spec.pack_logical_tensors([module.gate_proj, module.up_proj]),
        requires_grad=module.gate_proj.requires_grad,
    )
    module.gate_proj = None
    module.up_proj = None
    register_packed_parameter_state_dict_hooks(module, spec)


def fuse_qkv_projections(module: nn.Module) -> None:
    """Pack an attention module's query, key and value projections into one linear layer.

    The packed weight is ``[q_size + k_size + v_size, hidden_size]``, so the projections
    become a single GEMM whose output is split along the last dimension.
    """
    projections = {"q_proj": module.q_proj, "k_proj": module.k_proj, "v_proj": module.v_proj}
    sizes = tuple(projection.out_features for projection in projections.values())
    has_bias = module.q_proj.bias is not None

    # Built on meta so the packed parameters assigned below are the only allocation
    qkv_proj = nn.Linear(module.q_proj.in_features, sum(sizes), bias=has_bias, device="meta")
    for parameter_name in ("weight", "bias") if has_bias else ("weight",):
        spec = PackedParameterSpec(
            f"qkv_proj.{parameter_name}",
            [f"{name}.{parameter_name}" for name in projections],
            sizes,
            dim=0,
        )
        tensors = [getattr(projection, parameter_name) for projection in projections.values()]
        setattr(
            qkv_proj,
            parameter_name,
            nn.Parameter(spec.pack_logical_tensors(tensors), requires_grad=tensors[0].requires_grad),
        )
        register_packed_parameter_state_dict_hooks(module, spec)

    module.qkv_proj = qkv_proj
    # The projections stay in place, emptied. DCP resolves every checkpoint key by
    # attribute traversal, so the canonical q/k/v names still need a module to land on.
    for projection in projections.values():
        projection.weight = None
        projection.bias = None


def apply_model_fusions(
    model: nn.Module, requested_fusions: Sequence[str], raise_on_fail: bool = True
) -> dict[str, int]:
    """Apply each requested fusion to every module that supports it; returns the module count per fusion."""
    fused_module_counts: dict[str, int] = {}
    for fusion_name in requested_fusions:
        num_fused_modules = 0
        for module in model.modules():
            fusion: RuntimeFusion | None = getattr(module, "supported_fusions", {}).get(fusion_name)
            if fusion is None:
                continue
            fusion(module)
            num_fused_modules += 1
        if num_fused_modules == 0:
            message = f"The model does not support the {fusion_name!r} runtime fusion"
            if raise_on_fail:
                raise ValueError(message)
            get_logger().warning(f"{message}; continuing without it")
            continue
        fused_module_counts[fusion_name] = num_fused_modules
    return fused_module_counts


def join_module_path(module_path: str, name: str) -> str:
    """Join a module path and a name, tolerating the empty path of the root module."""
    return f"{module_path}.{name}" if module_path else name


class PackedParameterInfo(NamedTuple):
    fqn: str
    """Fully qualified name of the packed parameter."""

    logical_fqns: tuple[str, ...]
    """Fully qualified canonical names of the parameters it packs, in packing order."""

    parameter: nn.Parameter
    spec: PackedParameterSpec


def get_model_packed_parameters(model: nn.Module) -> Iterator[PackedParameterInfo]:
    """Yield every packed parameter in the model with the canonical names it stands in for."""
    for module_path, module in model.named_modules():
        for spec in getattr(module, "packed_parameter_specs", ()):
            fqn = resolve_fqn(model, join_module_path(module_path, spec.name))
            prefix = fqn.removesuffix(spec.name)
            yield PackedParameterInfo(
                fqn=fqn,
                logical_fqns=tuple(f"{prefix}{logical_name}" for logical_name in spec.logical_names),
                parameter=module.get_parameter(spec.name),
                spec=spec,
            )


def get_fsdp_shard_placement_fn(model: nn.Module) -> Callable[[nn.Parameter], Shard | None]:
    """FSDP placement that keeps every packed parameter's packing dimension unsharded.

    Splitting a DTensor along an unsharded dimension is a local view, so the canonical
    state-dict entries alias the packed storage instead of being replicated copies. FSDP's
    default ``Shard(0)`` is kept for everything else, including parameters packed along
    dim 0 that have no other dimension to shard.
    """
    packing_dim_by_param_id = {
        id(packed_info.parameter): packed_info.spec.dim for packed_info in get_model_packed_parameters(model)
    }

    def shard_placement_fn(parameter: nn.Parameter) -> Shard | None:
        if packing_dim_by_param_id.get(id(parameter)) == 0 and parameter.ndim > 1:
            return Shard(1)
        return None

    return shard_placement_fn


def write_back_loaded_packed_parameters(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Write the logical entries of a state dict loaded in place back into the packed parameters.

    Loading into ``model.state_dict()`` in place only reaches a packed parameter through
    entries that alias it, so the entries that are copies are packed and written back here.
    """
    with torch.no_grad():
        for packed_info in get_model_packed_parameters(model):
            packed_info.parameter.copy_(
                packed_info.spec.pack_logical_tensors([state_dict[fqn] for fqn in packed_info.logical_fqns])
            )


def split_packed_optimizer_state_for_checkpoint(model: nn.Module, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Expose packed optimizer tensors under their logical names for DCP."""
    packed_infos = list(get_model_packed_parameters(model))
    if not packed_infos:
        return state_dict

    checkpoint_state_dict = dict(state_dict)
    checkpoint_state_dict["state"] = dict(state_dict["state"])
    checkpoint_state_dict["param_groups"] = [
        {**group, "params": list(group["params"])} for group in state_dict["param_groups"]
    ]

    for packed_info in packed_infos:
        if packed_info.fqn in checkpoint_state_dict["state"]:
            packed_param_state = checkpoint_state_dict["state"].pop(packed_info.fqn)
            logical_param_states: dict[str, dict[str, Any]] = {fqn: {} for fqn in packed_info.logical_fqns}
            for state_key, value in packed_param_state.items():
                if isinstance(value, torch.Tensor) and value.shape == packed_info.parameter.shape:
                    for fqn, logical_view in zip(
                        packed_info.logical_fqns, packed_info.spec.split_into_logical_views(value)
                    ):
                        logical_param_states[fqn][state_key] = logical_view
                else:
                    for logical_param_state in logical_param_states.values():
                        logical_param_state[state_key] = value
            checkpoint_state_dict["state"].update(logical_param_states)

        for group in checkpoint_state_dict["param_groups"]:
            param_fqns = []
            for param_fqn in group["params"]:
                param_fqns.extend(packed_info.logical_fqns if param_fqn == packed_info.fqn else [param_fqn])
            group["params"] = param_fqns

    return checkpoint_state_dict


def write_back_loaded_packed_optimizer_state(
    model: nn.Module, optimizers: Sequence[torch.optim.Optimizer], checkpoint_state_dict: dict[str, Any]
) -> None:
    """Pack the logical optimizer state DCP loaded in place into the optimizers' live state."""
    packed_infos = {id(packed_info.parameter): packed_info for packed_info in get_model_packed_parameters(model)}
    with torch.no_grad():
        for optimizer in optimizers:
            for parameter, state in optimizer.state.items():
                packed_info = packed_infos.get(id(parameter))
                if packed_info is None:
                    continue
                logical_param_states = [checkpoint_state_dict["state"][fqn] for fqn in packed_info.logical_fqns]
                for state_key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.shape == packed_info.parameter.shape:
                        value.copy_(
                            packed_info.spec.pack_logical_tensors(
                                [logical_param_state[state_key] for logical_param_state in logical_param_states]
                            )
                        )


def join_loaded_optimizer_state_for_runtime(
    model: nn.Module,
    checkpoint_state_dict: dict[str, Any],
    runtime_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Pack the logical optimizer state DCP loaded back into the runtime optimizer state."""
    packed_infos = list(get_model_packed_parameters(model))
    if not packed_infos:
        return checkpoint_state_dict

    restored_state_dict = dict(runtime_state_dict)
    restored_state_dict["state"] = dict(runtime_state_dict["state"])
    restored_state_dict["param_groups"] = [
        {**group, "params": list(group["params"])} for group in checkpoint_state_dict["param_groups"]
    ]

    for packed_info in packed_infos:
        if packed_info.fqn in restored_state_dict["state"]:
            packed_param_state = restored_state_dict["state"][packed_info.fqn]
            logical_param_states = [checkpoint_state_dict["state"][fqn] for fqn in packed_info.logical_fqns]
            first_logical_view = packed_info.spec.split_into_logical_views(packed_info.parameter)[0]
            for state_key, value in logical_param_states[0].items():
                if isinstance(value, torch.Tensor) and value.shape == first_logical_view.shape:
                    with torch.no_grad():
                        packed_param_state[state_key].copy_(
                            packed_info.spec.pack_logical_tensors(
                                [logical_param_state[state_key] for logical_param_state in logical_param_states]
                            )
                        )
                else:
                    packed_param_state[state_key] = value

        for group in restored_state_dict["param_groups"]:
            param_fqns = group["params"]
            if packed_info.logical_fqns[0] not in param_fqns:
                continue
            first_logical_index = min(param_fqns.index(fqn) for fqn in packed_info.logical_fqns)
            group["params"] = [fqn for fqn in param_fqns if fqn not in packed_info.logical_fqns]
            group["params"].insert(first_logical_index, packed_info.fqn)

    return restored_state_dict
