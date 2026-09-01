from collections.abc import Callable, Iterator, Sequence
from typing import Any, NamedTuple

import torch
from torch import nn

from prime_rl.utils.weights import resolve_fqn

# Replaces some of a module's parameters with a packed one. Advertised by modules
# through their ``supported_fusions``.
RuntimeFusion = Callable[[nn.Module], None]


class PackedParameter:
    """One physical parameter packing several logical parameters along a dimension.

    Training allocates, computes on, and optimizes the packed parameter. The logical
    parameters keep their canonical names in checkpoints, as views of the packed storage.

    ``name`` and ``logical_names`` are relative to the module the packing is registered
    on, and may name parameters of its children. ``dim`` is a non-negative dimension
    index into the packed parameter.
    """

    def __init__(self, name: str, logical_names: Sequence[str], sizes: Sequence[int], dim: int):
        self.name = name
        self.logical_names = tuple(logical_names)
        self.sizes = tuple(sizes)
        self.dim = dim

    def views(self, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """The logical parameters, in ``logical_names`` order, as views of ``tensor``."""
        return tensor.split(self.sizes, dim=self.dim)

    def pack(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.cat(list(tensors), dim=self.dim)

    def matrix_partitions(self, parameter: torch.Tensor) -> tuple[int, ...] | None:
        """Muon's partition sizes, or None when the packing is not on the output dimension."""
        return self.sizes if self.dim == parameter.ndim - 2 else None


def register_packed_parameter(module: nn.Module, packed: PackedParameter) -> None:
    """Record a packed parameter and keep the module's state dict in canonical form.

    The hooks below trade the packed entry for its canonical ones on the way out and
    back again on the way in, so checkpoints never see the packed layout. Torch stamps
    an attribute onto every hook it registers, so these have to be plain functions.
    """

    def split_state_dict(module, state_dict, prefix, local_metadata) -> None:
        views = packed.views(state_dict.pop(f"{prefix}{packed.name}"))
        for logical_name, view in zip(packed.logical_names, views):
            state_dict[f"{prefix}{logical_name}"] = view

    def join_state_dict(module, state_dict, prefix, *load_state_dict_args) -> None:
        state_dict[f"{prefix}{packed.name}"] = packed.pack(
            [state_dict.pop(f"{prefix}{logical_name}") for logical_name in packed.logical_names]
        )

    module.packed_parameters = (*getattr(module, "packed_parameters", ()), packed)
    module.register_state_dict_post_hook(split_state_dict)
    module.register_load_state_dict_pre_hook(join_state_dict)


def fuse_gate_up_projections(module: nn.Module) -> None:
    """Pack a gated expert's gate and up projections into one grouped-GEMM weight.

    The packed weight is ``[num_experts, 2 * hidden_dim, dim]``, so the forward path is a
    single grouped GEMM whose transpose and chunk are views.
    """
    packed = PackedParameter("gate_up_proj", ("gate_proj", "up_proj"), (module.hidden_dim, module.hidden_dim), dim=1)
    module.gate_up_proj = nn.Parameter(
        packed.pack([module.gate_proj, module.up_proj]),
        requires_grad=module.gate_proj.requires_grad,
    )
    module.gate_proj = None
    module.up_proj = None
    register_packed_parameter(module, packed)


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
        packed = PackedParameter(
            f"qkv_proj.{parameter_name}",
            [f"{name}.{parameter_name}" for name in projections],
            sizes,
            dim=0,
        )
        tensors = [getattr(projection, parameter_name) for projection in projections.values()]
        setattr(qkv_proj, parameter_name, nn.Parameter(packed.pack(tensors), requires_grad=tensors[0].requires_grad))
        register_packed_parameter(module, packed)

    module.qkv_proj = qkv_proj
    # The projections stay in place, emptied. DCP resolves every checkpoint key by
    # attribute traversal, so the canonical q/k/v names still need a module to land on.
    for projection in projections.values():
        projection.weight = None
        projection.bias = None


def apply_model_fusions(model: nn.Module, requested: Sequence[str]) -> dict[str, int]:
    applied: dict[str, int] = {}
    for name in requested:
        count = 0
        for module in model.modules():
            fusion: RuntimeFusion | None = getattr(module, "supported_fusions", {}).get(name)
            if fusion is None:
                continue
            fusion(module)
            count += 1
        if count == 0:
            raise ValueError(f"The model does not support the {name!r} runtime fusion")
        applied[name] = count
    return applied


def qualified_name(module_path: str, name: str) -> str:
    """Join a module path and a name, tolerating the empty path of the root module."""
    return f"{module_path}.{name}" if module_path else name


class PackedParameterInfo(NamedTuple):
    name: str
    """Fully qualified name of the packed parameter."""

    logical_names: tuple[str, ...]
    """Fully qualified canonical names of the parameters it packs, in packing order."""

    parameter: nn.Parameter
    packed: PackedParameter


def packed_parameters(model: nn.Module) -> Iterator[PackedParameterInfo]:
    """Yield every packed parameter in the model with the canonical names it stands in for."""
    for module_path, module in model.named_modules():
        for packed in getattr(module, "packed_parameters", ()):
            fqn = resolve_fqn(model, qualified_name(module_path, packed.name))
            prefix = fqn.removesuffix(packed.name)
            yield PackedParameterInfo(
                name=fqn,
                logical_names=tuple(f"{prefix}{logical_name}" for logical_name in packed.logical_names),
                parameter=module.get_parameter(packed.name),
                packed=packed,
            )


def optimizer_state_dict_for_checkpoint(model: nn.Module, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Expose packed optimizer tensors as logical zero-copy views for DCP."""
    packings = list(packed_parameters(model))
    if not packings:
        return state_dict

    checkpoint_state_dict = dict(state_dict)
    checkpoint_state_dict["state"] = dict(state_dict["state"])
    checkpoint_state_dict["param_groups"] = [
        {**group, "params": list(group["params"])} for group in state_dict["param_groups"]
    ]

    for info in packings:
        if info.name in checkpoint_state_dict["state"]:
            physical_state = checkpoint_state_dict["state"].pop(info.name)
            logical_states: dict[str, dict[str, Any]] = {name: {} for name in info.logical_names}
            for state_name, value in physical_state.items():
                if isinstance(value, torch.Tensor) and value.shape == info.parameter.shape:
                    for name, view in zip(info.logical_names, info.packed.views(value)):
                        logical_states[name][state_name] = view
                else:
                    for logical_state in logical_states.values():
                        logical_state[state_name] = value
            checkpoint_state_dict["state"].update(logical_states)

        for group in checkpoint_state_dict["param_groups"]:
            params = []
            for name in group["params"]:
                params.extend(info.logical_names if name == info.name else [name])
            group["params"] = params

    return checkpoint_state_dict


def optimizer_state_dict_for_runtime(
    model: nn.Module,
    checkpoint_state_dict: dict[str, Any],
    runtime_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Restore packed optimizer names after DCP loads through logical views."""
    packings = list(packed_parameters(model))
    if not packings:
        return checkpoint_state_dict

    restored_state_dict = dict(runtime_state_dict)
    restored_state_dict["state"] = dict(runtime_state_dict["state"])
    restored_state_dict["param_groups"] = [
        {**group, "params": list(group["params"])} for group in checkpoint_state_dict["param_groups"]
    ]

    for info in packings:
        if info.name in restored_state_dict["state"]:
            # DCP wrote every tensor through the logical views, which alias the packed
            # storage. Only values it could not write in place are carried over here.
            physical_state = restored_state_dict["state"][info.name]
            first_view = info.packed.views(info.parameter)[0]
            for state_name, value in checkpoint_state_dict["state"][info.logical_names[0]].items():
                if not (isinstance(value, torch.Tensor) and value.shape == first_view.shape):
                    physical_state[state_name] = value

        for group in restored_state_dict["param_groups"]:
            params = group["params"]
            if info.logical_names[0] not in params:
                continue
            insert_at = min(params.index(name) for name in info.logical_names)
            group["params"] = [name for name in params if name not in info.logical_names]
            group["params"].insert(insert_at, info.name)

    return restored_state_dict
