from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol

import torch
from torch import nn

from prime_rl.utils.weights import resolve_fqn


class RuntimeFusion(Protocol):
    """A parameter-layout transformation advertised by a model module."""

    @classmethod
    def apply(cls, module: nn.Module) -> None: ...

    @staticmethod
    def logical_parameter_views(
        module: nn.Module,
        tensor: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]: ...


class GroupedExpertsGateUpFusion:
    @classmethod
    def apply(cls, module: nn.Module) -> None:
        gate_up_proj = nn.Parameter(
            torch.stack((module.gate_proj, module.up_proj), dim=2),
            requires_grad=module.gate_proj.requires_grad,
        )
        module.gate_proj = None
        module.up_proj = None
        module.gate_up_proj = gate_up_proj
        module.applied_fusions = {"gate_up_proj": cls}
        module.register_state_dict_post_hook(cls.export_state_dict)
        module.register_load_state_dict_pre_hook(cls.import_state_dict)

    @staticmethod
    def logical_parameter_views(
        module: nn.Module,
        tensor: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        gate_proj, up_proj = tensor.unbind(2)
        return {"gate_proj": gate_proj, "up_proj": up_proj}

    @staticmethod
    def export_state_dict(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
    ) -> None:
        gate_up_proj = state_dict.pop(f"{prefix}gate_up_proj")
        state_dict[f"{prefix}gate_proj"], state_dict[f"{prefix}up_proj"] = gate_up_proj.unbind(2)

    @staticmethod
    def import_state_dict(
        module: nn.Module,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        state_dict[f"{prefix}gate_up_proj"] = torch.stack(
            (state_dict.pop(f"{prefix}gate_proj"), state_dict.pop(f"{prefix}up_proj")),
            dim=2,
        )


def apply_model_fusions(model: nn.Module, requested: Sequence[str]) -> dict[str, int]:
    applied: dict[str, int] = {}
    for name in requested:
        count = 0
        for module in model.modules():
            supported: Mapping[str, type[RuntimeFusion]] = getattr(module, "supported_fusions", {})
            fusion = supported.get(name)
            if fusion is None:
                continue
            fusion.apply(module)
            count += 1
        if count == 0:
            raise ValueError(f"The model does not support the {name!r} runtime fusion")
        applied[name] = count
    return applied


def applied_parameter_fusions(
    model: nn.Module,
) -> Iterator[tuple[str, nn.Parameter, nn.Module, type[RuntimeFusion]]]:
    for module in model.modules():
        for parameter_name, fusion in getattr(module, "applied_fusions", {}).items():
            yield parameter_name, getattr(module, parameter_name), module, fusion


def optimizer_state_dict_for_checkpoint(model: nn.Module, state_dict: dict[str, Any]) -> dict[str, Any]:
    """Expose fused optimizer tensors as logical zero-copy views for DCP."""
    parameter_fusions = list(applied_parameter_fusions(model))
    if not parameter_fusions:
        return state_dict

    checkpoint_state_dict = dict(state_dict)
    checkpoint_state_dict["state"] = dict(state_dict["state"])
    checkpoint_state_dict["param_groups"] = [
        {**group, "params": list(group["params"])} for group in state_dict["param_groups"]
    ]

    fused_parameter_ids = {id(parameter) for _, parameter, _, _ in parameter_fusions}
    parameter_names = {
        id(parameter): resolve_fqn(model, name)
        for name, parameter in model.named_parameters()
        if id(parameter) in fused_parameter_ids
    }
    for parameter_name, parameter, module, fusion in parameter_fusions:
        physical_name = parameter_names[id(parameter)]
        prefix = physical_name.removesuffix(parameter_name)
        logical_parameter_views = fusion.logical_parameter_views(module, parameter)
        logical_names = {name: f"{prefix}{name}" for name in logical_parameter_views}

        if physical_name in checkpoint_state_dict["state"]:
            physical_state = checkpoint_state_dict["state"].pop(physical_name)
            logical_states = {name: {} for name in logical_names}
            for state_name, value in physical_state.items():
                if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                    for name, view in fusion.logical_parameter_views(module, value).items():
                        logical_states[name][state_name] = view
                else:
                    for logical_state in logical_states.values():
                        logical_state[state_name] = value
            for name, logical_state in logical_states.items():
                checkpoint_state_dict["state"][logical_names[name]] = logical_state

        for group in checkpoint_state_dict["param_groups"]:
            params = []
            for name in group["params"]:
                if name == physical_name:
                    params.extend(logical_names.values())
                else:
                    params.append(name)
            group["params"] = params

    return checkpoint_state_dict


def optimizer_state_dict_for_runtime(
    model: nn.Module,
    checkpoint_state_dict: dict[str, Any],
    runtime_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Restore physical optimizer names after DCP loads through logical views."""
    parameter_fusions = list(applied_parameter_fusions(model))
    if not parameter_fusions:
        return checkpoint_state_dict

    restored_state_dict = dict(runtime_state_dict)
    restored_state_dict["state"] = dict(runtime_state_dict["state"])
    restored_state_dict["param_groups"] = [
        {**group, "params": list(group["params"])} for group in checkpoint_state_dict["param_groups"]
    ]

    fused_parameter_ids = {id(parameter) for _, parameter, _, _ in parameter_fusions}
    parameter_names = {
        id(parameter): resolve_fqn(model, name)
        for name, parameter in model.named_parameters()
        if id(parameter) in fused_parameter_ids
    }
    for parameter_name, parameter, module, fusion in parameter_fusions:
        physical_name = parameter_names[id(parameter)]
        prefix = physical_name.removesuffix(parameter_name)
        logical_parameter_views = fusion.logical_parameter_views(module, parameter)
        logical_names = {name: f"{prefix}{name}" for name in logical_parameter_views}

        if physical_name in restored_state_dict["state"]:
            physical_state = restored_state_dict["state"][physical_name]
            first_logical_name = next(iter(logical_names))
            logical_state = checkpoint_state_dict["state"][logical_names[first_logical_name]]
            for state_name, value in logical_state.items():
                if not (
                    isinstance(value, torch.Tensor) and value.shape == logical_parameter_views[first_logical_name].shape
                ):
                    physical_state[state_name] = value

        for group in restored_state_dict["param_groups"]:
            params = group["params"]
            if next(iter(logical_names.values())) not in params:
                continue
            indices = [params.index(name) for name in logical_names.values()]
            insert_at = min(indices)
            group["params"] = [name for name in params if name not in logical_names.values()]
            group["params"].insert(insert_at, physical_name)

    return restored_state_dict
