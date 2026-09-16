"""Translate a model's precision requirements into ModelExpress tensor overrides."""

import inspect

import torch
from torch import nn


def wire_dtype_overrides(model: nn.Module, tensors: dict[str, torch.Tensor]) -> dict[str, torch.dtype]:
    keep_in_fp32 = getattr(model, "keep_in_fp32_for_weight_transfer", None)
    if keep_in_fp32 is None:
        return {}
    return {name: torch.float32 for name in tensors if keep_in_fp32(name)}


def build_trainer_context(context_cls, model: nn.Module, tensors: dict[str, torch.Tensor]):
    """Construct `context_cls`, naming wire-dtype overrides only if it accepts them.

    `FSDPTrainerContext` grew `wire_dtype_overrides` after the released client,
    so naming it unconditionally makes the refit transport require an unreleased
    ModelExpress from every model -- including the ones that ask for no
    overrides at all, which is most of them.

    A model that does ask for them is a different matter. Transferring a tensor
    at BF16 when the model declared it needs FP32 loses bits silently and
    surfaces much later as a diverged policy, so an override the client cannot
    carry has to stop the run rather than warn.

    Takes the class as an argument rather than importing it so this stays
    testable, and usable, without the ModelExpress client installed.
    """
    overrides = wire_dtype_overrides(model, tensors)
    if "wire_dtype_overrides" in inspect.signature(context_cls).parameters:
        return context_cls(wire_dtype_overrides=overrides)
    if overrides:
        raise RuntimeError(
            f"{len(overrides)} tensors require FP32 on the wire, but the installed "
            f"modelexpress_rl {context_cls.__name__} does not accept wire_dtype_overrides. "
            "Upgrade ModelExpress to a revision that carries it."
        )
    return context_cls()
