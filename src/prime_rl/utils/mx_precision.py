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
    """Build a trainer context that preserves required transfer dtypes.

    Reject clients that cannot represent the model's FP32 overrides.
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
