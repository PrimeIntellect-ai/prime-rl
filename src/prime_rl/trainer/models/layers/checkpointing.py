from collections.abc import Callable, Iterable
from functools import partial
import inspect
from typing import TypeVar

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

SELECTIVE_AC_TARGETS = frozenset({"norm", "attention_sdpa", "mla_up_proj", "routed_experts"})
_SELECTIVE_AC_ATTR = "_prime_rl_selective_ac_targets"
_NORM_PATCH_ATTR = "_prime_rl_norm_forward_patched"
_SELF_ATTN_SELECTIVE_AC_TARGETS = frozenset({"attention_sdpa", "mla_up_proj"})
_MLP_SELECTIVE_AC_TARGETS = frozenset({"routed_experts"})

T = TypeVar("T")


def run_with_optional_checkpoint(enabled: bool, function: Callable[..., T], *args, **kwargs) -> T:
    checkpoint_function = partial(function, **kwargs) if kwargs else function
    if not enabled:
        return checkpoint_function(*args)
    return checkpoint(checkpoint_function, *args, use_reentrant=False, preserve_rng_state=False)


def _is_norm_module(module: nn.Module) -> bool:
    return "norm" in type(module).__name__.lower()


def _patch_norm_forward(module: nn.Module) -> None:
    if getattr(module, _NORM_PATCH_ATTR, False):
        return

    original_forward = module.forward

    def checkpointed_forward(*args, **kwargs):
        return run_with_optional_checkpoint(should_checkpoint(module, "norm"), original_forward, *args, **kwargs)

    module.forward = checkpointed_forward  # type: ignore[method-assign]
    setattr(module, _NORM_PATCH_ATTR, True)


def _configure_norm_checkpointing(layer: nn.Module, enable_norm: bool) -> None:
    norm_targets = frozenset({"norm"}) if enable_norm else frozenset()
    for module in layer.modules():
        if module is layer or not _is_norm_module(module):
            continue
        setattr(module, _SELECTIVE_AC_ATTR, norm_targets)
        if enable_norm:
            _patch_norm_forward(module)


def get_supported_targets(layer: nn.Module) -> frozenset[str]:
    supported_targets = {"norm"}
    self_attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    self_attn_forward_params = set(inspect.signature(self_attn.forward).parameters) if self_attn is not None else set()

    if "checkpoint_attention_sdpa" in self_attn_forward_params:
        supported_targets.add("attention_sdpa")
    if "checkpoint_mla_up_proj" in self_attn_forward_params:
        supported_targets.add("mla_up_proj")
    if hasattr(mlp, "_run_routed_experts"):
        supported_targets.add("routed_experts")

    return frozenset(supported_targets)


def set_selective_activation_checkpointing(layer: nn.Module, targets: Iterable[str]) -> None:
    normalized_targets = frozenset(targets)
    invalid_targets = normalized_targets - SELECTIVE_AC_TARGETS
    if invalid_targets:
        raise ValueError(f"Unsupported selective activation checkpoint targets: {sorted(invalid_targets)}")
    enabled_targets = normalized_targets & get_supported_targets(layer)

    setattr(layer, _SELECTIVE_AC_ATTR, enabled_targets)

    self_attn = getattr(layer, "self_attn", None)
    if self_attn is not None:
        setattr(self_attn, _SELECTIVE_AC_ATTR, enabled_targets & _SELF_ATTN_SELECTIVE_AC_TARGETS)

    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        setattr(mlp, _SELECTIVE_AC_ATTR, enabled_targets & _MLP_SELECTIVE_AC_TARGETS)

    _configure_norm_checkpointing(layer, "norm" in enabled_targets)


def supports_selective_activation_checkpointing(layer: nn.Module) -> bool:
    return type(layer).__module__.startswith("prime_rl.trainer.models.")


def get_requested_targets(layer: nn.Module) -> frozenset[str]:
    return getattr(layer, _SELECTIVE_AC_ATTR, frozenset())


def should_checkpoint(layer: nn.Module, target: str) -> bool:
    return torch.is_grad_enabled() and layer.training and target in get_requested_targets(layer)
