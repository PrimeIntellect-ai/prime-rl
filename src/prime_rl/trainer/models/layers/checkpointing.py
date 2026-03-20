from collections.abc import Callable, Iterable
from functools import partial
from typing import TypeVar

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

DEFAULT_SELECTIVE_AC_TARGETS = frozenset({"norm"})
ATTENTION_SELECTIVE_AC_TARGETS = frozenset({"attention_sdpa"})
MLA_SELECTIVE_AC_TARGETS = frozenset({"mla_up_proj"})
MOE_SELECTIVE_AC_TARGETS = frozenset({"routed_experts"})
SELECTIVE_AC_TARGETS = (
    DEFAULT_SELECTIVE_AC_TARGETS | ATTENTION_SELECTIVE_AC_TARGETS | MLA_SELECTIVE_AC_TARGETS | MOE_SELECTIVE_AC_TARGETS
)
_INTERNAL_SELECTIVE_AC_TARGETS = frozenset(
    {
        "attn_norm",
        "ffn_norm",
        "qk_norm_rope",
        "attention_sdpa",
        "mla_up_proj",
        "routed_experts",
    }
)
_SELECTIVE_AC_ATTR = "_prime_rl_selective_ac_targets"

T = TypeVar("T")


def run_with_optional_checkpoint(enabled: bool, function: Callable[..., T], *args, **kwargs) -> T:
    checkpoint_function = partial(function, **kwargs) if kwargs else function
    if not enabled:
        return checkpoint_function(*args)
    return checkpoint(checkpoint_function, *args, use_reentrant=False, preserve_rng_state=False)


def _supports_attention_selective(layer: nn.Module) -> bool:
    return hasattr(getattr(layer, "self_attn", None), "forward_selective")


def _supports_routed_experts(layer: nn.Module) -> bool:
    return hasattr(getattr(layer, "mlp", None), "_run_routed_experts")


def _supports_qk_norm_rope(layer: nn.Module) -> bool:
    self_attn = getattr(layer, "self_attn", None)
    return _supports_attention_selective(layer) and getattr(self_attn, "use_qk_norm", False)


def _supports_mla_up_proj(layer: nn.Module) -> bool:
    self_attn = getattr(layer, "self_attn", None)
    return getattr(self_attn, "supports_mla_up_proj_activation_checkpointing", False)


def _expand_selective_targets(layer: nn.Module, targets: frozenset[str]) -> frozenset[str]:
    expanded_targets: set[str] = set()
    if "norm" in targets:
        expanded_targets.update({"attn_norm", "ffn_norm"})
        if _supports_qk_norm_rope(layer):
            expanded_targets.add("qk_norm_rope")
    if "attention_sdpa" in targets and _supports_attention_selective(layer):
        expanded_targets.add("attention_sdpa")
    if "mla_up_proj" in targets and _supports_mla_up_proj(layer):
        expanded_targets.add("mla_up_proj")
    if "routed_experts" in targets and _supports_routed_experts(layer):
        expanded_targets.add("routed_experts")

    invalid_internal_targets = expanded_targets - _INTERNAL_SELECTIVE_AC_TARGETS
    if invalid_internal_targets:
        raise ValueError(
            f"Unsupported internal selective activation checkpoint targets: {sorted(invalid_internal_targets)}"
        )
    return frozenset(expanded_targets)


def set_selective_activation_checkpointing(layer: nn.Module, targets: Iterable[str]) -> None:
    normalized_targets = frozenset(targets)
    invalid_targets = normalized_targets - SELECTIVE_AC_TARGETS
    if invalid_targets:
        raise ValueError(f"Unsupported selective activation checkpoint targets: {sorted(invalid_targets)}")
    setattr(layer, _SELECTIVE_AC_ATTR, _expand_selective_targets(layer, normalized_targets))


def get_requested_targets(layer: nn.Module) -> frozenset[str]:
    return getattr(layer, _SELECTIVE_AC_ATTR, frozenset())


def get_supported_targets(layer: nn.Module) -> frozenset[str]:
    supported_targets = getattr(layer, "supported_selective_activation_checkpoint_targets", None)
    if supported_targets is not None:
        return supported_targets

    inferred_targets = set(DEFAULT_SELECTIVE_AC_TARGETS)
    if _supports_attention_selective(layer):
        inferred_targets.update(ATTENTION_SELECTIVE_AC_TARGETS)
    if _supports_mla_up_proj(layer):
        inferred_targets.update(MLA_SELECTIVE_AC_TARGETS)
    if _supports_routed_experts(layer):
        inferred_targets.update(MOE_SELECTIVE_AC_TARGETS)
    return frozenset(inferred_targets)


def should_checkpoint(layer: nn.Module, target: str) -> bool:
    return layer.training and target in get_requested_targets(layer)
