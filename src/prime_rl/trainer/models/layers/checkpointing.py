"""Selective activation checkpointing hooks for custom decoder layers.

New model integration contract:

- `norm`: no explicit hook is required. Every child module whose type name
  contains `norm` is checkpointed through `forward()`.
- `attn_proj`: expose `layer.self_attn.attn_projections(...)` for
  projection-side attention work before the core kernel, plus
  `layer.self_attn.output_proj(...)` for the post-attention output projection.
  This target is meant for attention-local work outside the main attention
  kernel, such as q/k/v projections, attention-local norms, RoPE, gating, and
  similar model-specific helpers.
- `mla_up_proj`: expose `layer.self_attn.mla_up_proj(...)` when MLA-style
  up-projection work should be checkpointed separately from
  `attn_projections(...)`.
- `mlp`: expose a dense `layer.mlp.forward(...)`. A module is treated as dense
  when it does not define `_run_routed_experts` or `tokens_per_expert`.
- `routed_experts`: expose `layer.mlp._run_routed_experts(...)` for the MoE
  expert path.
- `linear_attn`: expose a token-mixer module on `layer.linear_attn` or
  `layer.mamba`, or reuse `layer.self_attn` when
  `layer.attention_type == "sliding_attention"`.

These hook names are an intentional public interface for custom model support,
so new models should follow them directly rather than adding private aliases.
"""

from collections.abc import Iterable
from functools import wraps

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

SELECTIVE_AC_TARGETS = frozenset({"norm", "attn_proj", "mlp", "mla_up_proj", "routed_experts", "linear_attn"})
_PATCHED_METHODS_ATTR = "_prime_rl_selective_ac_patched_methods"


def _is_norm_module(module: nn.Module) -> bool:
    return "norm" in type(module).__name__.lower()


def _should_checkpoint(module: nn.Module) -> bool:
    return torch.is_grad_enabled() and module.training


def checkpoint_method(module: nn.Module, method_name: str) -> None:
    patched_methods = frozenset(getattr(module, _PATCHED_METHODS_ATTR, ()))
    if method_name in patched_methods:
        return

    original = getattr(module, method_name)

    @wraps(original)
    def checkpointed(*args, **kwargs):
        if not _should_checkpoint(module):
            return original(*args, **kwargs)

        def fn(*checkpoint_args):
            return original(*checkpoint_args, **kwargs)

        return checkpoint(fn, *args, use_reentrant=False, preserve_rng_state=False)

    setattr(module, method_name, checkpointed)
    setattr(module, _PATCHED_METHODS_ATTR, patched_methods.union({method_name}))


def _configure_norm_checkpointing(layer: nn.Module) -> None:
    for module in layer.modules():
        if module is layer or not _is_norm_module(module):
            continue
        checkpoint_method(module, "forward")


def _is_dense_mlp(mlp: nn.Module) -> bool:
    return not hasattr(mlp, "_run_routed_experts") and not hasattr(mlp, "tokens_per_expert")


def _supports_attn_proj(self_attn: nn.Module | None) -> bool:
    return self_attn is not None and hasattr(self_attn, "attn_projections") and hasattr(self_attn, "output_proj")


def _get_linear_attn_module(layer: nn.Module) -> nn.Module | None:
    linear_attn = getattr(layer, "linear_attn", None)
    if linear_attn is not None:
        return linear_attn

    mamba = getattr(layer, "mamba", None)
    if mamba is not None:
        return mamba

    if getattr(layer, "attention_type", None) == "sliding_attention":
        return getattr(layer, "self_attn", None)

    return None


def get_supported_targets(layer: nn.Module) -> frozenset[str]:
    """Infer which selective activation checkpoint targets a decoder layer supports."""
    supported_targets = {"norm"}
    self_attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    linear_attn = _get_linear_attn_module(layer)

    if _supports_attn_proj(self_attn):
        supported_targets.add("attn_proj")
    if self_attn is not None and hasattr(self_attn, "mla_up_proj"):
        supported_targets.add("mla_up_proj")
    if mlp is not None and _is_dense_mlp(mlp):
        supported_targets.add("mlp")
    if mlp is not None and hasattr(mlp, "_run_routed_experts"):
        supported_targets.add("routed_experts")
    if linear_attn is not None:
        supported_targets.add("linear_attn")

    return frozenset(supported_targets)


def set_selective_activation_checkpointing(layer: nn.Module, targets: Iterable[str]) -> None:
    target_set = frozenset(targets)
    invalid_targets = target_set - SELECTIVE_AC_TARGETS
    if invalid_targets:
        raise ValueError(f"Unsupported selective activation checkpoint targets: {sorted(invalid_targets)}")

    enabled_targets = target_set & get_supported_targets(layer)
    self_attn = getattr(layer, "self_attn", None)
    mlp = getattr(layer, "mlp", None)
    linear_attn = _get_linear_attn_module(layer)
    attn_proj_is_subsumed = "linear_attn" in enabled_targets and linear_attn is self_attn

    if self_attn is not None and "attn_proj" in enabled_targets and not attn_proj_is_subsumed:
        checkpoint_method(self_attn, "attn_projections")
        checkpoint_method(self_attn, "output_proj")
    if self_attn is not None and "mla_up_proj" in enabled_targets:
        checkpoint_method(self_attn, "mla_up_proj")
    if mlp is not None and "mlp" in enabled_targets:
        checkpoint_method(mlp, "forward")
    if mlp is not None and "routed_experts" in enabled_targets:
        checkpoint_method(mlp, "_run_routed_experts")
    if linear_attn is not None and "linear_attn" in enabled_targets:
        checkpoint_method(linear_attn, "forward")
    if "norm" in enabled_targets:
        _configure_norm_checkpointing(layer)


def supports_selective_activation_checkpointing(layer: nn.Module) -> bool:
    return type(layer).__module__.startswith("prime_rl.trainer.models.")
