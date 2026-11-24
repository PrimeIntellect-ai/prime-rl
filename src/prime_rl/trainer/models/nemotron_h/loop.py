"""Recurrent layer-looping adapters and initialization for NemotronH."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from prime_rl.trainer.models.nemotron_h.loop_unroll import (
    is_loop_param_name,
    resolve_loop_bounds,
    strip_loop_from_state_dict,
    unroll_looped_state_dict,
)

_BASE_PREFIX = "base_layer."


def freeze_all_except_loop(model: nn.Module) -> None:
    """Freeze all base parameters and leave only ``loop_*`` adapter parameters trainable."""
    for name, param in model.named_parameters():
        param.requires_grad = is_loop_param_name(name)


class LoopLoRALinear(nn.Module):
    """Linear projection with independent low-rank deltas for extra loop passes (``loop_idx >= 1``).

    On pass 0 (``loop_idx == 0``), executes ``base_layer(x)`` untouched. State-dict hooks preserve
    the wrapped ``nn.Linear``'s canonical key names (``*.weight`` / ``*.bias``) so base HF weights
    load without renaming.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        num_extra_loops: int,
        rank: int,
        alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.num_extra_loops = num_extra_loops
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        like = base_layer.weight
        self.loop_lora_A = nn.ParameterList(
            [
                nn.Parameter(torch.empty(rank, base_layer.in_features, device=like.device, dtype=like.dtype))
                for _ in range(num_extra_loops)
            ]
        )
        self.loop_lora_B = nn.ParameterList(
            [
                nn.Parameter(torch.empty(base_layer.out_features, rank, device=like.device, dtype=like.dtype))
                for _ in range(num_extra_loops)
            ]
        )
        if like.device.type != "meta":
            self.reset_loop_parameters()

        self._register_state_dict_hook(self._post_state_dict_hook)
        self.register_load_state_dict_pre_hook(self._pre_load_state_dict_hook)

    def reset_loop_parameters(self) -> None:
        for param in self.loop_lora_A:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in self.loop_lora_B:
            nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
        out = self.base_layer(x)
        if loop_idx > 0:
            idx = loop_idx - 1
            delta = F.linear(F.linear(x, self.loop_lora_A[idx]), self.loop_lora_B[idx])
            out = out + self.scaling * delta
        return out

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_layer, name)

    @staticmethod
    def _post_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> dict[str, Any]:
        old_prefix = f"{prefix}{_BASE_PREFIX}"
        for key in list(state_dict.keys()):
            if not key.startswith(old_prefix):
                continue
            state_dict[prefix + key[len(old_prefix) :]] = state_dict.pop(key)
        return state_dict

    @staticmethod
    def _pre_load_state_dict_hook(
        module: nn.Module,
        state_dict: dict[str, Any],
        prefix: str,
        *args: Any,
    ) -> None:
        old_prefix = prefix
        new_prefix = f"{prefix}{_BASE_PREFIX}"
        for key in list(state_dict.keys()):
            if not key.startswith(old_prefix) or is_loop_param_name(key[len(old_prefix) :]):
                continue
            state_dict[new_prefix + key[len(old_prefix) :]] = state_dict.pop(key)


def call_loop_linear(module: nn.Module, x: torch.Tensor, loop_idx: int = 0) -> torch.Tensor:
    """Call ``module(x, loop_idx)`` if ``module`` is a ``LoopLoRALinear`` on pass ``>= 1``, else ``module(x)``."""
    if loop_idx > 0 and isinstance(module, LoopLoRALinear):
        return module(x, loop_idx)
    return module(x)


def wrap_linear_with_loop_lora(
    parent: nn.Module,
    attr: str,
    num_extra_loops: int,
    rank: int,
    alpha: float,
) -> None:
    """Replace ``getattr(parent, attr)`` with a ``LoopLoRALinear`` when it is an ``nn.Linear``."""
    child = getattr(parent, attr, None)
    if isinstance(child, nn.Linear):
        setattr(parent, attr, LoopLoRALinear(child, num_extra_loops=num_extra_loops, rank=rank, alpha=alpha))


@torch.no_grad()
def init_loop_params_from_base(model: nn.Module) -> None:
    """Initialize loop-pass adapters from loaded pretrained base weights (DTensor/FSDP2-safe)."""
    for module in model.modules():
        if hasattr(module, "loop_norms") and module.loop_norms is not None:
            for loop_norm in module.loop_norms:
                loop_norm.weight.copy_(module.norm.weight)
        if hasattr(module, "loop_gammas") and module.loop_gammas is not None:
            for gamma in module.loop_gammas:
                gamma.zero_()
        if hasattr(module, "loop_routers") and module.loop_routers is not None:
            base_router = module.mlp.router
            for loop_router in module.loop_routers:
                loop_router.gate.weight.copy_(base_router.gate.weight)
                if base_router.selection_bias is not None and loop_router.selection_bias is not None:
                    loop_router.selection_bias.copy_(base_router.selection_bias)
        if isinstance(module, LoopLoRALinear):
            module.reset_loop_parameters()


__all__ = [
    "LoopLoRALinear",
    "call_loop_linear",
    "freeze_all_except_loop",
    "init_loop_params_from_base",
    "is_loop_param_name",
    "resolve_loop_bounds",
    "strip_loop_from_state_dict",
    "unroll_looped_state_dict",
    "wrap_linear_with_loop_lora",
]
