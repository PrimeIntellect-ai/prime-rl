from typing import Iterable

import torch
import torch.nn as nn

from prime_rl.trainer.models.layers.attn import FlashAttention


KV_PREFIX_KEY_NAME = "kv_prefix_key"
KV_PREFIX_VALUE_NAME = "kv_prefix_value"


def is_kv_prefix_param_name(name: str) -> bool:
    return KV_PREFIX_KEY_NAME in name or KV_PREFIX_VALUE_NAME in name


def strip_kv_prefix_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if not is_kv_prefix_param_name(key)}


def iter_kv_prefix_named_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, param in model.named_parameters():
        if is_kv_prefix_param_name(name):
            yield name, param


def freeze_all_except_kv_prefix(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = is_kv_prefix_param_name(name)


def apply_kv_prefix_to_model(
    model: nn.Module,
    num_tokens: int,
    init: str = "normal",
    init_std: float = 0.02,
) -> list[str]:
    module_names: list[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, FlashAttention):
            continue
        module.enable_kv_prefix(num_tokens=num_tokens, init=init, init_std=init_std)
        module_names.append(module_name)

    if not module_names:
        raise ValueError("No compatible flash-attention modules found for KV-prefix tuning.")

    return module_names
