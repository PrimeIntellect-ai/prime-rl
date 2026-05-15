import torch
from torch import Tensor

_HF_GATE_PREFIX = ".mlp.gate."
_PRIME_ROUTER_PREFIX = ".mlp.router."


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(key.split(".")[2]) for key in state_dict if key.startswith("model.layers.")) + 1


def is_hf_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(_HF_GATE_PREFIX in name for name in state_dict)


def is_prime_state_dict(state_dict: dict[str, Tensor]) -> bool:
    return any(_PRIME_ROUTER_PREFIX in name or name.endswith(".mlp.experts.w1") for name in state_dict)


def _rename_layer_prefix(state_dict: dict[str, Tensor], old_prefix: str, new_prefix: str) -> None:
    for key in [key for key in state_dict if key.startswith(old_prefix)]:
        state_dict[new_prefix + key[len(old_prefix) :]] = state_dict.pop(key)


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"
    _rename_layer_prefix(state_dict, f"{prefix}.mlp.gate.", f"{prefix}.mlp.router.")

    gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
    down_key = f"{prefix}.mlp.experts.down_proj"
    if gate_up_key in state_dict:
        state_dict[gate_up_key] = state_dict[gate_up_key].contiguous()
    if down_key in state_dict:
        state_dict[down_key] = state_dict[down_key].contiguous()


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"

    state_dict.pop(f"{prefix}.mlp.tokens_per_expert", None)
    _rename_layer_prefix(state_dict, f"{prefix}.mlp.router.", f"{prefix}.mlp.gate.")

    gate_up_key = f"{prefix}.mlp.experts.gate_up_proj"
    down_key = f"{prefix}.mlp.experts.down_proj"
    if gate_up_key in state_dict:
        state_dict[gate_up_key] = state_dict[gate_up_key].contiguous()
    if down_key in state_dict:
        state_dict[down_key] = state_dict[down_key].contiguous()

    w1_key = f"{prefix}.mlp.experts.w1"
    w2_key = f"{prefix}.mlp.experts.w2"
    w3_key = f"{prefix}.mlp.experts.w3"
    if w1_key in state_dict:
        w1 = state_dict.pop(w1_key)
        w2 = state_dict.pop(w2_key)
        w3 = state_dict.pop(w3_key)
        state_dict[gate_up_key] = torch.cat([w1, w3], dim=1).contiguous()
        state_dict[down_key] = w2.contiguous()


def convert_hf_to_prime(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer_idx in range(get_max_layer_num(state_dict)):
        convert_hf_layer_to_prime(state_dict, layer_idx)
    return state_dict


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    for layer_idx in range(get_max_layer_num(state_dict)):
        convert_prime_layer_to_hf(state_dict, layer_idx)
    return state_dict


__all__ = [
    "convert_hf_layer_to_prime",
    "convert_hf_to_prime",
    "convert_prime_layer_to_hf",
    "convert_prime_to_hf",
    "get_max_layer_num",
    "is_hf_state_dict",
    "is_prime_state_dict",
]
