import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(key.split(".")[2]) for key in state_dict if key.startswith("model.layers.")) + 1


def _pop_first(state_dict: dict[str, Tensor], keys: tuple[str, ...]) -> Tensor | None:
    for key in keys:
        if key in state_dict:
            return state_dict.pop(key)
    return None


def convert_hf_layer_to_prime(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"
    gate_key = f"{prefix}.mlp.gate.weight"
    if gate_key not in state_dict:
        return

    state_dict[f"{prefix}.mlp.router.gate.weight"] = state_dict.pop(gate_key)

    expert_bias = _pop_first(
        state_dict,
        (
            f"{prefix}.mlp.experts.e_score_correction_bias",
            f"{prefix}.mlp.gate.e_score_correction_bias",
        ),
    )
    if expert_bias is not None:
        state_dict[f"{prefix}.mlp.expert_bias"] = expert_bias

    fused_key = f"{prefix}.mlp.experts.gate_up_proj"
    if fused_key in state_dict:
        gate_up_proj = state_dict.pop(fused_key)
        down_proj = state_dict.pop(f"{prefix}.mlp.experts.down_proj")
        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj
    else:
        expert_ids = sorted(
            int(key[len(f"{prefix}.mlp.experts.") :].split(".")[0])
            for key in state_dict
            if key.startswith(f"{prefix}.mlp.experts.") and key.endswith(".gate_proj.weight")
        )
        if not expert_ids:
            return

        first_down = state_dict[f"{prefix}.mlp.experts.{expert_ids[0]}.down_proj.weight"]
        dim, moe_dim = first_down.shape
        w1 = torch.empty((len(expert_ids), moe_dim, dim), dtype=first_down.dtype, device=first_down.device)
        w2 = torch.empty((len(expert_ids), dim, moe_dim), dtype=first_down.dtype, device=first_down.device)
        w3 = torch.empty((len(expert_ids), moe_dim, dim), dtype=first_down.dtype, device=first_down.device)
        for out_idx, expert_idx in enumerate(expert_ids):
            w1[out_idx].copy_(state_dict.pop(f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight"))
            w2[out_idx].copy_(state_dict.pop(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight"))
            w3[out_idx].copy_(state_dict.pop(f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight"))

    state_dict[f"{prefix}.mlp.experts.w1"] = w1
    state_dict[f"{prefix}.mlp.experts.w2"] = w2
    state_dict[f"{prefix}.mlp.experts.w3"] = w3

    shared_prefix = None
    for candidate in (f"{prefix}.mlp.shared_expert", f"{prefix}.mlp.shared_experts"):
        if f"{candidate}.gate_proj.weight" in state_dict:
            shared_prefix = candidate
            break

    if shared_prefix is not None:
        state_dict[f"{prefix}.shared_expert.w1.weight"] = state_dict.pop(f"{shared_prefix}.gate_proj.weight")
        state_dict[f"{prefix}.shared_expert.w2.weight"] = state_dict.pop(f"{shared_prefix}.down_proj.weight")
        state_dict[f"{prefix}.shared_expert.w3.weight"] = state_dict.pop(f"{shared_prefix}.up_proj.weight")


def convert_prime_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    prefix = f"model.layers.{layer_idx}"
    router_key = f"{prefix}.mlp.router.gate.weight"
    if router_key not in state_dict:
        return

    if f"{prefix}.mlp.tokens_per_expert" in state_dict:
        del state_dict[f"{prefix}.mlp.tokens_per_expert"]
    if f"{prefix}.mlp.expert_bias" in state_dict:
        state_dict[f"{prefix}.mlp.experts.e_score_correction_bias"] = state_dict.pop(f"{prefix}.mlp.expert_bias")

    state_dict[f"{prefix}.mlp.gate.weight"] = state_dict.pop(router_key)

    w1 = state_dict.pop(f"{prefix}.mlp.experts.w1")
    w2 = state_dict.pop(f"{prefix}.mlp.experts.w2")
    w3 = state_dict.pop(f"{prefix}.mlp.experts.w3")
    for expert_idx in range(w1.shape[0]):
        state_dict[f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight"] = w1[expert_idx]
        state_dict[f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight"] = w2[expert_idx]
        state_dict[f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight"] = w3[expert_idx]

    shared_key = f"{prefix}.shared_expert.w1.weight"
    if shared_key in state_dict:
        state_dict[f"{prefix}.mlp.shared_expert.gate_proj.weight"] = state_dict.pop(shared_key)
        state_dict[f"{prefix}.mlp.shared_expert.down_proj.weight"] = state_dict.pop(f"{prefix}.shared_expert.w2.weight")
        state_dict[f"{prefix}.mlp.shared_expert.up_proj.weight"] = state_dict.pop(f"{prefix}.shared_expert.w3.weight")


def convert_hf_to_prime(state_dict: dict[str, Tensor]) -> None:
    for layer_idx in range(get_max_layer_num(state_dict)):
        convert_hf_layer_to_prime(state_dict, layer_idx)


def convert_prime_to_hf(state_dict: dict[str, Tensor]) -> None:
    for layer_idx in range(get_max_layer_num(state_dict)):
        convert_prime_layer_to_hf(state_dict, layer_idx)


__all__ = [
    "convert_hf_layer_to_prime",
    "convert_hf_to_prime",
    "convert_prime_layer_to_hf",
    "convert_prime_to_hf",
]
