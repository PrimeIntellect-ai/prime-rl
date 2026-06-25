import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single layer from HF format to TT format in-place.

    Handles both source formats:
    - hub checkpoint (tencent/Hy3-preview): per-expert `mlp.experts.{j}.{gate,up,down}_proj.weight`,
      `mlp.router.gate.weight`, `mlp.expert_bias`, `mlp.shared_mlp.*`
    - transformers in-memory format: fused `mlp.experts.{gate_up,down}_proj`, `mlp.gate.weight`,
      `mlp.e_score_correction_bias`, `mlp.shared_experts.*`
    """
    i = layer_idx
    prefix = f"model.layers.{i}."

    # MTP layers (speculative decoding only, marked by eh_proj) are not trained: drop entirely
    if f"{prefix}eh_proj.weight" in state_dict:
        for key in [k for k in state_dict if k.startswith(prefix)]:
            del state_dict[key]
        return

    # Router: transformers names it mlp.gate, the hub checkpoint already uses mlp.router.gate
    if f"{prefix}mlp.gate.weight" in state_dict:
        state_dict[f"{prefix}mlp.router.gate.weight"] = state_dict.pop(f"{prefix}mlp.gate.weight")
    if f"{prefix}mlp.router.gate.weight" not in state_dict:
        return  # dense layer

    # Correction bias: transformers names it e_score_correction_bias, the hub uses expert_bias
    if f"{prefix}mlp.e_score_correction_bias" in state_dict:
        state_dict[f"{prefix}mlp.expert_bias"] = state_dict.pop(f"{prefix}mlp.e_score_correction_bias")

    # Routed experts
    if f"{prefix}mlp.experts.gate_up_proj" in state_dict:
        # Fused format: gate_up_proj has shape (num_experts, 2*moe_dim, dim)
        gate_up_proj = state_dict.pop(f"{prefix}mlp.experts.gate_up_proj")
        down_proj = state_dict.pop(f"{prefix}mlp.experts.down_proj")

        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]  # Gate: (num_experts, moe_dim, dim)
        w3 = gate_up_proj[:, moe_dim:, :]  # Up: (num_experts, moe_dim, dim)
        w2 = down_proj  # Down: (num_experts, dim, moe_dim)
    else:
        # Per-expert format
        num_experts = len([j for j in state_dict.keys() if f"{prefix}mlp.experts." in j]) // 3
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"{prefix}mlp.experts.0.down_proj.weight"].shape
        dtype = state_dict[f"{prefix}mlp.experts.0.down_proj.weight"].dtype
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)  # Gate
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)  # Down
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)  # Up
        for j in range(num_experts):
            w1[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.gate_proj.weight"))
            w2[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.down_proj.weight"))
            w3[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.up_proj.weight"))

    state_dict[f"{prefix}mlp.experts.w1"] = w1
    state_dict[f"{prefix}mlp.experts.w2"] = w2
    state_dict[f"{prefix}mlp.experts.w3"] = w3

    # Shared experts: the hub uses shared_mlp, transformers uses shared_experts
    for hf_name in ("shared_mlp", "shared_experts"):
        if f"{prefix}mlp.{hf_name}.gate_proj.weight" in state_dict:
            state_dict[f"{prefix}mlp.shared_expert.w1"] = state_dict.pop(f"{prefix}mlp.{hf_name}.gate_proj.weight")
            state_dict[f"{prefix}mlp.shared_expert.w2"] = state_dict.pop(f"{prefix}mlp.{hf_name}.down_proj.weight")
            state_dict[f"{prefix}mlp.shared_expert.w3"] = state_dict.pop(f"{prefix}mlp.{hf_name}.up_proj.weight")


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from HF to TT format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert a layer from TT to hub format in-place.

    Emits the hub checkpoint format (per-expert experts, `mlp.router.gate.weight`,
    `mlp.expert_bias`, `mlp.shared_mlp.*`), which both vLLM and transformers load natively.
    """
    i = layer_index
    prefix = f"model.layers.{i}."

    # Load balancing stats are training-only state
    if f"{prefix}mlp.tokens_per_expert" in state_dict:
        del state_dict[f"{prefix}mlp.tokens_per_expert"]

    # Shared experts
    if f"{prefix}mlp.shared_expert.w1" in state_dict:
        state_dict[f"{prefix}mlp.shared_mlp.gate_proj.weight"] = state_dict.pop(f"{prefix}mlp.shared_expert.w1")
        state_dict[f"{prefix}mlp.shared_mlp.down_proj.weight"] = state_dict.pop(f"{prefix}mlp.shared_expert.w2")
        state_dict[f"{prefix}mlp.shared_mlp.up_proj.weight"] = state_dict.pop(f"{prefix}mlp.shared_expert.w3")

    # Router (mlp.router.gate.weight) and correction bias (mlp.expert_bias) already use hub names

    # Routed experts - convert to per-expert format
    if f"{prefix}mlp.experts.w1" in state_dict:
        w1 = state_dict.pop(f"{prefix}mlp.experts.w1")  # (num_experts, moe_dim, dim)
        w2 = state_dict.pop(f"{prefix}mlp.experts.w2")  # (num_experts, dim, moe_dim)
        w3 = state_dict.pop(f"{prefix}mlp.experts.w3")  # (num_experts, moe_dim, dim)

        num_experts = w1.shape[0]
        for j in range(num_experts):
            state_dict[f"{prefix}mlp.experts.{j}.gate_proj.weight"] = w1[j]
            state_dict[f"{prefix}mlp.experts.{j}.down_proj.weight"] = w2[j]
            state_dict[f"{prefix}mlp.experts.{j}.up_proj.weight"] = w3[j]


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from TT to hub format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
