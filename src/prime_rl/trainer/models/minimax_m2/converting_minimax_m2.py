import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a layer from HF to PrimeRL format in-place.

    HF MiniMax M2.1 uses `block_sparse_moe` as the MoE block name, with
    expert weights stored as `experts.{j}.w1.weight` (nn.Linear format).
    """
    i = layer_idx
    prefix = f"model.layers.{i}"

    # Check if this layer has MoE experts
    num_experts = len([k for k in state_dict.keys() if f"{prefix}.block_sparse_moe.experts" in k]) // 3
    if num_experts == 0:
        return

    # Router: block_sparse_moe.gate.weight -> mlp.router.gate.weight
    state_dict[f"{prefix}.mlp.router.gate.weight"] = state_dict[f"{prefix}.block_sparse_moe.gate.weight"]
    del state_dict[f"{prefix}.block_sparse_moe.gate.weight"]

    # e_score_correction_bias: direct tensor under block_sparse_moe -> mlp.expert_bias
    bias_key = f"{prefix}.block_sparse_moe.e_score_correction_bias"
    if bias_key in state_dict:
        state_dict[f"{prefix}.mlp.expert_bias"] = state_dict[bias_key]
        del state_dict[bias_key]

    # Expert weights: stack individual experts into grouped tensors
    # HF uses w1.weight, w2.weight, w3.weight (nn.Linear with .weight suffix)
    dim, moe_dim = state_dict[f"{prefix}.block_sparse_moe.experts.0.w2.weight"].shape
    dtype = state_dict[f"{prefix}.block_sparse_moe.experts.0.w2.weight"].dtype

    w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
    w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)
    w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)

    for j in range(num_experts):
        w1[j].copy_(state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w1.weight"])
        w2[j].copy_(state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w2.weight"])
        w3[j].copy_(state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w3.weight"])

        del state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w1.weight"]
        del state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w2.weight"]
        del state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w3.weight"]

    state_dict[f"{prefix}.mlp.experts.w1"] = w1
    state_dict[f"{prefix}.mlp.experts.w2"] = w2
    state_dict[f"{prefix}.mlp.experts.w3"] = w3


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a layer from PrimeRL to HF format in-place."""
    i = layer_idx
    prefix = f"model.layers.{i}"

    if f"{prefix}.mlp.router.gate.weight" not in state_dict:
        return

    # Router: mlp.router.gate.weight -> block_sparse_moe.gate.weight
    state_dict[f"{prefix}.block_sparse_moe.gate.weight"] = state_dict[f"{prefix}.mlp.router.gate.weight"]
    del state_dict[f"{prefix}.mlp.router.gate.weight"]

    # expert_bias -> block_sparse_moe.e_score_correction_bias
    if f"{prefix}.mlp.expert_bias" in state_dict:
        state_dict[f"{prefix}.block_sparse_moe.e_score_correction_bias"] = state_dict[f"{prefix}.mlp.expert_bias"]
        del state_dict[f"{prefix}.mlp.expert_bias"]

    # tokens_per_expert is a runtime buffer, remove if present
    if f"{prefix}.mlp.tokens_per_expert" in state_dict:
        del state_dict[f"{prefix}.mlp.tokens_per_expert"]

    # Expert weights: unstack grouped tensors into individual experts
    num_experts, moe_dim, dim = state_dict[f"{prefix}.mlp.experts.w1"].shape
    for j in range(num_experts):
        state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w1.weight"] = state_dict[f"{prefix}.mlp.experts.w1"][j]
        state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w2.weight"] = state_dict[f"{prefix}.mlp.experts.w2"][j]
        state_dict[f"{prefix}.block_sparse_moe.experts.{j}.w3.weight"] = state_dict[f"{prefix}.mlp.experts.w3"][j]

    del state_dict[f"{prefix}.mlp.experts.w1"]
    del state_dict[f"{prefix}.mlp.experts.w2"]
    del state_dict[f"{prefix}.mlp.experts.w3"]


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from HF to PrimeRL format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from PrimeRL to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
