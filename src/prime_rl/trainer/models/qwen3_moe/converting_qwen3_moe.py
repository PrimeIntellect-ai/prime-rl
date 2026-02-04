import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a layer from HF to TT format in-place."""
    i = layer_idx

    # Check if this is a MoE layer by looking for gate weight
    if f"model.layers.{i}.mlp.gate.weight" not in state_dict:
        return

    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.gate.weight"]

    # Check if this is the new fused format (transformers 5.0+) or old per-expert format
    if f"model.layers.{i}.mlp.experts.gate_up_proj" in state_dict:
        # New fused format: gate_up_proj has shape (num_experts, 2*moe_dim, dim)
        gate_up_proj = state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        down_proj = state_dict[f"model.layers.{i}.mlp.experts.down_proj"]

        num_experts, fused_dim, dim = gate_up_proj.shape
        moe_dim = fused_dim // 2

        # Split gate_up_proj into w1 (gate) and w3 (up)
        w1 = gate_up_proj[:, :moe_dim, :]  # Gate: (num_experts, moe_dim, dim)
        w3 = gate_up_proj[:, moe_dim:, :]  # Up: (num_experts, moe_dim, dim)
        w2 = down_proj  # Down: (num_experts, dim, moe_dim)

        del state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        del state_dict[f"model.layers.{i}.mlp.experts.down_proj"]
    else:
        # Old per-expert format
        num_experts = len([j for j in state_dict.keys() if f"model.layers.{i}.mlp.experts" in j]) // 3
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        w1 = torch.empty(
            (num_experts, moe_dim, dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        )  # Gate
        w2 = torch.empty(
            (num_experts, dim, moe_dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        )  # Down
        w3 = torch.empty(
            (num_experts, moe_dim, dim), dtype=state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        )  # Up
        for j in range(num_experts):
            w1[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"])
            w2[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"])
            w3[j].copy_(state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"])

            del state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"]
            del state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"]

    state_dict[f"model.layers.{i}.mlp.experts.w1"] = w1
    state_dict[f"model.layers.{i}.mlp.experts.w2"] = w2
    state_dict[f"model.layers.{i}.mlp.experts.w3"] = w3


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert a layer from TT to HF format in-place."""
    i = layer_index
    if f"model.layers.{i}.mlp.router.gate.weight" not in state_dict:
        return

    # Gate / Router
    state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict[f"model.layers.{i}.mlp.router.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.router.gate.weight"]

    # Routed experts - convert to new fused format (transformers 5.0+)
    w1 = state_dict[f"model.layers.{i}.mlp.experts.w1"]  # (num_experts, moe_dim, dim)
    w2 = state_dict[f"model.layers.{i}.mlp.experts.w2"]  # (num_experts, dim, moe_dim)
    w3 = state_dict[f"model.layers.{i}.mlp.experts.w3"]  # (num_experts, moe_dim, dim)

    # Fuse w1 (gate) and w3 (up) into gate_up_proj
    gate_up_proj = torch.cat([w1, w3], dim=1)  # (num_experts, 2*moe_dim, dim)
    state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"] = gate_up_proj
    state_dict[f"model.layers.{i}.mlp.experts.down_proj"] = w2

    del state_dict[f"model.layers.{i}.mlp.experts.w1"]
    del state_dict[f"model.layers.{i}.mlp.experts.w2"]
    del state_dict[f"model.layers.{i}.mlp.experts.w3"]


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from HF to TT format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from TT to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
