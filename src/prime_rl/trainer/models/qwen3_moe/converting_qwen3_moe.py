import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a layer from HF to TT format in-place."""
    i = layer_idx

    num_experts = len([j for j in state_dict.keys() if f"model.layers.{i}.mlp.experts" in j]) // 3
    if num_experts == 0:
        return

    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.gate.weight"]
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

    # Routed experts
    num_experts, moe_dim, dim = state_dict[f"model.layers.{i}.mlp.experts.w1"].shape
    for j in range(num_experts):
        state_dict[f"model.layers.{i}.mlp.experts.{j}.gate_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.experts.w1"
        ][j]
        state_dict[f"model.layers.{i}.mlp.experts.{j}.down_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.experts.w2"
        ][j]
        state_dict[f"model.layers.{i}.mlp.experts.{j}.up_proj.weight"] = state_dict[f"model.layers.{i}.mlp.experts.w3"][
            j
        ]
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
