import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _is_moe_layer(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    """Check if a layer is an MoE layer by looking for the router gate weight."""
    return f"model.layers.{layer_idx}.mlp.gate.weight" in state_dict


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    i = layer_idx

    if not _is_moe_layer(state_dict, i):
        return

    # Router: gate.weight -> router.gate.weight
    state_dict[f"model.layers.{i}.mlp.router.gate.weight"] = state_dict[f"model.layers.{i}.mlp.gate.weight"]
    del state_dict[f"model.layers.{i}.mlp.gate.weight"]

    # Routed experts: fused or per-expert format -> stacked w1/w2/w3
    if f"model.layers.{i}.mlp.experts.gate_up_proj" in state_dict:
        gate_up_proj = state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        down_proj = state_dict[f"model.layers.{i}.mlp.experts.down_proj"]

        num_experts, fused_dim, dim = gate_up_proj.shape
        moe_dim = fused_dim // 2

        w1 = gate_up_proj[:, :moe_dim, :]
        w3 = gate_up_proj[:, moe_dim:, :]
        w2 = down_proj

        del state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"]
        del state_dict[f"model.layers.{i}.mlp.experts.down_proj"]
    else:
        num_experts = len([j for j in state_dict.keys() if f"model.layers.{i}.mlp.experts" in j]) // 3
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].shape
        dtype = state_dict[f"model.layers.{i}.mlp.experts.0.down_proj.weight"].dtype
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype)
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

    # Shared experts
    state_dict[f"model.layers.{i}.mlp.shared_expert.w1"] = state_dict[
        f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
    ]
    state_dict[f"model.layers.{i}.mlp.shared_expert.w2"] = state_dict[
        f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
    ]
    state_dict[f"model.layers.{i}.mlp.shared_expert.w3"] = state_dict[
        f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
    ]
    del state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"]
    del state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"]
    del state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"]

    # Expert bias for load balancing
    state_dict[f"model.layers.{i}.mlp.expert_bias"] = state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]
    del state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"]


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    i = layer_index

    # Expert bias
    if f"model.layers.{i}.mlp.expert_bias" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.e_score_correction_bias"] = state_dict[
            f"model.layers.{i}.mlp.expert_bias"
        ]
        del state_dict[f"model.layers.{i}.mlp.expert_bias"]
    if f"model.layers.{i}.mlp.tokens_per_expert" in state_dict:
        del state_dict[f"model.layers.{i}.mlp.tokens_per_expert"]

    # Shared experts
    if f"model.layers.{i}.mlp.shared_expert.w1" in state_dict:
        state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w1"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w2"
        ]
        state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
            f"model.layers.{i}.mlp.shared_expert.w3"
        ]

        if state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"].shape[0] == 1:
            state_dict[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.up_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.down_proj.weight"
            ][0]
            state_dict[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = state_dict[
                f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"
            ][0]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w1"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w2"]
        del state_dict[f"model.layers.{i}.mlp.shared_expert.w3"]

    # Router
    if f"model.layers.{i}.mlp.router.gate.weight" in state_dict:
        state_dict[f"model.layers.{i}.mlp.gate.weight"] = state_dict[f"model.layers.{i}.mlp.router.gate.weight"]
        del state_dict[f"model.layers.{i}.mlp.router.gate.weight"]

        # Routed experts: stacked w1/w2/w3 -> fused gate_up_proj + down_proj
        w1 = state_dict[f"model.layers.{i}.mlp.experts.w1"]
        w2 = state_dict[f"model.layers.{i}.mlp.experts.w2"]
        w3 = state_dict[f"model.layers.{i}.mlp.experts.w3"]

        gate_up_proj = torch.cat([w1, w3], dim=1)
        state_dict[f"model.layers.{i}.mlp.experts.gate_up_proj"] = gate_up_proj
        state_dict[f"model.layers.{i}.mlp.experts.down_proj"] = w2

        del state_dict[f"model.layers.{i}.mlp.experts.w1"]
        del state_dict[f"model.layers.{i}.mlp.experts.w2"]
        del state_dict[f"model.layers.{i}.mlp.experts.w3"]


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
