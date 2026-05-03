import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    max_layer = -1
    for key in state_dict:
        if not key.startswith("model.layers."):
            continue
        layer_idx = key.split(".")[2]
        if layer_idx.isdigit():
            max_layer = max(max_layer, int(layer_idx))
    return max_layer + 1


def _convert_hf_moe_prefix_to_tt(state_dict: dict[str, Tensor], prefix: str):
    # Router: mlp.gate.weight -> mlp.router.gate.weight
    gate_key = f"{prefix}mlp.gate.weight"
    if gate_key not in state_dict:
        return

    state_dict[f"{prefix}mlp.router.gate.weight"] = state_dict.pop(gate_key)

    # Routed experts: convert to fused w1/w2/w3 format
    if f"{prefix}mlp.experts.gate_up_proj" in state_dict:
        # New fused format (transformers 5.0+): gate_up_proj shape (num_experts, 2*moe_dim, dim)
        gate_up_proj = state_dict.pop(f"{prefix}mlp.experts.gate_up_proj")
        down_proj = state_dict.pop(f"{prefix}mlp.experts.down_proj")

        moe_dim = gate_up_proj.shape[1] // 2
        w1 = gate_up_proj[:, :moe_dim, :]  # gate
        w3 = gate_up_proj[:, moe_dim:, :]  # up
        w2 = down_proj  # down
    else:
        # Old per-expert format
        num_experts = len([k for k in state_dict.keys() if f"{prefix}mlp.experts" in k and "gate_proj" in k])
        if num_experts == 0:
            return

        dim, moe_dim = state_dict[f"{prefix}mlp.experts.0.down_proj.weight"].shape
        first_expert = state_dict[f"{prefix}mlp.experts.0.down_proj.weight"]
        dtype = first_expert.dtype
        device = first_expert.device
        w1 = torch.empty((num_experts, moe_dim, dim), dtype=dtype, device=device)
        w2 = torch.empty((num_experts, dim, moe_dim), dtype=dtype, device=device)
        w3 = torch.empty((num_experts, moe_dim, dim), dtype=dtype, device=device)
        for j in range(num_experts):
            w1[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.gate_proj.weight"))
            w2[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.down_proj.weight"))
            w3[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.up_proj.weight"))

    state_dict[f"{prefix}mlp.experts.w1"] = w1
    state_dict[f"{prefix}mlp.experts.w2"] = w2
    state_dict[f"{prefix}mlp.experts.w3"] = w3

    # Shared expert: mlp.shared_expert.{gate,up,down}_proj -> shared_expert.{w1,w3,w2}
    se_gate_key = f"{prefix}mlp.shared_expert.gate_proj.weight"
    if se_gate_key in state_dict:
        state_dict[f"{prefix}shared_expert.w1.weight"] = state_dict.pop(se_gate_key)
        state_dict[f"{prefix}shared_expert.w2.weight"] = state_dict.pop(f"{prefix}mlp.shared_expert.down_proj.weight")
        state_dict[f"{prefix}shared_expert.w3.weight"] = state_dict.pop(f"{prefix}mlp.shared_expert.up_proj.weight")

    # Shared expert gate: mlp.shared_expert_gate.weight -> shared_expert_gate.weight
    seg_key = f"{prefix}mlp.shared_expert_gate.weight"
    if seg_key in state_dict:
        state_dict[f"{prefix}shared_expert_gate.weight"] = state_dict.pop(seg_key)


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single layer from HF to PrimeRL format in-place."""
    _convert_hf_moe_prefix_to_tt(state_dict, f"model.layers.{layer_idx}.")


def _convert_tt_moe_prefix_to_hf(state_dict: dict[str, Tensor], prefix: str):
    # Router
    router_key = f"{prefix}mlp.router.gate.weight"
    if router_key not in state_dict:
        return

    state_dict[f"{prefix}mlp.gate.weight"] = state_dict.pop(router_key)

    # Routed experts: w1/w2/w3 -> per-expert format
    w1 = state_dict.pop(f"{prefix}mlp.experts.w1")
    w2 = state_dict.pop(f"{prefix}mlp.experts.w2")
    w3 = state_dict.pop(f"{prefix}mlp.experts.w3")

    num_experts = w1.shape[0]
    for j in range(num_experts):
        state_dict[f"{prefix}mlp.experts.{j}.gate_proj.weight"] = w1[j]
        state_dict[f"{prefix}mlp.experts.{j}.down_proj.weight"] = w2[j]
        state_dict[f"{prefix}mlp.experts.{j}.up_proj.weight"] = w3[j]

    # Shared expert: shared_expert.{w1,w2,w3} -> mlp.shared_expert.{gate,down,up}_proj
    se_w1_key = f"{prefix}shared_expert.w1.weight"
    if se_w1_key in state_dict:
        state_dict[f"{prefix}mlp.shared_expert.gate_proj.weight"] = state_dict.pop(se_w1_key)
        state_dict[f"{prefix}mlp.shared_expert.down_proj.weight"] = state_dict.pop(f"{prefix}shared_expert.w2.weight")
        state_dict[f"{prefix}mlp.shared_expert.up_proj.weight"] = state_dict.pop(f"{prefix}shared_expert.w3.weight")

    # Shared expert gate
    seg_key = f"{prefix}shared_expert_gate.weight"
    if seg_key in state_dict:
        state_dict[f"{prefix}mlp.shared_expert_gate.weight"] = state_dict.pop(seg_key)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single layer from PrimeRL to HF format in-place."""
    _convert_tt_moe_prefix_to_hf(state_dict, f"model.layers.{layer_idx}.")


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert all MoE weights from HF to PrimeRL format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)
    convert_hf_to_tt_mtp(state_dict)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert all MoE weights from PrimeRL to HF format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
    convert_tt_to_hf_mtp(state_dict)


def convert_hf_to_tt_mtp(state_dict: dict[str, Tensor]) -> None:
    if not any(key.startswith("mtp.") for key in state_dict):
        return

    rename = {
        "mtp.pre_fc_norm_embedding.weight": "mtp_layers.0.enorm.weight",
        "mtp.pre_fc_norm_hidden.weight": "mtp_layers.0.hnorm.weight",
        "mtp.fc.weight": "mtp_layers.0.eh_proj.weight",
        "mtp.norm.weight": "mtp_layers.0.norm.weight",
    }
    for src, dst in rename.items():
        if src in state_dict:
            state_dict[dst] = state_dict.pop(src)

    for key in [key for key in state_dict if key.startswith("mtp.layers.")]:
        parts = key.split(".")
        if len(parts) < 4 or not parts[2].isdigit():
            continue
        layer_idx = parts[2]
        suffix = ".".join(parts[3:])
        state_dict[f"mtp_layers.{layer_idx}.block.{suffix}"] = state_dict.pop(key)

    mtp_indices = sorted(
        {int(key.split(".")[1]) for key in state_dict if key.startswith("mtp_layers.") and key.split(".")[1].isdigit()}
    )
    for layer_idx in mtp_indices:
        _convert_hf_moe_prefix_to_tt(state_dict, f"mtp_layers.{layer_idx}.block.")


def convert_tt_to_hf_mtp(state_dict: dict[str, Tensor]) -> None:
    if not any(key.startswith("mtp_layers.") for key in state_dict):
        return

    mtp_indices = sorted(
        {int(key.split(".")[1]) for key in state_dict if key.startswith("mtp_layers.") and key.split(".")[1].isdigit()}
    )
    for layer_idx in mtp_indices:
        _convert_tt_moe_prefix_to_hf(state_dict, f"mtp_layers.{layer_idx}.block.")

    rename = {
        "mtp_layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        "mtp_layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        "mtp_layers.0.eh_proj.weight": "mtp.fc.weight",
        "mtp_layers.0.norm.weight": "mtp.norm.weight",
    }
    for src, dst in rename.items():
        if src in state_dict:
            state_dict[dst] = state_dict.pop(src)

    for key in [key for key in state_dict if key.startswith("mtp_layers.") and ".block." in key]:
        parts = key.split(".")
        if len(parts) < 5 or not parts[1].isdigit():
            continue
        layer_idx = parts[1]
        suffix = ".".join(parts[3:])
        state_dict[f"mtp.layers.{layer_idx}.{suffix.removeprefix('block.')}"] = state_dict.pop(key)
