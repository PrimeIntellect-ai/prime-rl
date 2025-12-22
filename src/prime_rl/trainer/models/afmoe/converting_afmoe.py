import re

import torch
from torch import Tensor


def get_num_experts_from_state_dict(state_dict: dict[str, Tensor], layer_idx: int) -> int:
    """Get the number of experts from the HF state dict for a given layer."""
    pattern = re.compile(rf"model\.layers\.{layer_idx}\.mlp\.experts\.(\d+)\.")
    expert_indices = set()
    for key in state_dict.keys():
        match = pattern.search(key)
        if match:
            expert_indices.add(int(match.group(1)))
    return len(expert_indices)


def is_moe_layer_hf(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    """Check if a layer is a MoE layer in HF format."""
    return any(f"model.layers.{layer_idx}.mlp.experts." in k for k in state_dict.keys())


def is_moe_layer_tt(state_dict: dict[str, Tensor], layer_idx: int) -> bool:
    """Check if a layer is a MoE layer in TT format."""
    return f"model.layers.{layer_idx}.mlp.experts.w1" in state_dict


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    """Convert a single MoE layer from HF format to TT format in-place.
    
    Args:
        state_dict: The state dict to modify in-place.
        layer_idx: The layer index to convert.
    """
    prefix = f"model.layers.{layer_idx}.mlp"
    
    # Check if this is a MoE layer (has individual experts)
    is_moe_layer = is_moe_layer_hf(state_dict, layer_idx)
    if not is_moe_layer:
        return  # Dense layer, no conversion needed
    
    num_experts = get_num_experts_from_state_dict(state_dict, layer_idx)
    if num_experts == 0:
        return
    
    # Convert shared experts: shared_experts -> shared_expert, gate_proj/up_proj/down_proj -> w1/w3/w2
    shared_mappings = [
        (f"{prefix}.shared_experts.gate_proj.weight", f"{prefix}.shared_expert.w1"),
        (f"{prefix}.shared_experts.down_proj.weight", f"{prefix}.shared_expert.w2"),
        (f"{prefix}.shared_experts.up_proj.weight", f"{prefix}.shared_expert.w3"),
    ]
    for hf_key, tt_key in shared_mappings:
        if hf_key in state_dict:
            state_dict[tt_key] = state_dict.pop(hf_key)

    # Dense layers should not carry expert_bias; keep it for MoE layers.
    expert_bias_key = f"{prefix}.expert_bias"
    if expert_bias_key in state_dict and not is_moe_layer:
        del state_dict[expert_bias_key]

    
    # Stack individual expert weights into grouped format
    # HF: experts.{i}.gate_proj.weight [hidden_dim, hidden_size]
    # TT: experts.w1 [num_experts, hidden_dim, hidden_size]
    expert_mappings = [
        ("gate_proj", "w1"),
        ("down_proj", "w2"),
        ("up_proj", "w3"),
    ]
    
    for hf_name, tt_name in expert_mappings:
        expert_weights = []
        for i in range(num_experts):
            hf_key = f"{prefix}.experts.{i}.{hf_name}.weight"
            if hf_key in state_dict:
                expert_weights.append(state_dict.pop(hf_key))
        
        if expert_weights:
            # Stack along first dimension: [num_experts, out_dim, in_dim]
            stacked = torch.stack(expert_weights, dim=0)
            state_dict[f"{prefix}.experts.{tt_name}"] = stacked


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_idx: int) -> None:
    """Convert a single MoE layer from TT format to HF format in-place.
    
    Args:
        state_dict: The state dict to modify in-place.
        layer_idx: The layer index to convert.
    """
    prefix = f"model.layers.{layer_idx}.mlp"
    
    # Check if this is a MoE layer in TT format
    if not is_moe_layer_tt(state_dict, layer_idx):
        return  # Dense layer, no conversion needed
    
    # Convert shared expert: shared_expert -> shared_experts, w1/w2/w3 -> gate_proj/down_proj/up_proj
    shared_mappings = [
        (f"{prefix}.shared_expert.w1", f"{prefix}.shared_experts.gate_proj.weight"),
        (f"{prefix}.shared_expert.w2", f"{prefix}.shared_experts.down_proj.weight"),
        (f"{prefix}.shared_expert.w3", f"{prefix}.shared_experts.up_proj.weight"),
    ]
    for tt_key, hf_key in shared_mappings:
        if tt_key in state_dict:
            state_dict[hf_key] = state_dict.pop(tt_key)
    
    # Unstack grouped expert weights into individual experts
    expert_mappings = [
        ("w1", "gate_proj"),
        ("w2", "down_proj"),
        ("w3", "up_proj"),
    ]
    
    for tt_name, hf_name in expert_mappings:
        tt_key = f"{prefix}.experts.{tt_name}"
        if tt_key in state_dict:
            stacked = state_dict.pop(tt_key)
            num_experts = stacked.shape[0]
            for i in range(num_experts):
                hf_key = f"{prefix}.experts.{i}.{hf_name}.weight"
                state_dict[hf_key] = stacked[i]
    
    # Remove TT-specific buffers that don't exist in HF
    for key in list(state_dict.keys()):
        if f"{prefix}.tokens_per_expert" in key:
            del state_dict[key]
        if f"{prefix}.reorderer" in key:
            del state_dict[key]


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum layer number from the state dict."""
    max_layer = -1
    for key in state_dict.keys():
        match = re.search(r"model\.layers\.(\d+)\.", key)
        if match:
            layer_idx = int(match.group(1))
            max_layer = max(max_layer, layer_idx)
    return max_layer


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]) -> None:
    """Convert all MoE layers from HF format to TT format in-place."""
    max_layer = get_max_layer_num(state_dict)
    for layer_idx in range(max_layer + 1):
        convert_hf_layer_to_tt(state_dict, layer_idx)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]) -> None:
    """Convert all MoE layers from TT format to HF format in-place."""
    max_layer = get_max_layer_num(state_dict)
    for layer_idx in range(max_layer + 1):
        convert_tt_layer_to_hf(state_dict, layer_idx)
