import torch
from torch import Tensor


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a layer from HF to TT format in-place.
    """
    i = layer_idx
    if f"model.layers.{i}.self_attn.o_proj.bias" in state_dict:
        del state_dict[f"model.layers.{i}.self_attn.o_proj.bias"]


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert a layer from TT to HF format in-place."""
    i = layer_index

    sinks_key = f"model.layers.{i}.self_attn.sinks"
    if sinks_key not in state_dict:
        q_proj_key = f"model.layers.{i}.self_attn.q_proj.weight"

        if q_proj_key in state_dict:
            q_proj_dim = state_dict[q_proj_key].shape[0]
            num_heads = 8  # fallback

            for common_head_dim in [64, 128, 256]:
                if q_proj_dim % common_head_dim == 0:
                    num_heads = q_proj_dim // common_head_dim
                    break

            state_dict[sinks_key] = torch.zeros(num_heads, device=state_dict[q_proj_key].device)


def convert_hf_to_tt_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from HF to TT format in-place.
    """
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_to_hf_moe(state_dict: dict[str, Tensor]):
    """Convert MoE weights from TT to HF format in-place.
    """
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
