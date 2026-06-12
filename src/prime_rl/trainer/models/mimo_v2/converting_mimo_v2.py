import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import dequantize_fp8_blockwise

# Towers and MTP layers present in the hub checkpoint that the text backbone does not train
_DROPPED_PREFIXES = ("model.mtp.", "visual.", "audio_encoder.", "speech_embeddings.")


def get_max_layer_num(state_dict: dict[str, Tensor]) -> int:
    """Get the maximum number of layers in the model."""
    return max(int(i.split(".")[2]) for i in state_dict.keys() if "model.layers." in i) + 1


def _dequantize_layer(state_dict: dict[str, Tensor], prefix: str):
    """Dequantize all block-wise FP8 weights under a layer prefix in-place."""
    scale_keys = [k for k in state_dict if k.startswith(prefix) and k.endswith(".weight_scale_inv")]
    for scale_key in scale_keys:
        weight_key = scale_key.removesuffix("_scale_inv")
        state_dict[weight_key] = dequantize_fp8_blockwise(state_dict[weight_key], state_dict.pop(scale_key))


def convert_hf_layer_to_tt(state_dict: dict[str, Tensor], layer_idx: int):
    """Convert a single layer from HF (hub) format to TT format in-place.

    The attention projections (fused `qkv_proj`, `o_proj`) and the `attention_sink_bias` keep
    their hub names; only the FP8 weights and the MoE block are converted.
    """
    i = layer_idx
    prefix = f"model.layers.{i}."

    _dequantize_layer(state_dict, prefix)

    if f"{prefix}mlp.gate.weight" not in state_dict:
        return  # dense layer

    state_dict[f"{prefix}mlp.router.gate.weight"] = state_dict.pop(f"{prefix}mlp.gate.weight")
    state_dict[f"{prefix}mlp.expert_bias"] = state_dict.pop(f"{prefix}mlp.gate.e_score_correction_bias")

    num_experts = len([k for k in state_dict if f"{prefix}mlp.experts." in k]) // 3
    if num_experts == 0:
        return

    reference = state_dict[f"{prefix}mlp.experts.0.down_proj.weight"]
    dim, moe_dim = reference.shape
    w1 = torch.empty((num_experts, moe_dim, dim), dtype=reference.dtype, device=reference.device)  # Gate
    w2 = torch.empty((num_experts, dim, moe_dim), dtype=reference.dtype, device=reference.device)  # Down
    w3 = torch.empty((num_experts, moe_dim, dim), dtype=reference.dtype, device=reference.device)  # Up
    for j in range(num_experts):
        w1[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.gate_proj.weight"))
        w2[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.down_proj.weight"))
        w3[j].copy_(state_dict.pop(f"{prefix}mlp.experts.{j}.up_proj.weight"))

    state_dict[f"{prefix}mlp.experts.w1"] = w1
    state_dict[f"{prefix}mlp.experts.w2"] = w2
    state_dict[f"{prefix}mlp.experts.w3"] = w3


def convert_hf_to_tt(state_dict: dict[str, Tensor]):
    """Convert weights from HF (hub) format to TT format in-place."""
    for key in [k for k in state_dict if k.startswith(_DROPPED_PREFIXES)]:
        del state_dict[key]

    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_hf_layer_to_tt(state_dict, i)


def convert_tt_layer_to_hf(state_dict: dict[str, Tensor], layer_index: int):
    """Convert a layer from TT to HF (hub) format in-place.

    Emits BF16 weights with the hub key names. Re-quantizing to the hub's block-wise FP8 is not
    performed here.
    """
    i = layer_index
    prefix = f"model.layers.{i}."

    # Load balancing stats are training-only state
    if f"{prefix}mlp.tokens_per_expert" in state_dict:
        del state_dict[f"{prefix}mlp.tokens_per_expert"]

    if f"{prefix}mlp.router.gate.weight" not in state_dict:
        return  # dense layer

    state_dict[f"{prefix}mlp.gate.weight"] = state_dict.pop(f"{prefix}mlp.router.gate.weight")
    state_dict[f"{prefix}mlp.gate.e_score_correction_bias"] = state_dict.pop(f"{prefix}mlp.expert_bias")

    if f"{prefix}mlp.experts.w1" in state_dict:
        w1 = state_dict.pop(f"{prefix}mlp.experts.w1")  # (num_experts, moe_dim, dim)
        w2 = state_dict.pop(f"{prefix}mlp.experts.w2")  # (num_experts, dim, moe_dim)
        w3 = state_dict.pop(f"{prefix}mlp.experts.w3")  # (num_experts, moe_dim, dim)

        num_experts = w1.shape[0]
        for j in range(num_experts):
            state_dict[f"{prefix}mlp.experts.{j}.gate_proj.weight"] = w1[j]
            state_dict[f"{prefix}mlp.experts.{j}.down_proj.weight"] = w2[j]
            state_dict[f"{prefix}mlp.experts.{j}.up_proj.weight"] = w3[j]


def convert_tt_to_hf(state_dict: dict[str, Tensor]):
    """Convert weights from TT to HF (hub) format in-place."""
    num_layers = get_max_layer_num(state_dict)
    for i in range(num_layers):
        convert_tt_layer_to_hf(state_dict, i)
