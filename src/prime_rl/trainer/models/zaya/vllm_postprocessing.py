"""Zaya weight postprocessing for vLLM's original alternating-layer layout.

PrimeRL trains the Transformers-native hybrid decoder layout. The upstream vLLM
Zaya PR used the original alternating attention/MoE layer layout; this module is
its in-memory inverse of ``zaya_ref/convert_zaya_weights_to_hf.py``.
"""

import re

import torch
from torch import Tensor

from prime_rl.trainer.models.zaya.converting_zaya import convert_prime_to_hf, get_max_layer_num, is_prime_state_dict

_LAYER_PATTERN = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

_REVERSE_COMMON_REPLACEMENTS = (
    ("self_attn.qkv_proj.conv_qk_depthwise.", "self_attn.qkv.conv_qk.0."),
    ("self_attn.qkv_proj.conv_qk_grouped.", "self_attn.qkv.conv_qk.1."),
    ("self_attn.qk_norm.temp", "self_attn.qkv.temp"),
    ("self_attn.qkv_proj.q_proj.", "self_attn.qkv.linear_q."),
    ("self_attn.qkv_proj.k_proj.", "self_attn.qkv.linear_k."),
    ("self_attn.qkv_proj.v_proj_current.", "self_attn.qkv.val_proj1."),
    ("self_attn.qkv_proj.v_proj_delayed.", "self_attn.qkv.val_proj2."),
    ("self_attn.qkv_proj.", "self_attn.qkv."),
    ("mlp.gate.router_mlp.rmsnorm_eda.", "zaya_block.router.rmsnorm_eda."),
    ("mlp.gate.router_mlp.fc1.", "zaya_block.router.router_mlp.0."),
    ("mlp.gate.router_mlp.fc2.", "zaya_block.router.router_mlp.2."),
    ("mlp.gate.router_mlp.out_proj.", "zaya_block.router.router_mlp.4."),
    ("mlp.gate.", "zaya_block.router."),
    ("mlp.", "zaya_block."),
)


def _reverse_common(rest: str) -> str:
    for new, old in _REVERSE_COMMON_REPLACEMENTS:
        if rest.startswith(new):
            return old + rest.removeprefix(new)
    return rest


def _convert_hf_weight_name_to_vllm(name: str, num_hidden_layers: int) -> str | None:
    if name.startswith("model.input_"):
        return f"model.layers.0.res_scale.{name.removeprefix('model.input_')}"

    final_post_mlp_prefix = f"model.layers.{num_hidden_layers - 1}.post_mlp_residual_scale."
    if name.startswith(final_post_mlp_prefix):
        return f"model.res_scale.{name.removeprefix(final_post_mlp_prefix)}"

    match = _LAYER_PATTERN.match(name)
    if match is None:
        return name

    layer_idx = int(match.group(1))
    rest = match.group(2)

    if rest.startswith("mlp.experts.gate_up_proj") or rest.startswith("mlp.experts.down_proj"):
        return None

    if rest.startswith("self_attn."):
        return f"model.layers.{2 * layer_idx}.{_reverse_common(rest)}"
    if rest.startswith("input_layernorm."):
        return f"model.layers.{2 * layer_idx}.input_norm.{rest.removeprefix('input_layernorm.')}"
    if rest.startswith("post_mlp_residual_scale."):
        if layer_idx == num_hidden_layers - 1:
            return f"model.res_scale.{rest.removeprefix('post_mlp_residual_scale.')}"
        return f"model.layers.{2 * (layer_idx + 1)}.res_scale.{rest.removeprefix('post_mlp_residual_scale.')}"
    if rest.startswith("mlp."):
        return f"model.layers.{2 * layer_idx + 1}.{_reverse_common(rest)}"
    if rest.startswith("post_attention_layernorm."):
        return f"model.layers.{2 * layer_idx + 1}.input_norm.{rest.removeprefix('post_attention_layernorm.')}"
    if rest.startswith("post_attention_residual_scale."):
        return f"model.layers.{2 * layer_idx + 1}.res_scale.{rest.removeprefix('post_attention_residual_scale.')}"

    raise ValueError(f"Unexpected HF Zaya weight name: {name}")


def _add_vllm_experts(hf_state_dict: dict[str, Tensor], vllm_state_dict: dict[str, Tensor], layer_idx: int) -> None:
    gate_up_key = f"model.layers.{layer_idx}.mlp.experts.gate_up_proj"
    down_key = f"model.layers.{layer_idx}.mlp.experts.down_proj"
    if gate_up_key not in hf_state_dict:
        return

    gate_up = hf_state_dict[gate_up_key]
    down = hf_state_dict[down_key]
    moe_dim = gate_up.shape[1] // 2
    old_layer_idx = 2 * layer_idx + 1

    for expert_idx in range(gate_up.shape[0]):
        prefix = f"model.layers.{old_layer_idx}.zaya_block.experts.local_experts.{expert_idx}"
        vllm_state_dict[f"{prefix}.linear_fc1.weight"] = torch.cat(
            [gate_up[expert_idx, :moe_dim], gate_up[expert_idx, moe_dim:]], dim=0
        ).contiguous()
        vllm_state_dict[f"{prefix}.linear_fc2.weight"] = down[expert_idx].contiguous()


def _infer_num_hidden_layers(state_dict: dict[str, Tensor]) -> int:
    if not any(key.startswith("model.layers.") for key in state_dict):
        return 0
    return get_max_layer_num(state_dict)


def convert_hf_to_vllm(state_dict: dict[str, Tensor], num_hidden_layers: int | None = None) -> dict[str, Tensor]:
    """Convert Transformers-native Zaya weights to vLLM original Zaya weights."""
    if num_hidden_layers is None:
        num_hidden_layers = _infer_num_hidden_layers(state_dict)
    converted: dict[str, Tensor] = {}

    for name, tensor in state_dict.items():
        if name == "lm_head.weight":
            continue
        target = _convert_hf_weight_name_to_vllm(name, num_hidden_layers)
        if target is not None:
            converted[target] = tensor.contiguous()

    for layer_idx in range(num_hidden_layers):
        _add_vllm_experts(state_dict, converted, layer_idx)

    return converted


def convert_prime_to_vllm(state_dict: dict[str, Tensor], num_hidden_layers: int | None = None) -> dict[str, Tensor]:
    """Convert PrimeRL training-format Zaya weights to vLLM original Zaya weights."""
    hf_state_dict = dict(state_dict)
    if is_prime_state_dict(hf_state_dict):
        hf_state_dict = convert_prime_to_hf(hf_state_dict)
    return convert_hf_to_vllm(hf_state_dict, num_hidden_layers=num_hidden_layers)


class ZayaVLLMWeightPostprocessor:
    """Callable postprocessor shared by checkpoint and NCCL broadcast paths."""

    def __init__(self, num_hidden_layers: int | None = None):
        self.num_hidden_layers = num_hidden_layers

    def __call__(self, state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        return convert_prime_to_vllm(state_dict, num_hidden_layers=self.num_hidden_layers)


__all__ = ["ZayaVLLMWeightPostprocessor", "convert_hf_to_vllm", "convert_prime_to_vllm"]
