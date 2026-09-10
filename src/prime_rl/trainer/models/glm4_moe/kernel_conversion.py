"""GLM Air's MHA weights plus the shared GLM FP8 MoE transfer layout."""

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import quantize_to_vllm_kernel_format
from prime_rl.trainer.models.glm_moe_dsa.converting_glm_moe_dsa import convert_tt_layer_to_vllm_kernel


def convert_glm4_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor], layer_idx: int, quantize_fp8: bool = False
) -> dict[str, Tensor]:
    # Both GLM architectures use identical norms, router, dense/shared MLPs and
    # routed-expert storage. Their attention projections differ (MHA vs MLA).
    out = convert_tt_layer_to_vllm_kernel(state_dict, layer_idx, quantize_fp8)
    prefix = f"model.layers.{layer_idx}.self_attn"
    qkv_keys = [f"{prefix}.{part}_proj.weight" for part in ("q", "k", "v")]
    if any(key in state_dict for key in qkv_keys):
        qkv = torch.cat([state_dict[key] for key in qkv_keys], dim=0)
        if quantize_fp8:
            out[f"{prefix}.qkv_proj.weight"], out[f"{prefix}.qkv_proj.weight_scale_inv"] = (
                quantize_to_vllm_kernel_format(qkv)
            )
        else:
            out[f"{prefix}.qkv_proj.weight"] = qkv
    for suffix in ("q_norm.weight", "k_norm.weight"):
        name = f"{prefix}.{suffix}"
        if name in state_dict:
            out[name] = state_dict[name]
    bias_keys = [f"{prefix}.{part}_proj.bias" for part in ("q", "k", "v")]
    if any(key in state_dict for key in bias_keys):
        out[f"{prefix}.qkv_proj.bias"] = torch.cat([state_dict[key] for key in bias_keys], dim=0)

    # The online inference patch pads the activation K dimension. Preserve the
    # same padded kernel shape on the wire; scales already cover the tail block.
    if quantize_fp8:
        for name, weight in list(out.items()):
            if weight.dtype == torch.float8_e4m3fn and weight.ndim == 2 and weight.shape[-1] % 128:
                padding = (-weight.shape[-1]) % 128
                padded = torch.zeros(
                    weight.shape[0], weight.shape[1] + padding, dtype=weight.dtype, device=weight.device
                )
                padded[:, : weight.shape[1]].copy_(weight)
                out[name] = padded
    return out
