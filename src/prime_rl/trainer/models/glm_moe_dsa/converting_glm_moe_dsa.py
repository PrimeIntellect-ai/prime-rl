"""HF<->PrimeRL weight conversion for GLM-MoE-DSA.

The MoE layout is identical to GLM4-MoE. Attention and sparse-indexer
parameters retain their names in both formats.
"""

from __future__ import annotations

from torch import Tensor

from prime_rl.trainer.models.conversion_ops import ConvOp
from prime_rl.trainer.models.fp8 import quantize_to_fp8_checkpoint
from prime_rl.trainer.models.glm4_moe.converting_glm4_moe import glm_moe_layer_ops


def conversion_chain(config) -> list[ConvOp]:
    ops: list[ConvOp] = []
    for layer_idx in range(config.num_hidden_layers):
        ops.extend(glm_moe_layer_ops(layer_idx))
    return ops


def _keep_unquantized(name: str, tensor: Tensor) -> bool:
    """Whether ``name`` stays in bf16/fp32 in the wire (and disk FP8) format.

    Mirrors the module set the FP8 checkpoints of this architecture keep
    unquantized (their ``modules_to_not_convert``): layernorms, the router
    gate and its bias, the indexer k-norm, and the indexer ``weights_proj``
    (the engine stores the fused ``wk_weights_proj`` in bf16 and dequantizes
    the fp8 ``wk`` shard while loading).
    """
    return any(
        marker in name
        for marker in (
            "layernorm",  # input/post layernorms, q_a_layernorm, kv_a_layernorm
            ".indexer.k_norm",  # indexer norm weight and bias
            ".indexer.weights_proj.",  # bf16 fused wk_weights_proj shard 1
            ".mlp.gate.",  # router gate weight + e_score_correction_bias
        )
    )


def quantize_tt_layer_to_vllm_fp8_checkpoint(state_dict: dict[str, Tensor], layer_idx: int) -> dict[str, Tensor]:
    """Quantize one GLM layer from HF checkpoint naming to the FP8 checkpoint wire format.

    The input is a single layer in HF checkpoint naming (as produced by
    ``preprocess_layer_checkpoint``); the output carries fp8 e4m3 weights plus
    fp32 ``weight_scale_inv`` scales for exactly the tensors a vLLM fp8 engine
    loads quantized from a blockwise-fp8 checkpoint of this architecture.
    Receiving engines feed this stream through vLLM's own weight-loading path,
    which performs the TP/EP slicing, the fused-parameter layout (including
    ``fused_qkv_a_proj`` and the fused indexer ``wk_weights_proj``), the fp8
    scale handling, and the MLA absorbed-weight recompute.
    """
    prefix = f"model.layers.{layer_idx}"
    unexpected = [name for name in state_dict if not name.startswith(f"{prefix}.")]
    if unexpected:
        raise ValueError(f"FP8 checkpoint wire conversion got tensors from outside layer {layer_idx}: {unexpected}")

    return quantize_to_fp8_checkpoint(state_dict, keep_unquantized=_keep_unquantized)
