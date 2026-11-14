"""DeepSeek V4.1's indexer queries, keys and weights as vLLM's MXFP4 indexer scores them.

With ``attention_config.indexer_kv_dtype = "mxfp4"`` (SM100 only in vLLM 0.33, and required by
``indexer_sparse_logits``) both sides of the indexer dot product are MXFP4:

- Keys (``vllm/models/deepseek_v41/common/ops/indexer_k_store.py``, ``_indexer_k_norm_rope_quant_store_kernel``):
  each 128-wide key, after k_norm and RoPE with a BF16 round trip, is stored as 64 bytes of E2M1 pairs plus four
  UE8M0 bytes, one per 32 consecutive channels. No per-token scale.
- Queries (``vllm/models/deepseek_v4/common/ops/fused_indexer_q.py``, the MXFP4 kernels): each (token, head) row is
  quantized the same way, four 32-wide blocks with their own UE8M0 scales.

A block's scale is ``2^ceil(log2(max(amax, 6 * 2^-126) * fp32(1 / 6)))`` and its values ``e2m1(x * 2^-e)``,
rounded to nearest even and saturated at 6 (``cvt.rn.satfinite.e2m1x2``). DeepGEMM's ``fp8_fp4_mqa_logits`` and
sparse MQA-logits kernels multiply the decoded blocks with block-scaled MMAs, so the score is the plain
``sum_h w_h * relu(q_h . k)`` of the decoded values; each product is exact in float32.

On SM100 those kernels take BF16 weights: ``bf16(w * fp32(softmax_scale * head_scale))``, with no query scale
folded in (the MXFP4 query carries its own). Rounding the scaled weight to BF16 changes the relative weight of the
heads, so the trainer does the same.
"""

import torch

from prime_rl.trainer.models.kernels.deepseek_v4.dsv41_nvfp4_kv_fake_quant import ONE_SIXTH, e2m1_round

MXFP4_BLOCK = 32
# vLLM floors each block's absolute maximum at 6 * 2^-126, so the smallest scale is 2^-126
AMAX_FLOOR = 6.0 * 2.0**-126


def ue8m0_exponent(amax: torch.Tensor) -> torch.Tensor:
    """``ceil(log2(max(amax, 6 * 2^-126) * fp32(1 / 6)))`` as int32, exactly: the float32 bits plus the mantissa
    mask carry into the exponent unless the product is a power of two (the CuTe DSL query kernel's arithmetic)."""
    scale = amax.float().clamp_min(AMAX_FLOOR) * ONE_SIXTH
    bits = scale.view(torch.int32).long()
    return (((bits + 0x7FFFFF) >> 23) & 0xFF).int() - 127


def mxfp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The E2M1 values (float32, ``(..., n_blocks, 32)``) and block exponents (int32, ``(..., n_blocks, 1)``) vLLM
    stores for ``x`` (``(..., D)``, D a multiple of 32)."""
    D = x.shape[-1]
    assert D % MXFP4_BLOCK == 0, f"{D=} is not a multiple of {MXFP4_BLOCK}"
    blocks = x.float().reshape(*x.shape[:-1], D // MXFP4_BLOCK, MXFP4_BLOCK)
    exponent = ue8m0_exponent(blocks.abs().amax(dim=-1, keepdim=True))
    # Scaling by a power of two is exact (the scale keeps every block value at most 6, far from overflow)
    values = e2m1_round(blocks * torch.exp2(-exponent.float()))
    return values, exponent


def mxfp4_round(x: torch.Tensor) -> torch.Tensor:
    """``x`` as vLLM's MXFP4 indexer reads it back, in ``x``'s dtype. E2M1 times a power of two has at most two
    significant bits, so BF16 holds it exactly (down to 2^-127, a BF16 subnormal)."""
    values, exponent = mxfp4_quant(x)
    return (values * torch.exp2(exponent.float())).reshape(x.shape).to(x.dtype)


def mxfp4_indexer_weights(w: torch.Tensor, head_dim: int, n_heads: int) -> torch.Tensor:
    """The per-head weights vLLM's SM100 MXFP4 scoring kernels take: ``bf16(fp32(w) * fp32(head_dim^-0.5 *
    n_heads^-0.5))``."""
    scale = torch.tensor(head_dim**-0.5 * n_heads**-0.5, dtype=torch.float32)
    return (w.float() * scale.to(w.device)).to(torch.bfloat16)


__all__ = ["MXFP4_BLOCK", "mxfp4_indexer_weights", "mxfp4_quant", "mxfp4_round", "ue8m0_exponent"]
