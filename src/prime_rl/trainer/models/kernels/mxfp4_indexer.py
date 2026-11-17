"""Sparse-attention indexer queries, keys and weights as vLLM's MXFP4 indexer cache scores them.

Ported from #4492's ``dsv41_mxfp4_indexer`` (DeepSeek V4.1), which vLLM 0.33 quantizes the same way as DeepSeek V4.
With ``attention_config.indexer_kv_dtype = "mxfp4"`` (SM10x only, see ``dsa_indexer_uses_fp4`` in
``vllm/v1/attention/backends/mla/indexer.py``) both sides of the indexer dot product are MXFP4:

- Keys (DeepSeek V4: ``vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py``,
  ``_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn``): each 128-wide compressed key, after the norm and RoPE
  with a BF16 round trip, is stored as 64 bytes of E2M1 pairs plus four UE8M0 bytes, one per 32 consecutive
  channels. No per-token scale.
- Queries (``vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`` and its CuTe DSL twin
  ``nvidia/ops/fused_indexer_q_cutedsl.py``): each (token, head) row is quantized the same way, four 32-wide blocks
  with their own UE8M0 scales.

A block's scale is ``2^ceil(log2(max(amax, 6 * 2^-126) * fp32(1 / 6)))`` and its values ``e2m1(x * 2^-e)``, rounded
to nearest even and saturated at 6 (``cvt.rn.satfinite.e2m1x2``). DeepGEMM's ``fp8_fp4_mqa_logits`` multiplies the
decoded blocks with block-scaled MMAs, so the score is the plain ``sum_h w_h * relu(q_h . k)`` of the decoded values;
each product is exact in float32.

The weights carry no query scale (the MXFP4 query has its own): ``fp32(w) * fp32(softmax_scale * head_scale)``, which
``mqa_logits_mx_inputs`` casts to BF16 on SM100. Rounding the scaled weight to BF16 changes the relative weight of
the heads, so the trainer does the same.
"""

import torch

MXFP4_BLOCK = 32
# vLLM floors each block's absolute maximum at 6 * 2^-126, so the smallest scale is 2^-126
AMAX_FLOOR = 6.0 * 2.0**-126
# Triton's ``amax * (1.0 / 6.0)`` multiplies by the float32 constant, which is not the same as dividing by 6
ONE_SIXTH = torch.tensor(1.0 / 6.0, dtype=torch.float32).item()


def e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """``x`` (float32) rounded to the nearest E2M1 value (0, 0.5, 1, 1.5, 2, 3, 4, 6), ties to the even code, and
    saturated at 6, as ``cvt.rn.satfinite.e2m1x2`` does."""
    a = x.abs()
    # Interval edges; a tie goes to the neighbour with an even code (0, 1, 2, 4 have even codes; 0.5, 1.5, 3, 6 odd)
    out = torch.full_like(a, 6.0)
    out = torch.where(a <= 5.0, 4.0, out)
    out = torch.where(a < 3.5, 3.0, out)
    out = torch.where(a <= 2.5, 2.0, out)
    out = torch.where(a < 1.75, 1.5, out)
    out = torch.where(a <= 1.25, 1.0, out)
    out = torch.where(a < 0.75, 0.5, out)
    out = torch.where(a <= 0.25, 0.0, out)
    return torch.copysign(out, x)


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
    """The per-head weights vLLM's SM100 MXFP4 scoring takes: ``bf16(fp32(w) * fp32(head_dim^-0.5 *
    n_heads^-0.5))``."""
    scale = torch.tensor(head_dim**-0.5 * n_heads**-0.5, dtype=torch.float32)
    return (w.float() * scale.to(w.device)).to(torch.bfloat16)


__all__ = ["MXFP4_BLOCK", "e2m1_round", "mxfp4_indexer_weights", "mxfp4_quant", "mxfp4_round", "ue8m0_exponent"]
