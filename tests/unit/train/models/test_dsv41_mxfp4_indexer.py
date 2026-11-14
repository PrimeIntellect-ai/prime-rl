import math

import pytest
import torch

from prime_rl.trainer.models.kernels.deepseek_v4.dsv41_mxfp4_indexer import (
    mxfp4_indexer_weights,
    mxfp4_quant,
    mxfp4_round,
    ue8m0_exponent,
)

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # codes 0..7, magnitude only


def reference_e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """4-bit E2M1 code of the nearest value by brute force, ties to the even code, saturating at 6."""
    a = x.abs().clamp(max=6.0).unsqueeze(-1)
    distance = (a - E2M1_VALUES).abs()
    candidates = distance == distance.min(dim=-1, keepdim=True).values
    codes = torch.arange(8).expand_as(candidates)
    even = candidates & (codes % 2 == 0)
    code = torch.where(
        even.any(-1), torch.where(even, codes, 99).min(-1).values, torch.where(candidates, codes, 99).min(-1).values
    )
    # A negative value that rounds to zero keeps its sign bit, as the hardware cast does
    return code | (torch.signbit(x).int() << 3)


def decode_e2m1(code: torch.Tensor) -> torch.Tensor:
    magnitude = E2M1_VALUES.to(code.device)[(code & 7).long()]
    return torch.where((code & 8) != 0, -magnitude, magnitude)


def reference_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's MXFP4 indexer record for (..., D) rows, element by element: the E2M1 codes (one per value, before
    nibble packing) and the UE8M0 bytes (one per 32 values)."""
    blocks = x.float().reshape(-1, 32)
    codes, ue8m0 = [], []
    for block in blocks:
        amax = max(block.abs().max().item(), 6.0 * 2.0**-126)
        ratio = (torch.tensor(amax, dtype=torch.float32) * torch.tensor(1.0 / 6.0, dtype=torch.float32)).item()
        exponent = min(max(math.ceil(math.log2(ratio)), -127), 127)
        codes.append(reference_e2m1_code(block * torch.tensor(2.0**-exponent, dtype=torch.float32)))
        ue8m0.append(exponent + 127)
    codes = torch.stack(codes).reshape(x.shape)
    return codes, torch.tensor(ue8m0, dtype=torch.uint8).reshape(*x.shape[:-1], -1)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Two codes per byte, the even element in the low nibble (``_fp32x2_to_fp4x2(even, odd)``)."""
    codes = codes.int()
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def indexer_rows() -> torch.Tensor:
    torch.manual_seed(0)
    rows = torch.randn(48, 128) * torch.logspace(-30, 30, 48).unsqueeze(1)
    rows[0] = 0.0  # all-zero blocks: exponent floors at -126, values stay 0
    rows[1, :32] = 6.0 * 2.0**-3  # amax / 6 is a power of two: the scale is exactly it, nothing saturates
    rows[2, :32] = torch.tensor([6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25] * 4)  # every E2M1 tie at scale 1
    rows[3, :32] = 1e-38  # below the amax floor
    rows[4, :32] = -0.01
    rows[4, 0] = -6.0  # small negatives round to -0
    return rows.bfloat16()


def test_ue8m0_exponent_is_an_exact_ceil_log2():
    amax = torch.cat(
        [
            torch.tensor([0.0, 6.0 * 2.0**-126, 6.0, 12.0, 3.0, 1e-30, 3e38]),
            torch.logspace(-37, 38, 4001),
            # just above and below powers of two after the multiplication by fp32(1 / 6)
            6.0 * torch.exp2(torch.arange(-100, 100).float()) * (1 + 2.0**-20),
            6.0 * torch.exp2(torch.arange(-100, 100).float()) * (1 - 2.0**-20),
        ]
    )
    expected = []
    for a in amax.tolist():
        ratio = (
            torch.tensor(max(a, 6.0 * 2.0**-126), dtype=torch.float32) * torch.tensor(1 / 6, dtype=torch.float32)
        ).item()
        expected.append(math.ceil(math.log2(ratio)))
    assert ue8m0_exponent(amax).tolist() == expected


def test_mxfp4_round_matches_the_cache_record_bit_for_bit():
    x = indexer_rows()
    codes, ue8m0 = reference_mxfp4(x)
    values, exponent = mxfp4_quant(x)
    assert torch.equal((exponent.squeeze(-1) + 127).to(torch.uint8), ue8m0)
    expected_values = decode_e2m1(codes)
    assert torch.equal(values.reshape(x.shape), expected_values)
    # Decoded back to BF16 exactly: the float32 decode and the BF16 result agree
    decoded = (expected_values.reshape(*x.shape[:-1], -1, 32) * torch.exp2(exponent.float())).reshape(x.shape)
    out = mxfp4_round(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out.float(), decoded)
    assert torch.equal(out[0], torch.zeros(128, dtype=torch.bfloat16))
    assert torch.equal(out[1, :32], x[1, :32])


def test_mxfp4_round_quantizes_each_32_channel_block_on_its_own():
    x = torch.zeros(2, 128, dtype=torch.bfloat16)
    x[0, :32] = 1e6
    x[0, 32:64] = 0.75
    x[1, 32:64] = 0.75  # same block content, different neighbours: same result
    out = mxfp4_round(x)
    assert torch.equal(out[0, 32:64], out[1, 32:64])
    assert torch.equal(out[0, 32:64], x[0, 32:64])


def test_mxfp4_round_keeps_queries_per_head():
    torch.manual_seed(1)
    q = (torch.randn(5, 4, 128) * torch.logspace(-3, 3, 4).view(1, 4, 1)).bfloat16()
    assert torch.equal(mxfp4_round(q), mxfp4_round(q.reshape(-1, 128)).reshape(q.shape))


@pytest.mark.parametrize("n_heads", [32, 64])
def test_mxfp4_indexer_weights_round_like_the_sm100_kernels(n_heads):
    torch.manual_seed(2)
    w = torch.randn(7, n_heads).bfloat16()
    scale = torch.tensor(128**-0.5 * n_heads**-0.5, dtype=torch.float32)
    expected = (w.float() * scale).bfloat16()
    out = mxfp4_indexer_weights(w, 128, n_heads)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the indexer's scoring kernel is Triton")
def test_mxfp4_index_topk_scores_the_rounded_values():
    from prime_rl.trainer.models.kernels.dsv41_indexer import dsv41_index_topk

    torch.manual_seed(3)
    S, H, D, topk = 96, 8, 128, 16
    q = torch.randn(S, H, D, device="cuda").bfloat16()
    k = torch.randn(S, D, device="cuda").bfloat16()
    w = mxfp4_indexer_weights(torch.randn(S, H, device="cuda").bfloat16(), D, H)
    ks = torch.zeros(S, dtype=torch.int32, device="cuda")
    ke = torch.arange(1, S + 1, dtype=torch.int32, device="cuda")
    top_k, _ = dsv41_index_topk(q, k, w, ks, ke, topk, None, False, S, 4, 4, 1e-4, True)

    qr, kr = mxfp4_round(q).double(), mxfp4_round(k).double()
    scores = (torch.relu(torch.einsum("shd,td->sht", qr, kr)) * w.double()[:, :, None]).sum(1)
    scores = scores.masked_fill(torch.arange(S, device="cuda")[None, :] >= ke[:, None], float("-inf"))
    for row in range(S):
        picked = top_k[row][top_k[row] >= 0]
        n = min(topk, row + 1)
        assert picked.numel() == n
        # Every pick scores at least as high as the n-th best (ties at the boundary may go either way)
        threshold = scores[row].topk(n).values[-1]
        assert bool((scores[row, picked] >= threshold).all())


def sm100() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


@pytest.mark.skipif(not sm100(), reason="vLLM's MXFP4 indexer cache is SM100-only")
def test_keys_match_vllm_indexer_k_store_bytes():
    from vllm.models.deepseek_v41.common.ops import indexer_k_norm_rope_store

    torch.manual_seed(4)
    T = 64
    # Rows of +-1 have unit RMS, so k_norm returns bf16(+-weight) exactly; position 0 with cos 1, sin 0 is identity
    k_pre = torch.where(torch.rand(T, 128) < 0.5, -1.0, 1.0).bfloat16().cuda()
    weight = (torch.randn(128) * torch.logspace(-4, 4, 128)).bfloat16().cuda()
    cos_sin_cache = torch.cat([torch.ones(1, 32), torch.zeros(1, 32)], dim=1).cuda()
    k_cache = torch.zeros(T, 1, 68, dtype=torch.uint8, device="cuda")
    indexer_k_norm_rope_store(
        k_pre,
        torch.zeros(T, dtype=torch.int64, device="cuda"),
        cos_sin_cache,
        weight,
        1e-20,
        k_cache,
        torch.arange(T, device="cuda"),
        1,
        True,
    )
    k = (k_pre.float() * weight.float()).bfloat16()
    codes, ue8m0 = reference_mxfp4(k.cpu())
    assert torch.equal(k_cache[:, 0, :64].cpu(), pack_nibbles(codes))
    assert torch.equal(k_cache[:, 0, 64:].cpu(), ue8m0)
    values, exponent = mxfp4_quant(k.cpu())
    assert torch.equal((exponent.squeeze(-1) + 127).to(torch.uint8), ue8m0)


@pytest.mark.skipif(not sm100(), reason="vLLM's MXFP4 indexer queries are SM100-only")
def test_queries_and_weights_match_vllm_fused_indexer_q_bytes():
    from vllm.models.deepseek_v41.common.ops import fused_indexer_q_rope_quant

    torch.manual_seed(5)
    T, H = 16, 64
    q = (torch.randn(T, H, 128) * torch.logspace(-6, 6, H).view(1, H, 1)).bfloat16().cuda()
    w = torch.randn(T, H).bfloat16().cuda()
    cos_sin_cache = torch.cat([torch.ones(1, 32), torch.zeros(1, 32)], dim=1).cuda()
    (q_packed, q_scale), w_out = fused_indexer_q_rope_quant(
        torch.zeros(T, dtype=torch.int64, device="cuda"),
        q,
        cos_sin_cache,
        w,
        128**-0.5,
        H**-0.5,
        use_fp4=True,
        weights_out_dtype=torch.bfloat16,
    )
    codes, ue8m0 = reference_mxfp4(q.cpu())
    assert torch.equal(q_packed.cpu(), pack_nibbles(codes))
    assert torch.equal(q_scale.contiguous().view(torch.uint8).reshape(T, H, 4).cpu(), ue8m0)
    assert torch.equal(w_out.cpu(), mxfp4_indexer_weights(w, 128, H).cpu())
