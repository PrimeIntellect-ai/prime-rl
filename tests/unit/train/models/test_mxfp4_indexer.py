import math

import pytest
import torch

from prime_rl.trainer.models.kernels.mxfp4_indexer import (
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


def _reference_topk(q, k, w, ks, ke, topk):
    """Each query's picks by the float64 score of the rounded values, as a set per row."""
    qr, kr = mxfp4_round(q.float()).double(), mxfp4_round(k.float()).double()
    scores = (torch.relu(torch.einsum("shd,td->sht", qr, kr)) * w.double()[:, :, None]).sum(1)
    columns = torch.arange(k.shape[0])[None, :]
    return scores.masked_fill((columns < ks[:, None]) | (columns >= ke[:, None]), float("-inf"))


def test_fp8_indexer_mxfp4_picks_the_top_rounded_scores_per_document():
    """Two packed documents: each query picks the best-scoring keys of its own causal prefix, by the rounded values
    and the BF16 weights, and pads with the sentinel past the prefix."""
    from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer

    torch.manual_seed(3)
    doc_lens = (40, 56)
    S, H, D, topk = sum(doc_lens), 8, 128, 16
    q = torch.randn(S, H, D).bfloat16()
    k = torch.randn(S, D).bfloat16()
    w = mxfp4_indexer_weights(torch.randn(S, H).bfloat16(), D, H)
    starts = torch.tensor([0] * doc_lens[0] + [doc_lens[0]] * doc_lens[1], dtype=torch.int32)
    ke = torch.arange(1, S + 1, dtype=torch.int32)
    top_k = fp8_indexer(q, k, w, starts, ke, topk, mxfp4=True)

    scores = _reference_topk(q, k, w, starts, ke, topk)
    for row in range(S):
        picked = top_k[row][top_k[row] < S]
        n = min(topk, int(ke[row] - starts[row]))
        assert picked.numel() == n
        assert bool(((picked >= starts[row]) & (picked < ke[row])).all())
        # Every pick scores at least as high as the n-th best (ties at the boundary may go either way)
        assert scores[row, picked].min() >= scores[row].topk(n).values[-1]


def test_fp8_indexer_mxfp4_scores_decoded_values_not_fp8():
    """A key whose 32-channel blocks differ by 2^20 in scale: per-row FP8 flushes the small blocks to zero and loses
    the query's match, MXFP4 keeps each block's own scale."""
    from prime_rl.trainer.models.kernels.fp8_indexer import fp8_indexer

    D, H = 128, 1
    q = torch.zeros(2, H, D)
    q[:, 0, :32] = 2.0**20  # large block: the same on both keys
    q[:, 0, 32:64] = 2.0**40  # tiny-key block weighted up
    k = torch.zeros(2, D)
    k[:, :32] = 1.0
    k[1, 32:64] = 2.0**-20  # only key 1 has the small block
    q, k = q.bfloat16(), k.bfloat16()
    w = torch.ones(2, H).bfloat16()
    ks = torch.zeros(2, dtype=torch.int32)
    ke = torch.full((2,), 2, dtype=torch.int32)
    top_k = fp8_indexer(q, k, w, ks, ke, 1, mxfp4=True)
    assert top_k[:, 0].tolist() == [1, 1]


def test_deepseek_v4_indexer_follows_indexer_precision():
    from prime_rl.trainer.models.deepseek_v4 import DeepseekV4Config
    from prime_rl.trainer.models.deepseek_v4.attention import DeepseekV4Indexer
    from prime_rl.trainer.models.deepseek_v4.rotary import DeepseekV4RotaryEmbedding
    from tests.unit.train.models.test_deepseek_v4 import MODEL

    for precision, expected in (("fp8", False), ("mxfp4", True)):
        config = DeepseekV4Config(**{**MODEL, "indexer_precision": precision})
        assert DeepseekV4Indexer(config, DeepseekV4RotaryEmbedding(config)).mxfp4 is expected
