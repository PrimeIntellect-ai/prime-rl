import pytest
import torch

from prime_rl.trainer.models.kernels.deepseek_v4.dsv41_nvfp4_kv_fake_quant import (
    dsv41_nvfp4_kv_fake_quant,
    e2m1_round,
    nvfp4_ds_mla_quant,
)

E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # codes 0..7, magnitude only


def reference_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 magnitude by brute force, ties to the even code, saturating at 6."""
    a = x.abs().clamp(max=6.0).unsqueeze(-1)
    distance = (a - E2M1_VALUES).abs()
    best = distance.min(dim=-1, keepdim=True).values
    candidates = distance == best
    codes = torch.arange(8).expand_as(candidates)
    # Among tied codes, the even one
    even = candidates & (codes % 2 == 0)
    code = torch.where(
        even.any(-1), torch.where(even, codes, 99).min(-1).values, torch.where(candidates, codes, 99).min(-1).values
    )
    return torch.copysign(E2M1_VALUES[code], x)


def reference_record(x: torch.Tensor) -> torch.Tensor:
    """vLLM's `_rope_quant_insert_nvfp4_kernel` arithmetic on one (..., 512) float32 row, decoded back."""
    tiles = x.float().reshape(*x.shape[:-1], -1, 16)
    amax = tiles.abs().amax(-1, keepdim=True)
    scale = torch.clamp(amax * torch.tensor(1.0 / 6.0), 2.0**-9, 448.0).to(torch.float8_e4m3fn).float()
    return (reference_e2m1(tiles / scale) * scale).reshape(x.shape)


def test_e2m1_rounds_to_nearest_even_and_saturates():
    ties_and_edges = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 7.5, 1e9, 0.0, 0.2499, 0.2501, -0.75, -5.0]
    )
    assert torch.equal(e2m1_round(ties_and_edges), reference_e2m1(ties_and_edges))
    dense = torch.linspace(-8, 8, 20001)
    assert torch.equal(e2m1_round(dense), reference_e2m1(dense))


def test_fake_quant_matches_the_cache_record_bit_for_bit():
    torch.manual_seed(0)
    rows = torch.randn(64, 512) * torch.logspace(-4, 3, 64).unsqueeze(1)
    rows[0] = 0.0  # all-zero tile: scale floors at 2^-9, values stay 0
    rows[1, :16] = 1e-6  # below the scale floor
    rows[2, :16] = 1e6  # scale saturates at 448, values at 6
    rows[3, :16] = torch.tensor([6.0, 3.0, 0.25 * 6 / 6] + [0.0] * 13)  # exact ties after the division
    x = rows.bfloat16()
    expected = reference_record(x).bfloat16()
    out = dsv41_nvfp4_kv_fake_quant(x)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, expected)
    assert torch.equal(out[0], torch.zeros(512, dtype=torch.bfloat16))
    assert out[2, :16].abs().max().item() == 6.0 * 448.0
    # E2M1 times an E4M3 scale is exact in BF16: the float32 record and the BF16 result agree
    values, scale = nvfp4_ds_mla_quant(x)
    assert torch.equal(
        (values * scale.float()).reshape(x.shape).bfloat16().float(), (values * scale.float()).reshape(x.shape)
    )


def test_fake_quant_has_a_straight_through_gradient():
    x = torch.randn(4, 512, dtype=torch.bfloat16, requires_grad=True)
    dsv41_nvfp4_kv_fake_quant(x).sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="vLLM's nvfp4_ds_mla writer needs SM100",
)
def test_bytes_match_vllm_nvfp4_ds_mla_writer():
    from vllm.models.deepseek_v41.common.ops.fused_compress_quant_cache import rope_quant_insert

    torch.manual_seed(1)
    tokens = 128
    latent = (torch.randn(tokens, 512, device="cuda") * 3).bfloat16()
    positions = torch.arange(tokens, device="cuda")
    # Identity RoPE (cos 1, sin 0): the record quantizes the latent as is
    cos_sin = torch.cat([torch.ones(tokens, 32), torch.zeros(tokens, 32)], dim=1).cuda()
    cache = torch.zeros(1, tokens, 288, dtype=torch.uint8, device="cuda")
    rope_quant_insert(latent, positions, cos_sin, cache, positions, compress_ratio=1)
    record = cache[0]
    nibbles = torch.stack([record[:, :256] & 15, record[:, :256] >> 4], dim=-1).reshape(tokens, 512).long()
    magnitude = E2M1_VALUES.cuda()[nibbles & 7]
    values = torch.where(nibbles >= 8, -magnitude, magnitude)
    scales = record[:, 256:288].view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
    decoded = (values * scales).bfloat16()
    assert torch.equal(decoded, dsv41_nvfp4_kv_fake_quant(latent))
