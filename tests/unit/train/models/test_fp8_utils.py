import pytest
import torch

from prime_rl.trainer.models.kernels.fp8_utils import (
    per_block_cast_to_fp8_tp_triton,
    per_block_cast_to_fp8_triton,
    per_token_cast_to_fp8_tp_triton,
    per_token_cast_to_fp8_triton,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="block-fp8 cast kernels use Triton fp8e4nv (e4m3), only supported on Hopper (SM90) and newer",
    ),
]


@pytest.mark.parametrize("rows,cols", [(256, 256), (256, 512), (512, 256), (1024, 768), (384, 128)])
def test_block_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is *bit-identical* to unfused."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_block_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_block_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)


@pytest.mark.parametrize("rows,cols", [(256, 512), (512, 256), (128, 1024), (1024, 768), (384, 512)])
def test_token_tp_cast_matches_materialized_transpose(rows, cols):
    """The fused transpose+cast is *bit-identical* to unfused."""
    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * 0.3

    ref_q, ref_s = per_token_cast_to_fp8_triton(x.transpose(0, 1).contiguous(), False)
    tp_q, tp_s = per_token_cast_to_fp8_tp_triton(x, False)

    assert tp_q.shape == ref_q.shape == (cols, rows)
    assert tp_s.shape == ref_s.shape
    assert tp_q.is_contiguous()
    assert torch.equal(tp_q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(tp_s, ref_s)


@pytest.mark.parametrize(
    "rows,cols,input_scale",
    [(256, 256, 1.0), (256, 512, 0.3), (512, 256, 0.02), (1024, 768, 0.02), (384, 128, 0.005)],
)
def test_token_cast_matches_vllm_cuda_quant(rows, cols, input_scale):
    """The per-token cast is *bit-identical* to vLLM's production CUDA quantizer."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8

    torch.manual_seed(rows + cols)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16) * input_scale
    assert x.is_contiguous()

    ref_q, ref_s = per_token_group_quant_fp8(x, 128, use_ue8m0=False)
    q, s = per_token_cast_to_fp8_triton(x, False)

    assert q.shape == ref_q.shape == (rows, cols)
    assert s.shape == ref_s.shape == (rows, cols // 128)
    assert torch.equal(s, ref_s)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


def _philox_uniform(seed, indices):
    import numpy as np

    mask = np.uint64(0xFFFFFFFF)
    indices = np.asarray(indices, dtype=np.uint64)
    a, b = indices & mask, indices >> np.uint64(32)
    c, d = np.zeros_like(a), np.zeros_like(a)
    k0, k1 = np.uint64(seed & 0xFFFFFFFF), np.uint64(seed >> 32)
    for _ in range(10):
        p, q = np.uint64(0xD2511F53) * a, np.uint64(0xCD9E8D57) * c
        a, b, c, d = (q >> np.uint64(32)) ^ b ^ k0, q & mask, (p >> np.uint64(32)) ^ d ^ k1, p & mask
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & mask, (k1 + np.uint64(0xBB67AE85)) & mask
    signed = a.astype(np.uint32).view(np.int32).astype(np.int64)
    return np.where(signed < 0, -signed - 1, signed).astype(np.float32) * np.float32(4.6566127342e-10)


@pytest.mark.parametrize("seed", [0, 872341, 2**62 + 17])
@pytest.mark.parametrize("offset", [0, 2**32 - 129])
def test_sr_all_finite_bf16_values(seed, offset):
    import numpy as np
    import triton
    import triton.language as tl

    from prime_rl.trainer.models.kernels.fp8_utils import _stochastic_round_fp8

    @triton.jit
    def cast(X, Y, n, seed, offset, B: tl.constexpr):
        i = tl.program_id(0) * B + tl.arange(0, B)
        x = tl.load(X + i, i < n, other=0).to(tl.float32)
        y = _stochastic_round_fp8(x, seed, i.to(tl.int64) + offset)
        tl.store(Y + i, y, i < n)

    values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
    values = values[torch.isfinite(values) & (values.abs() <= 448)]
    levels_tensor = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    fractions = torch.tensor([0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0])
    boundaries = (levels_tensor[:-1, None] + levels_tensor.diff()[:, None] * fractions).flatten()
    boundaries = torch.cat((boundaries, -boundaries))
    values = torch.cat((values, boundaries, boundaries.nextafter(torch.zeros_like(boundaries))))
    x = values.cuda()
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    cast[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), seed, offset, 256)
    levels = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()
    magnitude = values.abs().numpy()
    upper_index = np.searchsorted(levels, magnitude, side="left").clip(1, len(levels) - 1)
    lower, upper = levels[upper_index - 1], levels[upper_index]
    probability = (magnitude - lower) / (upper - lower)
    uniform = _philox_uniform(seed, np.arange(len(values), dtype=np.uint64) + np.uint64(offset))
    expected = np.copysign(np.where(uniform < probability, upper, lower), values.numpy())
    expected_bytes = torch.from_numpy(expected).to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(out.view(torch.uint8).cpu(), expected_bytes)


@pytest.mark.parametrize("use_ue8m0", [False, True])
@pytest.mark.parametrize("shape", [(128, 256), (129, 255), (256, 512)])
def test_sr_transpose_replay_and_expert_shards(use_ue8m0, shape):
    from prime_rl.trainer.models.kernels.fp8_utils import grouped_per_block_cast_to_fp8_triton

    torch.manual_seed(17)
    weights = torch.randn(4, *shape, device="cuda", dtype=torch.bfloat16)
    seed = torch.tensor(1123581321, dtype=torch.int64, device="cuda")
    kwargs = dict(rounding_seed=seed, rounding_salt=987654321)
    q, s = grouped_per_block_cast_to_fp8_triton(weights, use_ue8m0, **kwargs)
    _, rne_s = grouped_per_block_cast_to_fp8_triton(weights, use_ue8m0)
    assert torch.equal(s, rne_s)
    qt, st = grouped_per_block_cast_to_fp8_triton(weights.transpose(1, 2), use_ue8m0, rng_transpose=True, **kwargs)
    assert torch.equal(q.view(torch.uint8).transpose(1, 2), qt.view(torch.uint8))
    assert torch.equal(s.transpose(1, 2), st)
    for start in range(4):
        qs, ss = grouped_per_block_cast_to_fp8_triton(
            weights[start : start + 1], use_ue8m0, rounding_offset=start * shape[0] * shape[1], **kwargs
        )
        assert torch.equal(q[start : start + 1].view(torch.uint8), qs.view(torch.uint8))
        assert torch.equal(s[start : start + 1], ss)
    qd, sd = per_block_cast_to_fp8_triton(weights[0], use_ue8m0, **kwargs)
    qdt, sdt = per_block_cast_to_fp8_tp_triton(weights[0], use_ue8m0, **kwargs)
    assert torch.equal(qd.view(torch.uint8).T, qdt.view(torch.uint8))
    assert torch.equal(sd.T, sdt)
    torch.rand(1000, device="cuda")
    replay, _ = grouped_per_block_cast_to_fp8_triton(weights, use_ue8m0, **kwargs)
    assert torch.equal(q.view(torch.uint8), replay.view(torch.uint8))
    changed, _ = grouped_per_block_cast_to_fp8_triton(
        weights, use_ue8m0, rounding_seed=seed + 1, rounding_salt=987654321
    )
    assert not torch.equal(q.view(torch.uint8), changed.view(torch.uint8))


def test_sr_rounding_probabilities():
    from prime_rl.trainer.models.kernels.fp8_utils import per_block_cast_to_fp8_triton

    # The 448 sentinel fixes scale=1; 1.03125 lies one quarter of the way from 1 to 1.125.
    x = torch.full((2048, 1024), 1.03125, device="cuda", dtype=torch.bfloat16)
    x[::128, ::128] = 448
    q, _ = per_block_cast_to_fp8_triton(x, False, rounding_seed=torch.tensor(123, device="cuda"))
    sample = q.float()[x != 448]
    frequency = (sample == 1.125).float().mean().item()
    assert abs(frequency - 0.25) < 0.002
    assert set(sample.unique().tolist()) == {1.0, 1.125}


@pytest.mark.parametrize("stochastic", [False, True])
def test_block_cast_large_expert_stride(stochastic):
    from prime_rl.trainer.models.kernels.fp8_utils import grouped_per_block_cast_to_fp8_triton

    # The last expert starts beyond the signed 32-bit element-offset limit.
    x = torch.empty_strided((3, 128, 128), (2**30, 128, 1), dtype=torch.bfloat16, device="cuda")
    x.copy_(torch.randn(x.shape, device="cuda", dtype=x.dtype))
    seed = torch.tensor(17, device="cuda") if stochastic else None
    q, s = grouped_per_block_cast_to_fp8_triton(x, False, rounding_seed=seed)
    qr, sr = grouped_per_block_cast_to_fp8_triton(x.contiguous(), False, rounding_seed=seed)
    assert torch.equal(q.view(torch.uint8), qr.view(torch.uint8))
    assert torch.equal(s, sr)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_sr_preserves_nonfinite_weights(value):
    x = torch.ones(128, 128, dtype=torch.bfloat16, device="cuda")
    x[0, 0] = value
    q, s = per_block_cast_to_fp8_triton(x, False, rounding_seed=torch.tensor(5, device="cuda"))
    qr, sr = per_block_cast_to_fp8_triton(x, False)
    assert torch.equal(q.float().isnan(), qr.float().isnan())
    assert q.float()[0, 0].isnan()
    assert torch.equal(s, sr)
