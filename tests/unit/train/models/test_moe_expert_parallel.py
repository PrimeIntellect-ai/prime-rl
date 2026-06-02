"""Numerical validation of the local variable-arity ``_expert_parallel`` wrapper.

The wrapper vendors torchtitan's body but accepts ``*weights`` so gpt-oss's
4-tensor expert layout (gate_up_proj, gate_up_proj_bias, down_proj,
down_proj_bias) dispatches alongside the 3-weight (GroupedExperts) and
2+dummy (NonGatedGroupedExperts) paths. These tests assert the decorated fns
match an independent per-expert reference, including:

  - non-zero gpt-oss biases (init zeros them, so set them here),
  - token counts that are NOT a multiple of TOKEN_GROUP_ALIGN_SIZE_M=8 (so the
    generate_permute_indices alignment pad row is exercised),
  - the for-loop path (use_grouped_mm=False),
  - 3-weight + NonGated regressions (variable-arity must not perturb N=3 numerics).
"""

import pytest
import torch

from prime_rl.trainer.models.layers.moe import (
    GPT_OSS_ALPHA,
    GPT_OSS_LIMIT,
    _run_experts_grouped_mm,
    _run_gpt_oss_experts_for_loop,
    _run_gpt_oss_experts_grouped_mm,
    _run_nongated_experts_grouped_mm,
    relu2,
)
from prime_rl.utils.utils import default_dtype

pytestmark = [pytest.mark.gpu]

# Per-expert token counts: sum=23, and NO entry is a multiple of 8 -> forces the
# generate_permute_indices alignment pad (TOKEN_GROUP_ALIGN_SIZE_M=8) to fire.
NUM_TOKENS = [5, 7, 3, 8]


def _gpt_oss_reference(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, counts):
    """Independent dense per-expert gpt-oss forward (clamped sigmoid-glu + biases)."""
    splits = torch.split(x, counts, dim=0)
    outs = []
    for e, x_e in enumerate(splits):
        gate_up = x_e @ gate_up_proj[e] + gate_up_proj_bias[e]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=GPT_OSS_LIMIT)
        up = up.clamp(min=-GPT_OSS_LIMIT, max=GPT_OSS_LIMIT)
        glu = gate * torch.sigmoid(gate * GPT_OSS_ALPHA)
        h = (up + 1) * glu
        outs.append(h @ down_proj[e] + down_proj_bias[e])
    return torch.cat(outs, dim=0)


def _gated_reference(w1, w2, w3, x, counts):
    """Independent dense per-expert silu-glu forward (3-weight GroupedExperts)."""
    splits = torch.split(x, counts, dim=0)
    outs = []
    for e, x_e in enumerate(splits):
        h = torch.nn.functional.silu(x_e @ w1[e].transpose(-2, -1))
        h = h * (x_e @ w3[e].transpose(-2, -1))
        outs.append(h @ w2[e].transpose(-2, -1))
    return torch.cat(outs, dim=0)


def _nongated_reference(w1, w2, x, counts):
    """Independent dense per-expert relu^2 forward (NonGatedGroupedExperts)."""
    splits = torch.split(x, counts, dim=0)
    outs = []
    for e, x_e in enumerate(splits):
        h = relu2(x_e @ w1[e].transpose(-2, -1))
        outs.append(h @ w2[e].transpose(-2, -1))
    return torch.cat(outs, dim=0)


def _setup():
    E, H, I = 4, 64, 128
    counts = torch.tensor(NUM_TOKENS, dtype=torch.int64, device="cuda")
    total = int(counts.sum())
    with torch.device("cuda"), default_dtype(torch.float32):
        x = torch.randn(total, H)
    return E, H, I, counts, total, x


def test_gpt_oss_grouped_mm_matches_reference_with_biases_and_pad():
    E, H, I, counts, total, x = _setup()
    with torch.device("cuda"), default_dtype(torch.float32):
        gate_up_proj = torch.randn(E, H, 2 * I) * 0.02
        gate_up_proj_bias = torch.randn(E, 2 * I) * 0.1  # NON-ZERO: init zeros these
        down_proj = torch.randn(E, I, H) * 0.02
        down_proj_bias = torch.randn(E, H) * 0.1  # NON-ZERO

    out = _run_gpt_oss_experts_grouped_mm(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, counts)
    ref = _gpt_oss_reference(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, NUM_TOKENS)

    assert out.shape == (total, H)
    # grouped_mm runs in bf16 internally (see _impl); reference is fp32. bf16 tol.
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_gpt_oss_for_loop_matches_reference_with_biases_and_pad():
    E, H, I, counts, total, x = _setup()
    with torch.device("cuda"), default_dtype(torch.float32):
        gate_up_proj = torch.randn(E, H, 2 * I) * 0.02
        gate_up_proj_bias = torch.randn(E, 2 * I) * 0.1
        down_proj = torch.randn(E, I, H) * 0.02
        down_proj_bias = torch.randn(E, H) * 0.1

    out = _run_gpt_oss_experts_for_loop(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, counts)
    ref = _gpt_oss_reference(gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias, x, NUM_TOKENS)

    assert out.shape == (total, H)
    # The for-loop impl computes in fp32 (no bf16 cast); expect tight agreement.
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-4, atol=1e-4)


def test_three_weight_grouped_mm_regression():
    """Variable-arity wrapper must leave 3-weight numerics unchanged."""
    E, H, I, counts, total, x = _setup()
    with torch.device("cuda"), default_dtype(torch.float32):
        w1 = torch.randn(E, I, H) * 0.02
        w2 = torch.randn(E, H, I) * 0.02
        w3 = torch.randn(E, I, H) * 0.02

    out = _run_experts_grouped_mm(w1, w2, w3, x, counts)
    ref = _gated_reference(w1, w2, w3, x, NUM_TOKENS)

    assert out.shape == (total, H)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_nongated_grouped_mm_regression():
    """2+dummy-weight path must also be unperturbed by variable-arity dispatch."""
    E, H, I, counts, total, x = _setup()
    with torch.device("cuda"), default_dtype(torch.float32):
        w1 = torch.randn(E, I, H) * 0.02
        w2 = torch.randn(E, H, I) * 0.02
        w3_dummy = torch.empty(0, device="cuda")

    out = _run_nongated_experts_grouped_mm(w1, w2, w3_dummy, x, counts)
    ref = _nongated_reference(w1, w2, x, NUM_TOKENS)

    assert out.shape == (total, H)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
