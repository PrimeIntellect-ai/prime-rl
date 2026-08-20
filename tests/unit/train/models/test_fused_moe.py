"""Numerics of the fused MoE kernel against the grouped-mm expert path it replaces.

The kernel is forward only: `_FusedMoE` runs it in forward and computes gradients by hand
(bf16) or by autograd through the reference (mxfp8), so both halves need checking against
the path a run would take with `model.moe_fused_kernel=false`.
"""

import importlib.util

import pytest
import torch

from prime_rl.trainer.models.layers.moe import _FusedMoE, _run_experts_fused_reference


def _unavailable_reason() -> str | None:
    if not torch.cuda.is_available():
        return "no CUDA device"
    if importlib.util.find_spec("prime_kernels") is None:
        return "prime-kernels is not installed (`uv sync --extra kernels`)"
    import prime_kernels

    return prime_kernels.unavailable_reason("flash_moe")


_UNAVAILABLE = _unavailable_reason()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_UNAVAILABLE is not None, reason=f"fused MoE kernel unavailable: {_UNAVAILABLE}"),
]

NUM_EXPERTS, TOP_K, DIM, HIDDEN_DIM, NUM_TOKENS = 4, 2, 256, 128, 512

# Both paths are bf16 GEMMs with fp32 accumulation but different reduction orders, so they
# agree to bf16 precision, not bitwise. mxfp8 additionally quantizes both operands to e4m3
# with one e8m0 scale per 32 elements, which costs roughly another order of magnitude.
BF16_TOL = 3e-2
MXFP8_TOL = 1.5e-1


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative Frobenius error, the scale-free way to compare two low-precision GEMM paths."""
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()


@pytest.fixture
def moe_inputs() -> tuple[torch.Tensor, ...]:
    """Routed-expert inputs shaped like the fused path expects: hidden_dim % 128, dim % 256."""
    torch.manual_seed(0)
    randn = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    x = randn(NUM_TOKENS, DIM) * 0.2
    w1, w3 = randn(NUM_EXPERTS, HIDDEN_DIM, DIM) * 0.05, randn(NUM_EXPERTS, HIDDEN_DIM, DIM) * 0.05
    w2 = randn(NUM_EXPERTS, DIM, HIDDEN_DIM) * 0.05
    top_scores, selected_experts_indices = torch.softmax(
        torch.randn(NUM_TOKENS, NUM_EXPERTS, device="cuda", dtype=torch.float32), dim=-1
    ).topk(TOP_K, dim=-1)
    return x, w1, w2, w3, selected_experts_indices, top_scores


def _leaves(*tensors: torch.Tensor) -> list[torch.Tensor]:
    return [t.detach().clone().requires_grad_(True) for t in tensors]


@pytest.mark.parametrize("mxfp8", [False, True])
def test_fused_forward_matches_grouped_mm(moe_inputs, mxfp8: bool):
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs

    out = _FusedMoE.apply(x, w1, w2, w3, selected_experts_indices, top_scores, NUM_EXPERTS, mxfp8)
    expected = _run_experts_fused_reference(x, w1, w2, w3, selected_experts_indices, top_scores, NUM_EXPERTS)

    assert out.shape == expected.shape and out.dtype == x.dtype
    assert _rel_err(out, expected) < (MXFP8_TOL if mxfp8 else BF16_TOL)


def test_fused_bf16_backward_matches_grouped_mm(moe_inputs):
    """`_run_experts_fused_backward_bf16` is closed form — check it against plain autograd."""
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs
    fused = _leaves(x, w1, w2, w3, top_scores)
    grouped_mm = _leaves(x, w1, w2, w3, top_scores)

    out = _FusedMoE.apply(*fused[:4], selected_experts_indices, fused[4], NUM_EXPERTS, False)
    expected = _run_experts_fused_reference(*grouped_mm[:4], selected_experts_indices, grouped_mm[4], NUM_EXPERTS)

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    expected.backward(grad_out)

    for name, actual, reference in zip(("x", "w1", "w2", "w3", "top_scores"), fused, grouped_mm):
        assert _rel_err(actual.grad, reference.grad) < BF16_TOL, f"grad_{name} disagrees with the grouped-mm path"


def test_fused_mxfp8_backward_populates_every_grad(moe_inputs):
    """The mxfp8 backward differentiates the reference, so only its wiring is worth asserting."""
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs
    leaves = _leaves(x, w1, w2, w3, top_scores)

    out = _FusedMoE.apply(*leaves[:4], selected_experts_indices, leaves[4], NUM_EXPERTS, True)
    out.backward(torch.randn_like(out))

    for name, leaf in zip(("x", "w1", "w2", "w3", "top_scores"), leaves):
        assert leaf.grad is not None, f"grad_{name} is None"
        assert leaf.grad.shape == leaf.shape
        assert torch.isfinite(leaf.grad).all(), f"grad_{name} is not finite"
