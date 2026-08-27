"""Numerics of the fused MoE kernel against the grouped-mm expert path it replaces.

The kernel is forward only: `_FusedMoE` runs it in forward and computes gradients by hand
(bf16) or by autograd through the reference (mxfp8), so both halves need checking against
the path a run would take with `model.moe_fused_kernel=false`.
"""

import importlib.util

import pytest
import torch

from prime_rl.trainer.models.layers.moe import (
    _FusedMoE,
    _load_fused_moe_kernel,
    _quantize_mxfp8,
    _run_experts_fused_reference,
)


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

# Measured on a B200 at the shapes below: bf16 forward and gradients all land at 4e-3..5.5e-3
# (two bf16 paths with different reduction orders), mxfp8 forward at 6.8e-2 — which is 1.14x
# the 5.9e-2 that quantizing the inputs costs by itself, so almost all of it is e4m3, not the
# kernel. These are backstops at ~3x and ~1.5x the measured values; the sharper checks are the
# fp32 ratios in the forward test, which need no constant at all. `pytest -s` prints margins.
BF16_TOL = 1.5e-2
MXFP8_TOL = 1e-1
# The mxfp8 backward re-runs the very reference this compares against, on unquantized
# weights, so anything above float32 round-off means the glue around it is wrong. In
# practice it is bitwise equal — the same kernels on the same inputs in the same order.
GLUE_TOL = 1e-5
# Quantizing to e4m3 costs accuracy no kernel can avoid, so the mxfp8 forward is judged
# against that floor rather than a number picked by hand.
MXFP8_FLOOR_RATIO = 1.5

GRAD_NAMES = ("x", "w1", "w2", "w3", "top_scores")


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative Frobenius error, the scale-free way to compare two low-precision GEMM paths."""
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, label: str) -> None:
    """Compare, and report the measured error either way — `pytest -s` prints the margins."""
    norm = expected.float().norm().item()
    # A relative error of zero means "identical"; against an all-zero reference it would
    # instead mean "both sides computed nothing", which must not read as a pass.
    assert norm > 0, f"{label}: reference is all zeros, the comparison would be vacuous"
    err = _rel_err(actual, expected)
    print(f"{label}: rel_err={err:.3e} tol={tol:.1e} margin={tol / max(err, 1e-12):.1f}x |ref|={norm:.3e}")
    assert err < tol, f"{label}: relative error {err:.3e} exceeds {tol:.1e}"


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


def _fp32_ground_truth(moe_inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Expert-by-expert MoE in fp32 — an independent yardstick both paths can be measured against."""
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs
    xf, w1f, w2f, w3f = x.float(), w1.float(), w2.float(), w3.float()
    out = torch.zeros_like(xf)
    for expert in range(NUM_EXPERTS):
        rows, slot = (selected_experts_indices == expert).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        tokens = xf[rows]
        hidden = torch.nn.functional.silu(tokens @ w1f[expert].T) * (tokens @ w3f[expert].T)
        out.index_add_(0, rows, (hidden @ w2f[expert].T) * top_scores[rows, slot].unsqueeze(-1).float())
    return out


def _mxfp8_round_trip(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize to e4m3 and back, to price what mxfp8 costs before any kernel runs."""
    block = _load_fused_moe_kernel().MXFP8_SCALE_BLOCK
    data, scales = _quantize_mxfp8(tensor, block)
    return data.float().unflatten(-1, (-1, block)).mul(scales.float().unsqueeze(-1)).flatten(-2)


@pytest.mark.parametrize("mxfp8", [False, True])
def test_fused_forward_matches_grouped_mm(moe_inputs, mxfp8: bool):
    """Pairwise against grouped-mm, and — the tolerance-free part — against fp32.

    A fixed tolerance only says "close enough"; the ratio against the grouped-mm path's own
    fp32 error says the kernel is no *less* accurate than what it replaces, which is the
    claim that actually matters and needs no magic number.
    """
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs
    label = "mxfp8" if mxfp8 else "bf16"

    out = _FusedMoE.apply(x, w1, w2, w3, selected_experts_indices, top_scores, NUM_EXPERTS, mxfp8)
    expected = _run_experts_fused_reference(x, w1, w2, w3, selected_experts_indices, top_scores, NUM_EXPERTS)

    truth = _fp32_ground_truth(moe_inputs)
    grouped_mm_err, fused_err = _rel_err(expected, truth), _rel_err(out, truth)
    print(f"forward[{label}] vs fp32: grouped_mm={grouped_mm_err:.3e} fused={fused_err:.3e}")

    assert out.shape == expected.shape and out.dtype == x.dtype
    _assert_close(out, expected, MXFP8_TOL if mxfp8 else BF16_TOL, f"forward[{label}]")

    if not mxfp8:
        assert fused_err <= 1.5 * grouped_mm_err, (
            f"fused bf16 is less accurate than grouped-mm: {fused_err:.3e} vs {grouped_mm_err:.3e} against fp32"
        )
        return

    # What e4m3 with one e8m0 scale per 32 elements costs on these inputs, kernel aside.
    floor_inputs = (_mxfp8_round_trip(x), *(_mxfp8_round_trip(w) for w in (w1, w2, w3)), *moe_inputs[4:])
    floor_err = _rel_err(_fp32_ground_truth(floor_inputs), truth)
    print(f"forward[mxfp8] vs fp32: quantization floor={floor_err:.3e} fused={fused_err:.3e}")
    assert fused_err <= MXFP8_FLOOR_RATIO * floor_err, (
        f"fused mxfp8 error {fused_err:.3e} exceeds {MXFP8_FLOOR_RATIO}x the {floor_err:.3e} "
        "that quantizing the inputs costs on its own — the kernel is losing accuracy beyond mxfp8"
    )


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

    for name, actual, reference in zip(GRAD_NAMES, fused, grouped_mm):
        _assert_close(actual.grad, reference.grad, BF16_TOL, f"grad_{name}[bf16]")


def test_fused_mxfp8_backward_matches_its_reference(moe_inputs):
    """The mxfp8 backward differentiates the reference, so this checks the glue around it.

    Same math on both sides, so the gradients must agree to round-off. What it catches is
    everything between: the padded group layout, the `needs_input_grad` bookkeeping, and the
    order gradients are returned in — `w1` and `w3` share a shape, so a swapped slot is
    invisible to a shape check and obvious here.
    """
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs
    fused = _leaves(x, w1, w2, w3, top_scores)
    reference = _leaves(x, w1, w2, w3, top_scores)

    out = _FusedMoE.apply(*fused[:4], selected_experts_indices, fused[4], NUM_EXPERTS, True)
    expected = _run_experts_fused_reference(
        *reference[:4],
        selected_experts_indices,
        reference[4],
        NUM_EXPERTS,
        align_m=_load_fused_moe_kernel().MXFP8_SCALE_BLOCK,
    )

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    expected.backward(grad_out)

    for name, actual, ref in zip(GRAD_NAMES, fused, reference):
        assert actual.grad is not None and torch.isfinite(actual.grad).all(), f"grad_{name} is missing or non-finite"
        _assert_close(actual.grad, ref.grad, GLUE_TOL, f"grad_{name}[mxfp8]")


@pytest.mark.parametrize("mxfp8", [False, True])
def test_fused_backward_honours_needs_input_grad(moe_inputs, mxfp8: bool):
    """Freezing some leaves must leave their slots None without shifting the others."""
    x, w1, w2, w3, selected_experts_indices, top_scores = moe_inputs
    wanted = (True, False, True, False, True)  # x, w2 and top_scores only: w1/w3 frozen
    make = lambda: [t.detach().clone().requires_grad_(r) for t, r in zip((x, w1, w2, w3, top_scores), wanted)]
    fused, reference = make(), make()

    out = _FusedMoE.apply(*fused[:4], selected_experts_indices, fused[4], NUM_EXPERTS, mxfp8)
    align_m = _load_fused_moe_kernel().MXFP8_SCALE_BLOCK if mxfp8 else None
    expected = _run_experts_fused_reference(
        *reference[:4], selected_experts_indices, reference[4], NUM_EXPERTS, align_m=align_m
    )

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    expected.backward(grad_out)

    tol = GLUE_TOL if mxfp8 else BF16_TOL
    for name, actual, ref, is_wanted in zip(GRAD_NAMES, fused, reference, wanted):
        if not is_wanted:
            assert actual.grad is None, f"grad_{name} was computed for a frozen leaf"
            continue
        _assert_close(actual.grad, ref.grad, tol, f"grad_{name}[{'mxfp8' if mxfp8 else 'bf16'}, partial]")
