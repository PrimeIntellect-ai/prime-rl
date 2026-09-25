"""Numerics of the FP8 / MXFP8 GEMM paths against the BF16 path they replace.

Both backends are *supposed* to be less accurate than BF16 — quantizing to e4m3 costs
accuracy no kernel can avoid. So a hand-picked tolerance would either be loose enough to
hide a wrong scale or tight enough to flake on a new kernel version. Instead every check
compares against the **quantization floor**: the same operands rounded to e4m3 with the
same block layout, multiplied in fp32. Anything the kernel adds on top of that floor is
the kernel's own error, and that ratio is what these tests bound.

The floor is computed in plain torch here, deliberately duplicating the recipe rather than
calling the Triton casts, so a bug in the casts cannot cancel itself out of the comparison.

`pytest -s` prints every measured error and its margin.
"""

import pytest
import torch
from torch import Tensor, nn

from prime_rl.trainer.models.layers.moe import GroupedExperts

_CAPABILITY = torch.cuda.get_device_capability() if torch.cuda.is_available() else None

# DeepGEMM ships kernels for Hopper (SM90) and Blackwell datacenter (SM100) only. On
# other GPUs — SM120 consumer Blackwell included — it aborts with "Unknown recipe".
FP8_UNAVAILABLE = (
    None
    if _CAPABILITY is not None and _CAPABILITY[0] in (9, 10)
    else f"FP8 blockwise needs DeepGEMM (SM90 or SM100), got {_CAPABILITY}"
)
MXFP8_UNAVAILABLE = (
    None if _CAPABILITY is not None and _CAPABILITY >= (10, 0) else f"MXFP8 needs SM100+, got {_CAPABILITY}"
)
# torchao's MoE grouped-GEMM kernels are built for SM100/SM100a only. The dense MXFP8
# linear path runs on consumer Blackwell (SM120) too, so the two guards differ.
MXFP8_GROUPED_UNAVAILABLE = (
    None
    if _CAPABILITY is not None and _CAPABILITY[0] == 10
    else f"torchao MXFP8 grouped GEMM needs SM100, got {_CAPABILITY}"
)

requires_fp8 = pytest.mark.skipif(FP8_UNAVAILABLE is not None, reason=str(FP8_UNAVAILABLE))
requires_mxfp8 = pytest.mark.skipif(MXFP8_UNAVAILABLE is not None, reason=str(MXFP8_UNAVAILABLE))
requires_mxfp8_grouped = pytest.mark.skipif(
    MXFP8_GROUPED_UNAVAILABLE is not None, reason=str(MXFP8_GROUPED_UNAVAILABLE)
)

pytestmark = [pytest.mark.gpu]

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP8_BLOCK = 128
MX_BLOCK = 32

# How much error the kernel may add on top of the floor. A correct kernel differs from the
# floor only by accumulation order (fp32 tensor-core reduction vs one fp32 matmul), which
# lands just above 1.0; a swapped scale or a transposed operand lands an order of magnitude
# out. 2x leaves room for the layout details the floor does not model.
FLOOR_RATIO = 2.0


def _rel_err(actual: Tensor, expected: Tensor) -> float:
    """Relative Frobenius error — the scale-free way to compare two GEMM paths."""
    expected = expected.float()
    return ((actual.float() - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _qdq(t: Tensor, block: tuple[int, int], power_of_two_scale: bool) -> Tensor:
    """Round `t` to e4m3 with one scale per `block`, then back to fp32.

    This is the accuracy the arithmetic can reach at best — the floor every kernel below is
    measured against.
    """
    rows, cols = t.shape
    block_rows, block_cols = block
    assert rows % block_rows == 0 and cols % block_cols == 0, f"{t.shape} not tileable by {block}"

    blocks = t.float().reshape(rows // block_rows, block_rows, cols // block_cols, block_cols)
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12) / FP8_MAX
    if power_of_two_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (quantized.float() * scale).reshape(rows, cols)


def _assert_near_floor(actual: Tensor, reference: Tensor, floor: Tensor, label: str) -> None:
    """The kernel may be no worse than `FLOOR_RATIO` times the cost of quantizing alone."""
    kernel_err = _rel_err(actual, reference)
    floor_err = _rel_err(floor, reference)
    assert floor_err > 0, f"{label}: floor is exact, the comparison would be vacuous"
    # A kernel that silently fell back to BF16 would score a perfect zero here, which must
    # read as a failure, not as the best possible result.
    assert kernel_err > 0, f"{label}: output is bit-identical to BF16 — the quantized path did not run"
    ratio = kernel_err / floor_err
    print(f"{label}: kernel={kernel_err:.3e} floor={floor_err:.3e} ratio={ratio:.2f}x (limit {FLOOR_RATIO:.1f}x)")
    assert ratio < FLOOR_RATIO, f"{label}: {kernel_err:.3e} is {ratio:.2f}x the quantization floor {floor_err:.3e}"


def _linear_inputs(m: int, k: int, n: int, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]:
    torch.manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.3
    return randn(m, k), randn(n, k), randn(m, n)


def _run_linear(module: nn.Module, x: Tensor, grad_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Forward and backward through `module`, returning (out, dx, dw)."""
    x = x.clone().requires_grad_(True)
    out = module(x)
    # Note the explicit grad tensor: `out.sum().backward()` feeds an *expanded* ones tensor,
    # and torchao's MXFP8 backward asserts on non-contiguous grad_output.
    out.backward(grad_out)
    return out, x.grad, module.weight.grad


@requires_fp8
@pytest.mark.parametrize("m,k,n", [(512, 512, 256), (256, 1024, 512), (384, 256, 128)])
def test_fp8_linear_matches_quantization_floor(m, k, n):
    """FP8 blockwise linear: 1x128 scales on activations, 128x128 on weights."""
    from prime_rl.trainer.models.kernels.fp8_utils import ue8m0_for_device
    from prime_rl.trainer.models.layers.fp8_linear import Float8BlockwiseLinear

    x, w, grad_out = _linear_inputs(m, k, n)
    reference = nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(w)
    quantized = Float8BlockwiseLinear.from_linear(nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16))
    with torch.no_grad():
        quantized.weight.copy_(w)

    ref_out, ref_dx, ref_dw = _run_linear(reference, x, grad_out)
    out, dx, dw = _run_linear(quantized, x, grad_out)

    ue8m0 = ue8m0_for_device(x.device)
    x_q = _qdq(x, (1, FP8_BLOCK), ue8m0)
    w_q = _qdq(w, (FP8_BLOCK, FP8_BLOCK), ue8m0)
    grad_q = _qdq(grad_out, (1, FP8_BLOCK), ue8m0)

    _assert_near_floor(out, ref_out, x_q @ w_q.T, f"fp8 fwd {m}x{k}x{n}")
    _assert_near_floor(dx, ref_dx, grad_q @ w_q, f"fp8 dx {m}x{k}x{n}")
    # The wgrad reduces over tokens, so both operands are quantized along M (1x128 blocks
    # of the transposed tensors) — the axis the forward does not exercise.
    _assert_near_floor(
        dw,
        ref_dw,
        _qdq(grad_out.T.contiguous(), (1, FP8_BLOCK), ue8m0) @ _qdq(x.T.contiguous(), (1, FP8_BLOCK), ue8m0).T,
        f"fp8 dw {m}x{k}x{n}",
    )


@requires_fp8
@pytest.mark.parametrize(
    "tokens_per_expert",
    [
        pytest.param([128, 128, 128, 128], id="aligned"),
        # Sequence packing hands the experts group sizes that are not multiples of the
        # 128-token GEMM alignment; the padding and unpack path is where they break.
        pytest.param([112, 144, 96, 160], id="unaligned"),
        pytest.param([0, 256, 128, 128], id="empty-expert"),
    ],
)
def test_fp8_grouped_gemm_matches_grouped_mm(tokens_per_expert):
    """The FP8 grouped GEMM is a drop-in for `torch._grouped_mm` over expert weights."""
    from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm

    torch.manual_seed(0)
    num_experts, k, n = len(tokens_per_expert), 256, 128
    total_m = sum(tokens_per_expert)
    x = torch.randn(total_m, k, device="cuda", dtype=torch.bfloat16) * 0.3
    w = torch.randn(num_experts, k, n, device="cuda", dtype=torch.bfloat16) * 0.3
    grad_out = torch.randn(total_m, n, device="cuda", dtype=torch.bfloat16) * 0.3
    offs = torch.tensor(tokens_per_expert, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)

    def run(fn) -> tuple[Tensor, Tensor, Tensor]:
        xg, wg = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        out = fn(xg, wg, offs)
        out.backward(grad_out)
        return out, xg.grad, wg.grad

    ref_out, ref_dx, ref_dw = run(lambda xg, wg, o: torch._grouped_mm(xg, wg, offs=o))
    out, dx, dw = run(grouped_fp8_gemm)

    # Per-expert floors, concatenated back into the grouped layout.
    ue8m0 = True  # both supported architectures use UE8M0 scales in the grouped kernels
    starts = [0] + torch.tensor(tokens_per_expert).cumsum(0).tolist()
    floor_out, floor_dx, floor_dw = [], [], []
    for expert, (start, end) in enumerate(zip(starts[:-1], starts[1:])):
        x_e, g_e = x[start:end], grad_out[start:end]
        rows = end - start
        if rows == 0:
            floor_dw.append(torch.zeros_like(w[expert], dtype=torch.float32))
            continue
        pad = (-rows) % FP8_BLOCK  # the kernel zero-pads the token axis before the wgrad
        x_p = torch.nn.functional.pad(x_e, (0, 0, 0, pad))
        g_p = torch.nn.functional.pad(g_e, (0, 0, 0, pad))
        w_q = _qdq(w[expert], (FP8_BLOCK, FP8_BLOCK), ue8m0)
        floor_out.append(_qdq(x_e, (1, FP8_BLOCK), ue8m0) @ w_q)
        floor_dx.append(_qdq(g_e, (1, FP8_BLOCK), ue8m0) @ w_q.T)
        floor_dw.append(
            _qdq(x_p.T.contiguous(), (1, FP8_BLOCK), ue8m0) @ _qdq(g_p.T.contiguous(), (1, FP8_BLOCK), ue8m0).T
        )

    label = "-".join(str(t) for t in tokens_per_expert)
    _assert_near_floor(out, ref_out, torch.cat(floor_out), f"fp8 grouped fwd [{label}]")
    _assert_near_floor(dx, ref_dx, torch.cat(floor_dx), f"fp8 grouped dx [{label}]")
    _assert_near_floor(dw, ref_dw, torch.stack(floor_dw), f"fp8 grouped dw [{label}]")


@requires_mxfp8
@pytest.mark.parametrize("recipe", ["mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp"])
@pytest.mark.parametrize("m,k,n", [(512, 512, 256), (256, 1024, 512), (128, 256, 128)])
def test_mxfp8_linear_matches_quantization_floor(recipe, m, k, n):
    """MXFP8 linear: one power-of-two scale per 32 elements along the reduction axis."""
    from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear

    x, w, grad_out = _linear_inputs(m, k, n)
    reference = nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        reference.weight.copy_(w)
    quantized = MXFP8Linear.from_linear(
        nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16),
        wgrad_with_hp=recipe == "mxfp8_rceil_wgrad_with_hp",
    )
    with torch.no_grad():
        quantized.weight.copy_(w)

    ref_out, ref_dx, ref_dw = _run_linear(reference, x, grad_out)
    out, dx, dw = _run_linear(quantized, x, grad_out)

    # Every operand is quantized along the axis it is reduced over.
    _assert_near_floor(
        out, ref_out, _qdq(x, (1, MX_BLOCK), True) @ _qdq(w, (1, MX_BLOCK), True).T, f"mxfp8/{recipe} fwd {m}x{k}x{n}"
    )
    _assert_near_floor(
        dx,
        ref_dx,
        _qdq(grad_out, (1, MX_BLOCK), True) @ _qdq(w.T.contiguous(), (1, MX_BLOCK), True).T,
        f"mxfp8/{recipe} dx {m}x{k}x{n}",
    )
    if recipe == "mxfp8_rceil_wgrad_with_hp":
        # The point of this recipe is that the weight gradient skips quantization entirely.
        wgrad_err = _rel_err(dw, ref_dw)
        print(f"mxfp8/{recipe} dw {m}x{k}x{n}: rel_err={wgrad_err:.3e} (high-precision wgrad)")
        assert wgrad_err < _rel_err(_qdq(grad_out.T.contiguous(), (1, MX_BLOCK), True), grad_out.T.contiguous()), (
            "wgrad_with_hp should be more accurate than an MXFP8-quantized wgrad"
        )
    else:
        _assert_near_floor(
            dw,
            ref_dw,
            _qdq(grad_out.T.contiguous(), (1, MX_BLOCK), True) @ _qdq(x.T.contiguous(), (1, MX_BLOCK), True).T,
            f"mxfp8/{recipe} dw {m}x{k}x{n}",
        )


@requires_mxfp8
def test_mxfp8_linear_applies_bias():
    """`from_linear` carries the bias over, and the forward must actually add it."""
    from prime_rl.trainer.models.layers.mxfp8_linear import MXFP8Linear

    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    linear = nn.Linear(128, 64, bias=True, device="cuda", dtype=torch.bfloat16)
    quantized = MXFP8Linear.from_linear(linear)

    assert quantized.bias is linear.bias
    with torch.no_grad():
        # A bias far above the matmul's own scale, so the shift it produces is unambiguous
        # in bfloat16 — subtracting a small bias back out would drown in rounding.
        quantized.bias.fill_(100.0)
        biased = quantized(x)
        quantized.bias.zero_()
        shift = (biased - quantized(x)).float()

    torch.testing.assert_close(shift, torch.full_like(shift, 100.0), rtol=2e-2, atol=0.0)


@requires_mxfp8_grouped
@pytest.mark.parametrize("recipe", ["mxfp8_rceil", "mxfp8_rceil_wgrad_with_hp"])
def test_mxfp8_grouped_experts_match_bf16(recipe):
    """`apply_mxfp8_moe_grouped_gemm` swaps the expert GEMMs, not the routing around them."""
    from prime_rl.trainer.models.layers.mxfp8_grouped_gemm import apply_mxfp8_moe_grouped_gemm

    torch.manual_seed(0)
    dim, hidden_dim, num_experts = 256, 128, 4
    tokens_per_expert = torch.tensor([64, 96, 32, 64], device="cuda", dtype=torch.int32)
    x = torch.randn(int(tokens_per_expert.sum()), dim, device="cuda", dtype=torch.bfloat16) * 0.3
    grad_out = torch.randn_like(x) * 0.3

    def build() -> GroupedExperts:
        torch.manual_seed(1)
        experts = GroupedExperts(dim, hidden_dim, num_experts, use_grouped_mm=True).to("cuda", torch.bfloat16)
        experts.init_weights(0.02)
        return experts.to(torch.bfloat16)

    def run(experts: GroupedExperts) -> tuple[Tensor, Tensor, Tensor]:
        xg = x.clone().requires_grad_(True)
        out = experts(xg, tokens_per_expert)
        out.backward(grad_out)
        return out, xg.grad, experts.w1.grad

    reference = build()
    quantized = build()
    apply_mxfp8_moe_grouped_gemm(quantized, recipe)

    ref_out, ref_dx, ref_dw1 = run(reference)
    out, dx, dw1 = run(quantized)

    for label, actual, expected in (("fwd", out, ref_out), ("dx", dx, ref_dx), ("dw1", dw1, ref_dw1)):
        err = _rel_err(actual, expected)
        print(f"mxfp8/{recipe} experts {label}: rel_err={err:.3e}")
        assert err > 0, f"experts {label}: bit-identical to BF16 — the MXFP8 grouped GEMM did not run"
        # Three chained quantized GEMMs (w1, w3, then w2), so the per-GEMM ~4e-2 floor
        # compounds; this bounds the whole expert stack, not one matmul.
        assert err < 0.2, f"experts {label}: relative error {err:.3e} too large for an MXFP8 expert stack"
