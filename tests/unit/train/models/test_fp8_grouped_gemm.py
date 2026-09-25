import math
from contextlib import contextmanager

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from prime_rl.trainer.distributed.token_dispatcher import permute_for_grouped_gemm
from prime_rl.trainer.models.kernels.fp8_utils import (
    GROUP_ALIGNMENT,
    build_grouped_layout,
    grouped_per_block_cast_to_fp8_triton,
    grouped_per_channel_cast_to_fp8_rowmajor_triton,
    grouped_per_channel_cast_to_fp8_sm90_kmajor_triton,
    grouped_per_token_cast_to_fp8_triton,
    ue8m0_for_device,
)
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm
from prime_rl.trainer.models.layers.grouped_gemm import DeepGemmFP8GroupedGemm
from prime_rl.trainer.models.layers.moe import GroupedExperts
from tests.unit.train.models.test_moe import ReferenceGroupedGemm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="DeepGEMM's FP8 grouped GEMMs and the Triton fp8e4nv casts need Hopper (SM90) or newer",
    ),
]

NUM_EXPERTS = 8

# Tokens per expert, in expert order. "ragged" puts 7x the mean on one expert, the max_vio of about 6
# the DeepSeek V4 runs log, and leaves the first and last experts empty. "sub_alignment" keeps every
# group under the 128-row block the kernels quantize and multiply in, so every group is mostly padding.
COUNTS = {
    "balanced": [160] * NUM_EXPERTS,
    "ragged": [0, 896, 3, 60, 1, 40, 24, 0],
    "sub_alignment": [1, 127, 5, 64, 0, 33, 100, 7],
}

# How `x` and `offs` are laid out for the op, which takes 128-row groups. "align128" goes through the
# real dispatcher (`permute_for_grouped_gemm`), which pads each expert to 128 rows, gives an empty
# expert one alignment's worth of zero rows, and leaves a zero tail past `offs[-1]`. "raw128" pads
# each expert to 128 rows by hand but keeps an empty expert a truly empty group, followed by a zero
# tail. "align8", the dispatcher at the old alignment, only feeds the frozen wrapper.
DISPATCHES = ["align128", "raw128"]
RAW_TAIL_ROWS = 128

# (in_features, out_features) of the expert weight. Non-square in both directions so a transposed
# operand cannot pass.
SHAPES = [(512, 256), (256, 512)]
SHAPE_IDS = [f"k{k}-n{n}" for k, n in SHAPES]

WEIGHT_STD = 0.02

# The op against a float32 oracle that multiplies the same FP8-rounded operands. Both sides then see
# identical inputs, so the op may differ only by accumulation order and its bfloat16 output rounding,
# and it is bounded by a multiple of what that rounding costs the oracle itself. The same multiple is
# required of the FP8 rounding's own effect, so a GEMM that skipped quantization cannot pass either.
FP8_ORACLE_SLACK = 2.0

# The FP8 experts against the bfloat16 reference experts, relative to the reference's own scale. e4m3
# keeps 3 mantissa bits, so each quantized operand carries up to 2^-4 relative rounding, and the
# SwiGLU between the two GEMMs compounds it. 0.1 leaves room for that while a wrong expert, a
# transposed weight or a dropped group moves the output by the order of its whole scale.
EXPERTS_RTOL = 0.1


@pytest.fixture(autouse=True)
def _require_deep_gemm():
    pytest.importorskip("deep_gemm")


def _assert_relative(actual: torch.Tensor, reference: torch.Tensor, rtol: float, label: str) -> None:
    """Bound the largest absolute deviation by `rtol` times the reference's own scale."""
    actual, reference = actual.float(), reference.float()
    deviation = (actual - reference).abs().max()
    scale = reference.abs().max()
    assert deviation <= rtol * scale, f"{label}: max deviation {deviation} exceeds {rtol} * scale {scale}"


def _assert_bitwise(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, label
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), f"{label}: bytes differ"


@contextmanager
def _forbid_device_to_host_sync():
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode("default")


class _GroupedFP8CallCounter(TorchDispatchMode):
    """Count the registered forward and backward ops as the dispatcher sees them, so a bf16 fallback cannot pass."""

    def __init__(self) -> None:
        super().__init__()
        self.forward = 0
        self.backward = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.prime_rl.grouped_fp8_gemm.default:
            self.forward += 1
        elif func is torch.ops.prime_rl.grouped_fp8_gemm_backward.default:
            self.backward += 1
        return func(*args, **(kwargs or {}))


def _dispatch(
    tokens: torch.Tensor, counts: list[int], dispatch: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lay expert-sorted `tokens` out as the grouped GEMM sees them: `x`, int32 `offs`, and the mask of real rows."""
    if dispatch == "raw128":
        padded_counts = [math.ceil(count / GROUP_ALIGNMENT) * GROUP_ALIGNMENT for count in counts]
        x = tokens.new_zeros(sum(padded_counts) + RAW_TAIL_ROWS, tokens.shape[1])
        real_rows = torch.zeros(x.shape[0], dtype=torch.bool, device=tokens.device)
        src = dst = 0
        for count, padded_count in zip(counts, padded_counts):
            x[dst : dst + count] = tokens[src : src + count]
            real_rows[dst : dst + count] = True
            src += count
            dst += padded_count
        offs = torch.tensor(padded_counts, device=tokens.device).cumsum(0).to(torch.int32)
        return x, offs, real_rows

    alignment = {"align8": 8, "align128": 128}[dispatch]
    x, padded_counts, state = permute_for_grouped_gemm(
        tokens,
        torch.tensor(counts, dtype=torch.int64, device=tokens.device),
        experts_per_rank=len(counts),
        num_ranks=1,
        alignment=alignment,
    )
    offs = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    return x, offs, state.permuted_indices != -1


def _inputs(counts: list[int], k: int, n: int, dispatch: str, seed: int = 0):
    """`x`, `weight`, `offs`, the real-row mask and a loss probe that is zero off the real rows."""
    torch.manual_seed(seed)
    with torch.device("cuda"):
        tokens = torch.randn(sum(counts), k, dtype=torch.bfloat16)
        weight = (torch.randn(len(counts), k, n) * WEIGHT_STD).to(torch.bfloat16)
    x, offs, real_rows = _dispatch(tokens, counts, dispatch)
    probe = torch.randn(x.shape[0], n, device="cuda", dtype=torch.bfloat16) * real_rows.unsqueeze(1)
    return x, weight, offs, real_rows, probe


def _run_op(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor, probe: torch.Tensor):
    x_leaf = x.clone().requires_grad_(True)
    weight_leaf = weight.clone().requires_grad_(True)
    out = grouped_fp8_gemm(x_leaf, weight_leaf, offs)
    (out * probe).sum().backward()
    return out.detach(), x_leaf.grad, weight_leaf.grad


def _frozen_grad_weight(
    x, grad_output, weight, padded_total_m, block_to_group, ks_tensor, starts, actual_ms, block_starts
):
    import deep_gemm

    layout_args = (padded_total_m, block_to_group, starts, actual_ms, ks_tensor, block_starts)
    if torch.cuda.get_device_capability(x.device)[0] >= 10:
        x_fp8 = grouped_per_channel_cast_to_fp8_rowmajor_triton(x, *layout_args, True, GROUP_ALIGNMENT)
        dy_fp8 = grouped_per_channel_cast_to_fp8_rowmajor_triton(grad_output, *layout_args, True, GROUP_ALIGNMENT)
        grouped_weight_grad = deep_gemm.k_grouped_fp8_gemm_tn_contiguous
    else:
        x_fp8 = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(x, *layout_args, False, GROUP_ALIGNMENT)
        dy_fp8 = grouped_per_channel_cast_to_fp8_sm90_kmajor_triton(grad_output, *layout_args, False, GROUP_ALIGNMENT)
        grouped_weight_grad = deep_gemm.k_grouped_fp8_gemm_nt_contiguous
    grad_weight = torch.zeros(weight.shape, device=x.device, dtype=torch.float32)
    grouped_weight_grad(x_fp8, dy_fp8, grad_weight, ks_tensor.tolist(), ks_tensor, grad_weight)
    return grad_weight.to(weight.dtype)


def _frozen_unpack_rows(padded: torch.Tensor, total_m: int, starts, actual_ms, block_starts) -> torch.Tensor:
    """Move each group's rows from its 128-aligned slot in `padded` back to where it starts in `x`."""
    ends = starts + actual_ms
    rows = torch.arange(total_m, device=padded.device)
    group = torch.searchsorted(ends, rows, right=True)
    in_group = group < ends.numel()
    group = group.clamp(max=ends.numel() - 1)
    src_rows = block_starts[group] * GROUP_ALIGNMENT + rows - starts[group]
    out = padded.new_zeros((total_m, padded.size(1)))
    out[in_group] = padded[src_rows[in_group]]
    return out


def _frozen_wrapper(x: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor, grad_output: torch.Tensor):
    """The grouped FP8 GEMM wrapper as of ae37b35a3, built from the primitive kernels, returning out, grad_x, grad_weight."""
    import deep_gemm

    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    padded_total_m = int((torch.ceil(counts / GROUP_ALIGNMENT) * GROUP_ALIGNMENT).sum())
    grouped_layout, block_to_group, ks_tensor, starts, actual_ms, block_starts = build_grouped_layout(
        offs, padded_total_m
    )
    total_m = x.size(0)
    group_args = (block_to_group, starts, actual_ms, block_starts)
    cast_args = (padded_total_m, *group_args)
    use_ue8m0 = ue8m0_for_device(x.device)

    x_fp8 = grouped_per_token_cast_to_fp8_triton(x, *cast_args, use_ue8m0, GROUP_ALIGNMENT)
    weight_fp8 = grouped_per_block_cast_to_fp8_triton(weight.transpose(1, 2), use_ue8m0, GROUP_ALIGNMENT)
    out_padded = torch.empty((padded_total_m, weight.size(2)), device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(x_fp8, weight_fp8, out_padded, grouped_layout, use_psum_layout=False)
    out = _frozen_unpack_rows(out_padded, total_m, starts, actual_ms, block_starts)

    grad_output = grad_output.contiguous()
    grad_weight = _frozen_grad_weight(
        x, grad_output, weight, padded_total_m, block_to_group, ks_tensor, starts, actual_ms, block_starts
    )
    dy_fp8 = grouped_per_token_cast_to_fp8_triton(grad_output, *cast_args, use_ue8m0, GROUP_ALIGNMENT)
    weight_dx_fp8 = grouped_per_block_cast_to_fp8_triton(weight, use_ue8m0, GROUP_ALIGNMENT)
    grad_x_padded = torch.empty((padded_total_m, weight.size(1)), device=x.device, dtype=x.dtype)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        dy_fp8, weight_dx_fp8, grad_x_padded, grouped_layout, use_psum_layout=False
    )
    grad_x = _frozen_unpack_rows(grad_x_padded, total_m, starts, actual_ms, block_starts)
    return out, grad_x, grad_weight


def _fake_fp8(a: torch.Tensor, block_rows: int, block_cols: int) -> torch.Tensor:
    """Round float32 `a` onto the e4m3 grid with one amax scale per `block_rows x block_cols` block."""
    rows, cols = a.shape
    padded = a.new_zeros(math.ceil(rows / block_rows) * block_rows, math.ceil(cols / block_cols) * block_cols)
    padded[:rows, :cols] = a
    blocks = padded.view(padded.shape[0] // block_rows, block_rows, padded.shape[1] // block_cols, block_cols)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-10)
    scale = amax / torch.full_like(amax, 448.0)
    rounded = (blocks / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * scale
    return rounded.view_as(padded)[:rows, :cols]


def _kernel_rounded_weight(weight: torch.Tensor) -> torch.Tensor:
    """The weight as the kernels' 128x128 block cast rounds it, dequantized to float32.

    Taken from the kernel rather than rebuilt in torch because the block cast quantizes with a
    reciprocal multiply, as vLLM does, which a correctly rounded torch division disagrees with. Its
    bytes are pinned against vLLM by `test_block_cast_matches_vllm_online_quant`.
    """
    groups, rows, cols = weight.shape
    fp8, scales = grouped_per_block_cast_to_fp8_triton(weight, ue8m0_for_device(weight.device), GROUP_ALIGNMENT)
    blocks = fp8.float().view(groups, rows // 128, 128, cols // 128, 128) * scales.view(
        groups, rows // 128, 1, cols // 128, 1
    )
    return blocks.view(groups, rows, cols)


def _float32_grouped_reference(x, weight, offs, probe, *, quantize: bool):
    """Per-expert float32 out, grad_x and grad_weight over rows `[0, offs[-1])`.

    With `quantize`, every operand is first rounded the way the kernels round it: activations and
    output gradients in 1x128 blocks along the feature axis for the forward and dgrad, 128x1 blocks
    along the token axis within each expert for the wgrad, and weights in 128x128 blocks.
    """
    ends = offs.tolist()
    starts = [0, *ends[:-1]]
    weight_f32 = _kernel_rounded_weight(weight) if quantize else weight.float()
    out = x.new_zeros(ends[-1], weight.shape[2], dtype=torch.float32)
    grad_x = x.new_zeros(ends[-1], weight.shape[1], dtype=torch.float32)
    grad_weight = torch.zeros_like(weight, dtype=torch.float32)
    for expert, (start, end) in enumerate(zip(starts, ends)):
        if end == start:
            continue
        x_e, dy_e, w_e = x[start:end].float(), probe[start:end].float(), weight_f32[expert]
        if quantize:
            out[start:end] = _fake_fp8(x_e, 1, 128) @ w_e
            grad_x[start:end] = _fake_fp8(dy_e, 1, 128) @ w_e.T
            grad_weight[expert] = _fake_fp8(x_e, 128, 1).T @ _fake_fp8(dy_e, 128, 1)
        else:
            out[start:end] = x_e @ w_e
            grad_x[start:end] = dy_e @ w_e.T
            grad_weight[expert] = x_e.T @ dy_e
    return out, grad_x, grad_weight


@pytest.mark.parametrize(("k", "n"), SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("distribution", list(COUNTS))
@pytest.mark.parametrize("dispatch", DISPATCHES)
def test_op_is_bit_identical_to_the_frozen_wrapper(dispatch, distribution, k, n):
    """Forward, dgrad and wgrad bytes must match the ae37b35a3 wrapper; every optimization claims bitwise identity."""
    x, weight, offs, _, probe = _inputs(COUNTS[distribution], k, n, dispatch)
    used_rows = int(offs[-1])

    out, grad_x, grad_weight = _run_op(x, weight, offs, probe)
    frozen_out, frozen_grad_x, frozen_grad_weight = _frozen_wrapper(x, weight, offs, probe)

    assert out.shape == (x.shape[0], n) and grad_x.shape == x.shape and grad_weight.shape == weight.shape
    _assert_bitwise(out[:used_rows], frozen_out[:used_rows], "out")
    _assert_bitwise(grad_x[:used_rows], frozen_grad_x[:used_rows], "grad_x")
    _assert_bitwise(grad_weight, frozen_grad_weight, "grad_weight")


@pytest.mark.parametrize(("k", "n"), SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("distribution", list(COUNTS))
@pytest.mark.parametrize("dispatch", DISPATCHES)
def test_op_matches_the_quantized_float32_oracle(dispatch, distribution, k, n):
    """Forward, dgrad and wgrad against float32 on the same FP8-rounded operands, within the bfloat16 output floor."""
    x, weight, offs, _, probe = _inputs(COUNTS[distribution], k, n, dispatch)
    used_rows = int(offs[-1])

    candidates = [t[:used_rows] if t.shape[0] == x.shape[0] else t for t in _run_op(x, weight, offs, probe)]
    exact = _float32_grouped_reference(x, weight, offs, probe, quantize=False)
    quantized = _float32_grouped_reference(x, weight, offs, probe, quantize=True)

    for label, candidate, exact_t, quantized_t in zip(("out", "grad_x", "grad_weight"), candidates, exact, quantized):
        bf16_floor = (quantized_t.bfloat16().float() - quantized_t).abs().max()
        fp8_cost = (quantized_t - exact_t).abs().max()
        assert fp8_cost > FP8_ORACLE_SLACK * bf16_floor, (
            f"vacuous probe: FP8 rounding moves the float32 {label} by only {fp8_cost}, within reach of the "
            f"bound, so an unquantized GEMM could pass"
        )
        gap = (candidate.float() - quantized_t).abs().max()
        assert gap <= FP8_ORACLE_SLACK * bf16_floor, (
            f"{label}: the op differs from the quantized oracle by {gap}, more than {FP8_ORACLE_SLACK}x the "
            f"{bf16_floor} bfloat16 output rounding costs it"
        )


@pytest.mark.parametrize("distribution", ["ragged", "sub_alignment"])
def test_empty_experts_get_an_exactly_zero_weight_gradient(distribution):
    """The k-grouped wgrad skips an empty group entirely, so its gradient is whatever the buffer was zeroed to."""
    counts = COUNTS[distribution]
    for dispatch in DISPATCHES:
        x, weight, offs, _, probe = _inputs(counts, 512, 256, dispatch)
        _, _, grad_weight = _run_op(x, weight, offs, probe)
        for expert, count in enumerate(counts):
            if count == 0:
                assert torch.equal(grad_weight[expert], torch.zeros_like(grad_weight[expert])), (
                    f"{dispatch}: empty expert {expert} got a non-zero weight gradient"
                )
            else:
                assert grad_weight[expert].abs().max() > 0, f"{dispatch}: expert {expert} got no weight gradient"


@pytest.mark.parametrize("distribution", list(COUNTS))
def test_128_aligned_dispatch_matches_8_aligned_on_real_rows(distribution):
    """Aligning the dispatcher to 128 changes only how many zero rows pad each expert, not a byte of any real row.

    The 8-aligned side runs the frozen wrapper and the 128-aligned side the op, so this still pins
    the change in dispatcher alignment after the op itself is reworked for 128-aligned groups.
    """
    counts = COUNTS[distribution]
    torch.manual_seed(0)
    with torch.device("cuda"):
        tokens = torch.randn(sum(counts), 512, dtype=torch.bfloat16)
        weight = (torch.randn(len(counts), 512, 256) * WEIGHT_STD).to(torch.bfloat16)
        real_probe = torch.randn(sum(counts), 256, dtype=torch.bfloat16)

    results = {}
    for dispatch in ("align8", "align128"):
        x, offs, real_rows = _dispatch(tokens, counts, dispatch)
        assert torch.equal(x[real_rows], tokens), f"{dispatch}: real rows are not the tokens in expert order"
        assert torch.equal(x[~real_rows], torch.zeros_like(x[~real_rows])), f"{dispatch}: a padding row is not zero"
        probe = torch.zeros(x.shape[0], 256, device="cuda", dtype=torch.bfloat16)
        probe[real_rows] = real_probe
        if dispatch == "align8":
            out, grad_x, grad_weight = _frozen_wrapper(x, weight, offs, probe)
        else:
            out, grad_x, grad_weight = _run_op(x, weight, offs, probe)
        results[dispatch] = (out[real_rows], grad_x[real_rows], grad_weight)

    for label, aligned8, aligned128 in zip(("out", "grad_x", "grad_weight"), results["align8"], results["align128"]):
        _assert_bitwise(aligned128, aligned8, label)


def test_grouped_experts_route_through_the_fp8_op_and_match_the_reference():
    """`GroupedExperts` with the DeepGEMM FP8 grouped GEMM, end to end through the dispatcher's padding."""
    dim, hidden_dim = 512, 256
    counts = COUNTS["ragged"]
    grouped_gemm = DeepGemmFP8GroupedGemm()
    with torch.device("cuda"):
        experts = GroupedExperts(dim, hidden_dim, NUM_EXPERTS, grouped_gemm=grouped_gemm)
        reference = GroupedExperts(dim, hidden_dim, NUM_EXPERTS, grouped_gemm=ReferenceGroupedGemm())
        torch.manual_seed(0)
        experts.init_weights(WEIGHT_STD)
        reference.load_state_dict(experts.state_dict())
        tokens = torch.randn(sum(counts), dim, dtype=torch.bfloat16)

    x, padded_counts, state = permute_for_grouped_gemm(
        tokens,
        torch.tensor(counts, dtype=torch.int64, device="cuda"),
        experts_per_rank=NUM_EXPERTS,
        num_ranks=1,
        alignment=grouped_gemm.token_group_alignment,
    )
    used_rows = int(padded_counts.sum())
    probe = torch.randn(used_rows, dim, device="cuda", dtype=torch.bfloat16)

    outputs, input_grads = {}, {}
    for name, module in (("fp8", experts), ("reference", reference)):
        x_leaf = x.clone().requires_grad_(True)
        with _GroupedFP8CallCounter() as counter:
            output = module(x_leaf, padded_counts)
            (output[:used_rows] * probe).sum().backward()
        expected_calls = 3 if name == "fp8" else 0
        assert (counter.forward, counter.backward) == (expected_calls, expected_calls), (
            f"{name}: {counter.forward} forward and {counter.backward} backward FP8 op calls"
        )
        outputs[name] = output[:used_rows].detach()
        input_grads[name] = x_leaf.grad[:used_rows]

    _assert_relative(outputs["fp8"], outputs["reference"], EXPERTS_RTOL, "output")
    _assert_relative(input_grads["fp8"], input_grads["reference"], EXPERTS_RTOL, "input gradient")
    for name, param in reference.named_parameters():
        assert param.grad is not None and param.grad.abs().max() > 0, f"vacuous probe: reference {name} has no grad"
        _assert_relative(experts.get_parameter(name).grad, param.grad, EXPERTS_RTOL, name)


def test_op_traces_under_torch_compile():
    """`torch.compile(fullgraph=True)` through the op, forward and backward, exercising both `register_fake`s."""
    x, weight, offs, _, probe = _inputs(COUNTS["ragged"], 512, 256, "align128")
    used_rows = int(offs[-1])

    def grouped(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return grouped_fp8_gemm(x, weight, offs)

    eager = [x.clone().requires_grad_(True), weight.clone().requires_grad_(True)]
    compiled = [x.clone().requires_grad_(True), weight.clone().requires_grad_(True)]
    eager_out = grouped(*eager)
    (eager_out * probe).sum().backward()
    compiled_out = torch.compile(grouped, fullgraph=True)(*compiled)
    (compiled_out * probe).sum().backward()

    assert compiled_out.shape == eager_out.shape and compiled_out.dtype == eager_out.dtype
    _assert_bitwise(compiled_out[:used_rows].detach(), eager_out[:used_rows].detach(), "out")
    _assert_bitwise(compiled[0].grad[:used_rows], eager[0].grad[:used_rows], "grad_x")
    _assert_bitwise(compiled[1].grad, eager[1].grad, "grad_weight")


def test_forward_does_not_sync_with_the_host():
    """The forward launches without a device-to-host sync, so the CPU can run ahead of the GPU.

    The backward is exempt: DeepGEMM's k-grouped wgrad takes the group sizes as a host list, so it
    keeps one `tolist()` by construction.
    """
    x, weight, offs, _, _ = _inputs(COUNTS["ragged"], 512, 256, "align128")
    grouped_fp8_gemm(x, weight, offs)
    torch.cuda.synchronize()

    with _forbid_device_to_host_sync():
        grouped_fp8_gemm(x, weight, offs)


def test_build_grouped_layout_does_not_sync_with_the_host():
    x, _, offs, _, _ = _inputs(COUNTS["ragged"], 512, 256, "align128")
    torch.cuda.synchronize()

    with _forbid_device_to_host_sync():
        build_grouped_layout(offs, x.size(0))
