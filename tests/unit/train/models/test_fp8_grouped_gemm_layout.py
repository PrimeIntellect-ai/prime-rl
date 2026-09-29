import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from prime_rl.trainer.distributed.token_dispatcher import permute_for_grouped_gemm
from prime_rl.trainer.models.layers.fp8_grouped_gemm import grouped_fp8_gemm
from prime_rl.trainer.models.layers.grouped_gemm import DeepGemmFP8GroupedGemm
from prime_rl.trainer.models.layers.moe import GroupedExperts
from tests.unit.train.models.test_fp8_grouped_gemm import (
    COUNTS,
    DISPATCHES,
    EXPERTS_RTOL,
    NUM_EXPERTS,
    SHAPE_IDS,
    SHAPES,
    WEIGHT_STD,
    _assert_bitwise,
    _assert_relative,
    _frozen_wrapper,
    _inputs,
)
from tests.unit.train.models.test_moe import ReferenceGroupedGemm

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
        reason="DeepGEMM's FP8 grouped GEMMs and the Triton fp8e4nv casts need Hopper (SM90) or newer",
    ),
]


@pytest.fixture(autouse=True)
def _require_deep_gemm():
    pytest.importorskip("deep_gemm")


def _parameter_layout(weight: torch.Tensor) -> torch.Tensor:
    """`weight` (G, K, N) as `moe.py` passes it: a transposed view of a contiguous (G, N, K) parameter."""
    return weight.transpose(1, 2).contiguous().transpose(1, 2)


def _backward_op(x, weight, offs, probe, *, needs_grad_x=True, needs_grad_weight=True):
    grad_weight_transposed = not weight.is_contiguous()
    return torch.ops.prime_rl.grouped_fp8_gemm_backward(
        probe, x, weight, offs, needs_grad_x, needs_grad_weight, grad_weight_transposed
    )


def _accumulate_grad_copies(run) -> int:
    """Count the copies autograd's AccumulateGrad makes while `run` executes."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        run()
        torch.cuda.synchronize()
    copies = 0
    for event in profile.events():
        if event.name not in ("aten::copy_", "aten::clone"):
            continue
        parent = event.cpu_parent
        while parent is not None and "AccumulateGrad" not in parent.name:
            parent = parent.cpu_parent
        copies += parent is not None
    return copies


@pytest.mark.parametrize("layout", ["parameter", "contiguous"])
@pytest.mark.parametrize(("k", "n"), SHAPES, ids=SHAPE_IDS)
def test_grad_weight_has_the_weight_strides(k, n, layout):
    """The real op and its fake both return `grad_weight` with the strides of the `weight` they were given."""
    x, weight, offs, _, probe = _inputs(COUNTS["ragged"], k, n, "align128")
    if layout == "parameter":
        weight = _parameter_layout(weight)
    assert weight.is_contiguous() == (layout == "contiguous")

    grad_x, grad_weight = _backward_op(x, weight, offs, probe)
    assert grad_x.shape == x.shape and grad_x.stride() == x.stride()
    assert grad_weight.shape == weight.shape and grad_weight.stride() == weight.stride(), (
        f"grad_weight strides {grad_weight.stride()} differ from weight strides {weight.stride()}"
    )

    with FakeTensorMode() as mode:
        fake_args = [mode.from_tensor(t) for t in (x, weight, offs, probe)]
        fake_grad_x, fake_grad_weight = _backward_op(*fake_args)
    assert fake_grad_x.stride() == grad_x.stride()
    assert fake_grad_weight.stride() == grad_weight.stride(), (
        f"fake grad_weight strides {fake_grad_weight.stride()} differ from the real op's {grad_weight.stride()}"
    )


@pytest.mark.parametrize("layout", ["parameter", "contiguous"])
def test_grad_weight_placeholder_has_the_weight_strides(layout):
    """With `needs_grad_weight=False` the unused output still matches the fake's strides."""
    x, weight, offs, _, probe = _inputs(COUNTS["ragged"], 512, 256, "align128")
    if layout == "parameter":
        weight = _parameter_layout(weight)
    _, grad_weight = _backward_op(x, weight, offs, probe, needs_grad_weight=False)
    assert grad_weight.shape == weight.shape and grad_weight.stride() == weight.stride()


@pytest.mark.parametrize(("k", "n"), SHAPES, ids=SHAPE_IDS)
@pytest.mark.parametrize("distribution", list(COUNTS))
@pytest.mark.parametrize("dispatch", DISPATCHES)
def test_parameter_layout_grad_weight_is_bit_identical_to_the_frozen_wrapper(dispatch, distribution, k, n):
    """Writing the wgrad as (G, N, K) swaps DeepGEMM's two k-grouped operands; the values must not move a bit."""
    x, weight, offs, _, probe = _inputs(COUNTS[distribution], k, n, dispatch)
    _, _, frozen_grad_weight = _frozen_wrapper(x, weight, offs, probe)

    _, grad_weight = _backward_op(x, _parameter_layout(weight), offs, probe)
    assert torch.equal(grad_weight.contiguous().view(torch.uint8), frozen_grad_weight.view(torch.uint8)), (
        "swapping the k-grouped wgrad operands is not bitwise identical to the (G, K, N) wgrad"
    )


def test_parameter_grad_is_accumulated_without_a_copy():
    """Through an (E, N, K) parameter used as `param.transpose(-2, -1)`, AccumulateGrad takes the op's gradient as is."""
    x, weight, offs, _, probe = _inputs(COUNTS["ragged"], 512, 256, "align128")
    _, _, frozen_grad_weight = _frozen_wrapper(x, weight, offs, probe)
    param_data = weight.transpose(1, 2).contiguous()

    def backward_through(weight_view_of):
        param = torch.nn.Parameter(param_data.clone())
        out = grouped_fp8_gemm(x, weight_view_of(param), offs)
        copies = _accumulate_grad_copies(lambda: (out * probe).sum().backward())
        return param, copies

    param, copies = backward_through(lambda p: p.transpose(-2, -1))
    assert param.grad.is_contiguous() and param.grad.shape == param.shape
    _assert_bitwise(param.grad, frozen_grad_weight.transpose(1, 2).contiguous(), "param.grad")
    assert copies == 0, f"AccumulateGrad copied the weight gradient {copies} times"

    old_layout_param, old_layout_copies = backward_through(lambda p: p.transpose(-2, -1).contiguous())
    _assert_bitwise(old_layout_param.grad, param.grad, "param.grad through a contiguous weight")
    assert old_layout_copies > 0, (
        "vacuous probe: a (G, K, N) gradient into an (E, N, K) parameter must be copied by AccumulateGrad, "
        "but the profiler saw no copy"
    )


def test_grouped_experts_accumulate_weight_grads_without_a_copy():
    """`GroupedExperts` with the FP8 op: contiguous parameter grads, no AccumulateGrad copy, bf16 reference agreement."""
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

    x, padded_counts, _ = permute_for_grouped_gemm(
        tokens,
        torch.tensor(counts, dtype=torch.int64, device="cuda"),
        experts_per_rank=NUM_EXPERTS,
        num_ranks=1,
        alignment=grouped_gemm.token_group_alignment,
    )
    used_rows = int(padded_counts.sum())
    probe = torch.randn(used_rows, dim, device="cuda", dtype=torch.bfloat16)

    reference_output = reference(x, padded_counts)
    (reference_output[:used_rows] * probe).sum().backward()
    output = experts(x, padded_counts)
    copies = _accumulate_grad_copies(lambda: (output[:used_rows] * probe).sum().backward())

    assert copies == 0, f"AccumulateGrad copied FP8 expert weight gradients {copies} times"
    for name, param in reference.named_parameters():
        fp8_grad = experts.get_parameter(name).grad
        assert fp8_grad.is_contiguous() and fp8_grad.stride() == param.stride(), name
        assert param.grad.abs().max() > 0, f"vacuous probe: reference {name} has no grad"
        _assert_relative(fp8_grad, param.grad, EXPERTS_RTOL, name)


def test_parameter_layout_traces_under_torch_compile():
    """`torch.compile(fullgraph=True)` trusts the fake's strides; compiled parameter grads must equal eager's bitwise."""
    x, weight, offs, _, probe = _inputs(COUNTS["ragged"], 512, 256, "align128")
    param_data = weight.transpose(1, 2).contiguous()

    def grouped(x: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
        return grouped_fp8_gemm(x, param.transpose(-2, -1), offs)

    results, copies = {}, {}
    for name, fn in (("eager", grouped), ("compiled", torch.compile(grouped, fullgraph=True))):
        x_leaf = x.clone().requires_grad_(True)
        param = torch.nn.Parameter(param_data.clone())
        out = fn(x_leaf, param)
        copies[name] = _accumulate_grad_copies(lambda: (out * probe).sum().backward())
        results[name] = (x_leaf.grad, param.grad)

    used_rows = int(offs[-1])
    assert copies == {"eager": 0, "compiled": 0}, f"AccumulateGrad copies: {copies}"
    assert results["compiled"][1].is_contiguous() and results["eager"][1].is_contiguous()
    _assert_bitwise(results["compiled"][0][:used_rows], results["eager"][0][:used_rows], "grad_x")
    _assert_bitwise(results["compiled"][1], results["eager"][1], "param.grad")
