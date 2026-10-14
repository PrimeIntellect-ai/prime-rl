import prime_kernels
import pytest
import torch

from prime_rl.trainer.models.fp8_rounding import FP8RoundingState, setup_fp8_rounding
from prime_rl.trainer.models.layers.mxfp8_qat import mxfp8_qat_grouped_mm, mxfp8_qat_linear

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="MXFP8 QAT requires Blackwell",
    ),
]


@pytest.fixture(autouse=True)
def reset_compiler():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _decoded(x, **rounding):
    kernel = prime_kernels.load("mxfp8")
    return kernel.dequantize(*kernel.quantize(x, **rounding))


def _rounding(sr):
    state = FP8RoundingState(47, torch.empty((), device="cuda", dtype=torch.int64))
    state.set_step(7)
    return state, dict(rounding_seed=state.seed, rounding_salt=torch.tensor(19, device="cuda")) if sr else {}


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("groups", [1, 8])
@pytest.mark.parametrize("sr", [False, True])
def test_linear_forward_and_bf16_backward(groups, compiled, sr):
    torch.manual_seed(47)
    shape = (2, 64, 256) if groups == 1 else (2, 64, groups, 256)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((groups * 128, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    op = torch.compile(mxfp8_qat_linear, fullgraph=True) if compiled else mxfp8_qat_linear
    state, rounding = _rounding(sr)
    out = op(x, w, groups, **rounding)
    rounded = _decoded(w.detach(), **rounding)
    state.set_step(8)
    rounded_x = _decoded(x.detach().reshape(-1, 256)).reshape_as(x)
    if groups == 1:
        forward = torch.nn.functional.linear(rounded_x, rounded)
        surrogate = torch.nn.functional.linear(x, w + (rounded - w).detach())
    else:
        forward = (
            torch.bmm(
                rounded_x.reshape(-1, groups, 256).transpose(0, 1),
                rounded.view(groups, 128, 256).transpose(1, 2),
            )
            .transpose(0, 1)
            .reshape_as(out)
        )
        surrogate = (
            torch.bmm(
                x.reshape(-1, groups, 256).transpose(0, 1),
                (w + (rounded - w).detach()).view(groups, 128, 256).transpose(1, 2),
            )
            .transpose(0, 1)
            .reshape_as(out)
        )
    torch.testing.assert_close(out, forward, rtol=0, atol=0)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    expected = torch.autograd.grad(surrogate, (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("sr", [False, True])
def test_expert_forward_and_bf16_backward(compiled, empty, sr):
    torch.manual_seed(51)
    x = torch.randn((0 if empty else 512, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((4, 128, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    offs = torch.tensor([0, 0, 0, 0] if empty else [128, 128, 384, 512], device="cuda", dtype=torch.int32)
    op = torch.compile(mxfp8_qat_grouped_mm, fullgraph=True) if compiled else mxfp8_qat_grouped_mm
    state, rounding = _rounding(sr)
    out = op(x, w, offs, **rounding)
    state.set_step(8)
    grad = torch.randn_like(out)
    actual = torch.autograd.grad(out, (x, w), grad)
    if empty:
        assert actual[0].shape == x.shape
        assert torch.count_nonzero(actual[1]) == 0
        return
    rounded = _decoded(w.detach().transpose(1, 2), **rounding).transpose(1, 2)
    forward = torch._grouped_mm(_decoded(x.detach()), rounded, offs=offs)
    torch.testing.assert_close(out, forward, rtol=0, atol=0)
    surrogate = torch._grouped_mm(x, w + (rounded - w).detach(), offs=offs)
    expected = torch.autograd.grad(surrogate, (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_expert_dispatch_padding():
    from prime_rl.trainer.distributed.token_dispatcher import (
        permute_for_grouped_gemm,
        unpermute_from_grouped_gemm,
    )

    counts = torch.tensor([33, 0, 133, 7], device="cuda", dtype=torch.int32)
    x = torch.randn((173, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn((4, 256, 256), device="cuda", dtype=torch.bfloat16).transpose(1, 2).requires_grad_()
    padded, aligned_counts, permutation = permute_for_grouped_gemm(
        x, counts, experts_per_rank=4, num_ranks=1, alignment=128
    )
    offsets = aligned_counts.cumsum(0, dtype=torch.int32)
    assert offsets[-1] < padded.shape[0]
    output = unpermute_from_grouped_gemm(mxfp8_qat_grouped_mm(padded, w, offsets), permutation)
    rounded = _decoded(w.detach().transpose(1, 2)).transpose(1, 2)
    reference = []
    surrogate = []
    for index, chunk in enumerate(x.split([33, 0, 133, 7])):
        reference.append(_decoded(chunk.detach()) @ rounded[index])
        surrogate.append(chunk @ (w[index] + (rounded[index] - w[index]).detach()))
    torch.testing.assert_close(output, torch.cat(reference), atol=0, rtol=0)
    grad = torch.randn_like(output)
    actual = torch.autograd.grad(output, (x, w), grad)
    expected = torch.autograd.grad(torch.cat(surrogate), (x, w), grad)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["full", "selective"])
@pytest.mark.parametrize("sr", [False, True])
def test_checkpoint_replay(compiled, mode, sr):
    from prime_rl.configs.trainer import ActivationCheckpointConfig
    from prime_rl.trainer.activation_checkpointing import get_activation_checkpoint_wrapper
    from prime_rl.trainer.models.layers.mxfp8_qat import MXFP8QATLinear

    model = (
        torch.nn.Sequential(
            MXFP8QATLinear(128, 128, bias=False, stochastic_rounding=sr),
            torch.nn.SiLU(),
            MXFP8QATLinear(128, 128, bias=False, stochastic_rounding=sr),
        )
        .cuda()
        .bfloat16()
    )
    setup_fp8_rounding(model, 47)
    x = torch.randn((64, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    eager = torch.compile(model, fullgraph=True) if compiled else model
    expected = eager(x)
    expected_grads = torch.autograd.grad(expected.sum(), (x, *model.parameters()))
    wrapped = get_activation_checkpoint_wrapper(ActivationCheckpointConfig(mode=mode))(model)
    if compiled:
        wrapped = torch.compile(wrapped, fullgraph=True)
    actual = wrapped(x)
    actual_grads = torch.autograd.grad(actual.sum(), (x, *model.parameters()))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
