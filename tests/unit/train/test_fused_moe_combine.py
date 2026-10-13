import pytest
import torch

from prime_rl.trainer.distributed.fused_moe_combine import fused_moe_combine
from prime_rl.trainer.distributed.token_dispatcher import _combine_routed_output

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k,width", [(1, 13), (2, 257), (7, 3072), (8, 8192)])
@pytest.mark.parametrize("weighted", [False, True])
def test_fused_combine_forward_backward_and_repeatability(dtype, top_k, width, weighted):
    torch.manual_seed(7)
    rows, tokens = 11 * top_k, 11
    # Exercise strided input, scores and incoming gradients.
    values = torch.randn(rows, width * 2, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    order = torch.randperm(rows, device="cuda")
    scores = torch.rand(rows * 2, device="cuda")[::2].requires_grad_() if weighted else None
    actual = fused_moe_combine(values, order, scores, tokens)
    expected = _combine_routed_output(
        values,
        num_tokens=tokens,
        route_indices_experts_sorted=order,
        scores_after_experts=scores,
    )
    torch.testing.assert_close(actual, expected, rtol=0.008 if dtype == torch.bfloat16 else 0.001, atol=1e-5)
    for _ in range(3):
        torch.testing.assert_close(fused_moe_combine(values, order, scores, tokens), actual, rtol=0, atol=0)
    grad = torch.randn(tokens, width * 2, device="cuda", dtype=dtype)[:, ::2]
    inputs = (values, scores) if weighted else (values,)
    actual_grads = torch.autograd.grad(actual, inputs, grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, grad)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=0, atol=0)
    if weighted:
        torch.testing.assert_close(actual_grads[1], expected_grads[1], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("weighted", [False, True])
def test_fused_combine_compiled_dynamic_backward(weighted):
    compiled = torch.compile(fused_moe_combine, fullgraph=True, dynamic=True)
    for tokens, width, top_k in [(7, 13, 2), (11, 257, 8)]:
        values = torch.randn(tokens * top_k, width, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        order = torch.randperm(tokens * top_k, device="cuda")
        scores = torch.rand(tokens * top_k, device="cuda", requires_grad=True) if weighted else None
        eager = fused_moe_combine(values, order, scores, tokens)
        actual = compiled(values, order, scores, tokens)
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        inputs = (values, scores) if weighted else (values,)
        grad = torch.randn_like(actual)
        eager_grads = torch.autograd.grad(eager, inputs, grad)
        actual_grads = torch.autograd.grad(actual, inputs, grad)
        for x, y in zip(actual_grads, eager_grads, strict=True):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_fused_combine_registration_and_cancellation():
    values = torch.tensor([256, 1, -256, 1], device="cuda", dtype=torch.bfloat16).repeat(3, 17, 1)
    values = values.transpose(1, 2).reshape(12, 17).contiguous().requires_grad_()
    order = torch.randperm(12, device="cuda")
    values = values[order].detach().requires_grad_()
    scores = torch.ones(12, device="cuda", requires_grad=True)
    torch.library.opcheck(fused_moe_combine, (values, order, scores, 3))
    torch.testing.assert_close(fused_moe_combine(values, order, scores, 3), values.new_full((3, 17), 2), rtol=0, atol=0)


def test_fused_combine_opt_in_fallback():
    for device, dtype, tokens in [("cpu", torch.bfloat16, 2), ("cuda", torch.float32, 2), ("cuda", torch.bfloat16, 0)]:
        values = torch.randn(tokens * 2, 13, device=device, dtype=dtype, requires_grad=True)
        order = torch.randperm(tokens * 2, device=device)
        kwargs = dict(num_tokens=tokens, route_indices_experts_sorted=order, scores_after_experts=None)
        actual = _combine_routed_output(values, **kwargs, use_fused=True)
        expected = _combine_routed_output(values, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        assert values.grad.shape == values.shape


@pytest.mark.parametrize("score_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_fused_combine_score_dtypes(score_dtype):
    torch.manual_seed(7)
    values = torch.randn(56, 257, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    order = torch.randperm(56, device="cuda")
    scores = torch.rand(56, device="cuda", dtype=score_dtype, requires_grad=True)
    actual = fused_moe_combine(values, order, scores, 7)
    expected = _combine_routed_output(
        values, num_tokens=7, route_indices_experts_sorted=order, scores_after_experts=scores
    )
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (values, scores), grad)
    expected_grads = torch.autograd.grad(expected, (values, scores), grad)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], rtol=0.008, atol=1e-4)
