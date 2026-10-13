import pytest
import torch

from prime_rl.trainer.distributed.token_dispatcher import _local_reorder, _scatter_routed_output


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("top_k", [1, 2, 8])
@pytest.mark.parametrize("score_after_experts", [False, True])
def test_combine_restores_routes_and_gradients(device, dtype, top_k, score_after_experts):
    torch.manual_seed(7)
    num_tokens, dim, num_experts = 7, 13, 16
    selected = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(num_tokens)]).to(device)
    contributions = torch.randn(num_tokens, top_k, dim, device=device, dtype=dtype, requires_grad=True)
    scores = torch.rand(num_tokens, top_k, device=device, requires_grad=True)
    _, order, sorted_scores, _ = _local_reorder(
        torch.zeros(num_tokens, dim, device=device, dtype=dtype),
        scores,
        selected,
        num_experts=num_experts,
        top_k=top_k,
    )
    sorted_contributions = contributions.reshape(-1, dim)[torch.argsort(selected.flatten(), stable=True)]
    actual = _scatter_routed_output(
        sorted_contributions,
        num_tokens=num_tokens,
        token_indices_experts_sorted=order,
        scores_after_experts=sorted_scores if score_after_experts else None,
    )
    weighted = contributions
    if score_after_experts:
        weighted = (contributions.float() * scores.unsqueeze(-1)).to(dtype)
    accumulation_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    expected = weighted.sum(dim=1, dtype=accumulation_dtype).to(dtype)
    torch.testing.assert_close(actual, expected)
    gradient = torch.randn_like(actual)
    inputs = (contributions, scores) if score_after_experts else (contributions,)
    actual_grads = torch.autograd.grad(actual, inputs, gradient, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, inputs, gradient)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_combine_preserves_small_contributions_and_repeats(device):
    values = torch.tensor([256, 1, -256, 1], dtype=torch.bfloat16, device=device)
    values = values.repeat(257, 37).reshape(257, 37, 4).transpose(1, 2).contiguous()
    order = torch.randperm(257 * 4, device=device)
    top_k = 4
    sorted_values = values.reshape(-1, 37)[order]
    for _ in range(10):
        actual = _scatter_routed_output(
            sorted_values,
            num_tokens=257,
            token_indices_experts_sorted=order // top_k,
            scores_after_experts=None,
        )
        torch.testing.assert_close(actual, torch.full_like(actual, 2), rtol=0, atol=0)


def test_combine_empty_and_gradcheck(device):
    empty = torch.empty(0, 13, device=device, requires_grad=True)
    result = _scatter_routed_output(
        empty,
        num_tokens=0,
        token_indices_experts_sorted=torch.empty(0, device=device, dtype=torch.int64),
        scores_after_experts=None,
    )
    assert result.shape == (0, 13)
    result.sum().backward()
    assert empty.grad.shape == empty.shape

    top_k = 3
    order = torch.tensor([3, 0, 5, 1, 4, 2], device=device)
    values = torch.randn(6, 5, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: _scatter_routed_output(
            x,
            num_tokens=2,
            token_indices_experts_sorted=order // top_k,
            scores_after_experts=None,
        ),
        (values,),
    )
