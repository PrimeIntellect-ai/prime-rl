import pytest
import torch

from prime_rl.trainer.distributed.deterministic_scatter import deterministic_scatter_add
from prime_rl.trainer.distributed.token_dispatcher import _scatter_routed_output


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("warn_only", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_scatter_restores_determinism_settings(enabled, warn_only, fail):
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        out = torch.zeros(2, 3)
        src = torch.ones(4, 3)
        index = torch.tensor([0, 1, 0, 1]).view(-1, 1).expand_as(src)
        if fail:
            with pytest.raises(RuntimeError):
                deterministic_scatter_add(out, index.float(), src)
        else:
            result = deterministic_scatter_add(out, index, src)
            torch.testing.assert_close(result, torch.full_like(out, 2))
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scatter_compiled_dynamic_backward_and_registration():
    compiled = torch.compile(deterministic_scatter_add, fullgraph=True, dynamic=True)
    for tokens, width in [(3, 13), (7, 257)]:
        out = torch.randn(tokens, width, device="cuda", requires_grad=True)
        src = torch.randn(tokens * 8, width, device="cuda", dtype=torch.bfloat16).float().requires_grad_()
        index = torch.randperm(tokens * 8, device="cuda").div(8, rounding_mode="floor")[:, None].expand_as(src)
        expected = deterministic_scatter_add(out, index, src)
        actual = compiled(out, index, src)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for _ in range(3):
            torch.testing.assert_close(compiled(out, index, src), actual, rtol=0, atol=0)
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (out, src), grad)
        torch.testing.assert_close(actual_grads[0], grad, rtol=0, atol=0)
        torch.testing.assert_close(actual_grads[1], torch.gather(grad, 0, index), rtol=0, atol=0)
        assert not torch.are_deterministic_algorithms_enabled()
        torch.library.opcheck(deterministic_scatter_add, (out, index, src))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_scoring_preserves_contribution_rounding():
    compiled = torch.compile(_scatter_routed_output, fullgraph=True, dynamic=True)
    torch.manual_seed(7)
    values = torch.randn(56, 3072, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    scores = torch.rand(56, device="cuda", requires_grad=True)
    indices = torch.randperm(56, device="cuda") // 8
    kwargs = dict(num_tokens=7, token_indices_experts_sorted=indices, scores_after_experts=scores)
    expected = _scatter_routed_output(values, **kwargs)
    actual = compiled(values, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (values, scores), grad)
    expected_grads = torch.autograd.grad(expected, (values, scores), grad)
    torch.testing.assert_close(actual_grads[0], expected_grads[0], rtol=0, atol=0)
    torch.testing.assert_close(actual_grads[1], expected_grads[1], rtol=1e-4, atol=1e-5)
