import pytest
import torch

from prime_rl.trainer.optim.sinkhorn import sinkhorn_reference, sinkhorn_update_

pytestmark = [pytest.mark.gpu]


@pytest.mark.parametrize("shape", [(300, 256), (300, 200), (97, 1100)])
def test_sinkhorn_update_matches_dense_algorithm(shape):
    torch.manual_seed(0)
    weight = 1e-3 * torch.randn(shape, device="cuda")
    momentum = torch.zeros(shape, device="cuda")
    ref_weight, ref_momentum = weight.double(), momentum.double()
    hyper = dict(lr=1e-2, beta=0.95, num_iters=11, tau=1e-3, eps=1e-20, lr_scale=0.18)
    for step in range(3):
        # Row scales spanning orders of magnitude, rows without gradient (masked on the first step, moved by
        # momentum later) and a column without gradient.
        grad = torch.randn(shape, device="cuda") * torch.logspace(-6, 0, shape[0], device="cuda")[:, None]
        grad[step::7] = 0.0
        grad[:, 3] = 0.0
        ref_update = ref_weight
        ref_weight, ref_momentum = sinkhorn_reference(ref_weight, grad.double(), ref_momentum, **hyper)
        ref_update = ref_weight - ref_update
        before = weight.clone()
        sinkhorn_update_(weight, grad, momentum, **hyper)
        torch.testing.assert_close(momentum, ref_momentum.float(), rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(weight - before, ref_update.float(), rtol=1e-4, atol=1e-9)
        # Both sides start every step from the same fp32 weight and momentum.
        ref_weight, ref_momentum = weight.double(), momentum.double()
