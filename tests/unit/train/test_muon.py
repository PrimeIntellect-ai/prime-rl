import math

import pytest
import torch

from prime_rl.trainer.optim.muon import (
    HYBRID_NS_COEFFICIENTS,
    MuonParam,
    assign_owners,
    hybrid_newton_schulz,
    muon_step,
)

pytestmark = [pytest.mark.gpu]


def reference_newton_schulz(G: torch.Tensor) -> torch.Tensor:
    """DeepSeek-V4's hybrid Newton-Schulz (§2.4, eq. 28) on one matrix, in fp64."""
    X = G.double()
    transpose = X.shape[0] > X.shape[1]
    if transpose:
        X = X.T
    X = X / X.norm()
    for a, b, c in HYBRID_NS_COEFFICIENTS:
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    return X.T if transpose else X


def reference_muon(weight, grad, momentum, lr, mu, weight_decay, update_rms, row_blocks=None):
    """Algorithm 1 of the DeepSeek-V4 report for one parameter, matrix by matrix, in fp64."""
    weight, grad, momentum = weight.double(), grad.double(), momentum.double()
    momentum = mu * momentum + grad
    update = mu * momentum + grad
    ortho = []
    for matrix in update.reshape(-1, *update.shape[-2:]):
        parts = matrix.split(list(row_blocks)) if row_blocks is not None else [matrix]
        ortho.append(torch.cat([reference_newton_schulz(p) * update_rms * math.sqrt(max(p.shape)) for p in parts]))
    ortho = torch.stack(ortho).reshape(update.shape)
    return weight * (1 - lr * weight_decay) - lr * ortho, momentum


@pytest.mark.parametrize("shape", [(256, 640), (640, 256), (4, 128, 384)])
@pytest.mark.parametrize("use_triton", [True, False])
def test_hybrid_newton_schulz_matches_fp64(shape, use_triton):
    torch.manual_seed(0)
    G = torch.randn(shape, device="cuda") * torch.logspace(-3, 0, shape[-1], device="cuda")
    out = hybrid_newton_schulz(G, use_triton=use_triton).double()
    ref = torch.stack([reference_newton_schulz(g) for g in G.reshape(-1, *shape[-2:])]).reshape(shape)
    # bf16 iterations: the error is bf16 rounding amplified by the iteration, not an algorithmic difference.
    assert (out - ref).norm() / ref.norm() < 2e-2
    # The singular values end close to 1.
    s = torch.linalg.svdvals(out.float())
    assert s.min() > 0.6 and s.max() < 1.3


def test_muon_step_matches_reference():
    torch.manual_seed(0)
    hyper = dict(lr=1e-2, mu=0.95, weight_decay=0.1)
    shapes = [
        ((96, 160), (32, 32, 32)),
        ((3, 128, 192), None),
        ((300, 100), None),
        ((24, 320), (4, 4, 16)),
        ((2, 128, 96), (64, 64)),
    ]
    weights = [torch.randn(s, device="cuda") for s, _ in shapes]
    momenta = [torch.zeros_like(w) for w in weights]
    refs = [(w.clone(), m.clone()) for w, m in zip(weights, momenta)]
    for step in range(3):
        grads = [torch.randn_like(w) * (step + 1) for w in weights]
        items = [MuonParam(w, row_blocks=b, **hyper) for w, (_, b) in zip(weights, shapes)]
        muon_step(items, grads, momenta, update_rms=0.18)
        for k, ((w_ref, m_ref), g, (_, blocks)) in enumerate(zip(refs, grads, shapes)):
            new_w, new_m = reference_muon(w_ref, g, m_ref, update_rms=0.18, row_blocks=blocks, **hyper)
            torch.testing.assert_close(momenta[k], new_m.float(), rtol=1e-6, atol=1e-6)
            delta, ref_delta = weights[k] - w_ref, new_w.float() - w_ref
            assert (delta - ref_delta).norm() / ref_delta.norm() < 2e-2
            refs[k] = (weights[k].clone(), momenta[k].clone())


def test_assign_owners_balances_costs():
    costs = [9.0, 7.0, 6.0, 5.0, 5.0, 4.0, 3.0, 1.0]
    owners = assign_owners(costs, 3)
    loads = [sum(c for c, o in zip(costs, owners) if o == r) for r in range(3)]
    assert sorted(loads) == [13.0, 13.0, 14.0]
    assert assign_owners([1.0, 1.0], 4) == [0, 1]
