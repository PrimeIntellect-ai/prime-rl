import pytest
import torch

from prime_rl.trainer.optim.muon import NEWTON_SCHULZ_COEFFICIENTS, Muon, newton_schulz


@pytest.mark.parametrize(("coefficients", "low", "high"), [("muon", 0.6, 1.25), ("deepseek_v4", 0.99, 1.01)])
def test_newton_schulz_orthogonalizes(coefficients, low, high):
    G = torch.randn(3, 64, 48, generator=torch.Generator().manual_seed(0))
    singular_values = torch.linalg.svdvals(newton_schulz(G, NEWTON_SCHULZ_COEFFICIENTS[coefficients]).float())
    assert singular_values.min() > low and singular_values.max() < high


def test_packed_parameter_steps_like_its_matrices():
    generator = torch.Generator().manual_seed(0)
    # Unequal partitions (fused qkv) and equal ones over a batch of matrices (fused gate_up experts)
    shapes = {(48, 32): (32, 8, 8), (4, 64, 32): (32, 32)}
    packed = [torch.nn.Parameter(torch.randn(shape, generator=generator)) for shape in shapes]
    split = [
        torch.nn.Parameter(part.detach().clone())
        for p, sizes in zip(packed, shapes.values())
        for part in p.split(sizes, -2)
    ]
    grads = [torch.randn(p.shape, generator=generator) for p in packed]
    for p, g in zip(packed, grads):
        p.grad = g
    for p, g in zip(split, [part for g, sizes in zip(grads, shapes.values()) for part in g.split(sizes, -2)]):
        p.grad = g.clone()

    Muon([dict(params=packed)], lr=0.1, weight_decay=0.1, matrix_partitions=dict(zip(packed, shapes.values()))).step()
    Muon([dict(params=split)], lr=0.1, weight_decay=0.1).step()

    for p, sizes in zip(packed, shapes.values()):
        for part, reference in zip(p.split(sizes, -2), [split.pop(0) for _ in sizes]):
            torch.testing.assert_close(part, reference, atol=2e-3, rtol=0)
