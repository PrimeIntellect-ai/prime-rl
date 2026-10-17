import pytest
import torch

from prime_rl.configs.trainer import MuonConfig
from prime_rl.trainer.optim.muon import Muon, newton_schulz

COEFFICIENTS = MuonConfig().newton_schulz_coefficients


@pytest.mark.parametrize(
    ("coefficients", "low", "high"),
    [(COEFFICIENTS, 0.6, 1.25), ([(3.4445, -4.7750, 2.0315)] * 8 + [(2.0, -1.5, 0.5)] * 2, 0.99, 1.01)],
)
def test_newton_schulz_orthogonalizes(coefficients, low, high):
    G = torch.randn(3, 64, 48, generator=torch.Generator().manual_seed(0))
    singular_values = torch.linalg.svdvals(newton_schulz(G, tuple(coefficients)).float())
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

    Muon(
        [dict(params=packed)],
        lr=0.1,
        newton_schulz_coefficients=COEFFICIENTS,
        weight_decay=0.1,
        matrix_partitions=dict(zip(packed, shapes.values())),
    ).step()
    Muon([dict(params=split)], lr=0.1, newton_schulz_coefficients=COEFFICIENTS, weight_decay=0.1).step()

    for p, sizes in zip(packed, shapes.values()):
        for part, reference in zip(p.split(sizes, -2), [split.pop(0) for _ in sizes]):
            torch.testing.assert_close(part, reference, atol=2e-3, rtol=0)
