import random

import pytest
import torch

from prime_rl.trainer.gradient_normalization import GradientNormalization

_rng = random.Random(0)
RANDOM_COUNTS = [
    {name: _rng.choice([0.0, _rng.uniform(1.0, 1e9)]) for name in ("rl", "ce", "ref_kl")} for _ in range(4)
]


@pytest.mark.parametrize("placement", ["grads", "loss"])
@pytest.mark.parametrize(
    "counts",
    [
        {"rl": 1.0e8, "ce": 3.0e5, "ref_kl": 0.0},
        {"rl": 0.0, "ce": 0.0, "ref_kl": 0.0},
        {"ce": 12345.0},
        {"rl": 7.5, "ce": 0.0, "ref_kl": 2.0e6},
        *RANDOM_COUNTS,
    ],
)
def test_gradient_normalization_divides_each_component_by_its_count(counts, placement):
    fsdp_divide_factor = 8
    rng = random.Random(0)
    loss_sums = {name: rng.uniform(-1e3, 1e3) if count > 0 else 0.0 for name, count in counts.items()}
    expected = sum(loss_sums[name] / (count if count > 0 else 1.0) for name, count in counts.items())

    norm = GradientNormalization.from_counts(counts, fsdp_divide_factor, placement)
    backward_loss = sum(norm.loss_weights[name] * loss_sum for name, loss_sum in loss_sums.items())

    assert backward_loss * norm.grad_scale == pytest.approx(fsdp_divide_factor * expected, rel=1e-12)
    logged = norm.logged_loss(torch.tensor(backward_loss, dtype=torch.float64))
    assert logged.item() == pytest.approx(expected, rel=1e-12)
    if placement == "grads":
        assert min(norm.loss_weights.values()) == 1.0
