import pytest
import torch

from prime_rl.configs.es import ESAlgorithmConfig
from prime_rl.trainer.es.candidates import Candidate, estimate_gradient, make_candidates, noise_like, normalize_rewards


def test_one_sided_gradient_without_reward_normalization():
    theta = torch.zeros(8)
    candidates = [Candidate(idx=0, seed=11), Candidate(idx=1, seed=13)]
    rewards = {0: 2.0, 1: -1.0}

    grad = estimate_gradient(theta, candidates, rewards, sigma=0.1, normalization="none", mirrored=False)

    expected = ((2.0 / 0.1) * noise_like(theta, 11) - (1.0 / 0.1) * noise_like(theta, 13)) / 2.0
    assert torch.allclose(grad, expected)


def test_equal_rewards_zscore_to_zero_update():
    theta = torch.zeros(8)
    candidates = [Candidate(idx=0, seed=11), Candidate(idx=1, seed=13)]
    rewards = {0: 1.0, 1: 1.0}

    grad = estimate_gradient(theta, candidates, rewards, sigma=0.1, normalization="zscore", mirrored=False)

    assert torch.count_nonzero(grad) == 0


def test_mirrored_gradient_uses_pairwise_reward_difference():
    theta = torch.zeros(8)
    candidates = make_candidates(base_seed=7, step=3, population_size=4, mirrored=True)
    rewards = {candidate.idx: 3.0 if candidate.sign > 0 else 1.0 for candidate in candidates}

    grad = estimate_gradient(theta, candidates, rewards, sigma=0.5, normalization="zscore", mirrored=True)

    expected = torch.zeros_like(theta)
    for candidate in candidates:
        if candidate.sign > 0:
            expected += noise_like(theta, candidate.seed) * ((3.0 - 1.0) / (2.0 * 0.5))
    expected /= 2.0
    assert torch.allclose(grad, expected)


def test_normalize_rewards_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported reward normalization"):
        normalize_rewards([1.0, 2.0], "rank")


def test_mirrored_population_requires_even_size():
    with pytest.raises(ValueError, match="population_size must be even"):
        ESAlgorithmConfig(population_size=3, mirrored=True)
