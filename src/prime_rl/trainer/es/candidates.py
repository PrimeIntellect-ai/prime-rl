from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Candidate:
    idx: int
    seed: int
    sign: int = 1

    @property
    def name_suffix(self) -> str:
        if self.sign > 0:
            return f"cand_{self.idx:04d}"
        return f"cand_{self.idx:04d}_minus"


def make_candidates(base_seed: int, step: int, population_size: int, mirrored: bool) -> list[Candidate]:
    rng = np.random.default_rng(base_seed + step)
    if mirrored:
        seeds = [int(rng.integers(0, np.iinfo(np.uint32).max)) for _ in range(population_size // 2)]
        candidates: list[Candidate] = []
        for idx, seed in enumerate(seeds):
            candidates.append(Candidate(idx=2 * idx, seed=seed, sign=1))
            candidates.append(Candidate(idx=2 * idx + 1, seed=seed, sign=-1))
        return candidates
    return [
        Candidate(idx=idx, seed=int(rng.integers(0, np.iinfo(np.uint32).max)), sign=1) for idx in range(population_size)
    ]


def noise_like(theta: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=theta.device)
    generator.manual_seed(seed)
    return torch.randn(theta.shape, generator=generator, device=theta.device, dtype=torch.float32)


def normalize_rewards(rewards: list[float], mode: str) -> np.ndarray:
    arr = np.asarray(rewards, dtype=np.float32)
    if mode == "none":
        return arr
    mean = float(arr.mean()) if arr.size else 0.0
    centered = arr - mean
    if mode == "centered":
        return centered
    if mode != "zscore":
        raise ValueError(f"Unsupported reward normalization: {mode}")
    std = float(arr.std())
    if std < 1e-8:
        return np.zeros_like(arr)
    return centered / std


def estimate_gradient(
    theta: torch.Tensor,
    candidates: list[Candidate],
    rewards: dict[int, float],
    sigma: float,
    normalization: str,
    mirrored: bool,
) -> torch.Tensor:
    grad = torch.zeros_like(theta)
    if mirrored:
        by_seed: dict[int, dict[int, float]] = {}
        for candidate in candidates:
            by_seed.setdefault(candidate.seed, {})[candidate.sign] = rewards[candidate.idx]
        used_pairs = 0
        for seed, pair in by_seed.items():
            if 1 not in pair or -1 not in pair:
                continue
            grad.add_(noise_like(theta, seed), alpha=(pair[1] - pair[-1]) / (2.0 * sigma))
            used_pairs += 1
        if used_pairs:
            grad.div_(float(used_pairs))
        return grad

    ordered_rewards = [rewards[c.idx] for c in candidates]
    scores = normalize_rewards(ordered_rewards, normalization)
    for candidate, score in zip(candidates, scores, strict=True):
        grad.add_(noise_like(theta, candidate.seed), alpha=float(score) / sigma)
    if candidates:
        grad.div_(float(len(candidates)))
    return grad
