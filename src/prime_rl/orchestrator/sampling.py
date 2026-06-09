"""Sampling strategies for training rollouts.

``EnvMixStrategy`` (global) decides *which* env to draw from next — a swappable
seam between the train envs and the dispatcher. The default
(``WeightedRoundRobin``) reproduces the previous ``TrainSource`` behavior: a
weighted random choice over env names, weighted by configured ``ratio`` (when
every env sets one) or per-env dataset size.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod


class EnvMixStrategy(ABC):
    """Global: which env to draw from next. ``pick`` returns an env name."""

    @abstractmethod
    def pick(self) -> str:
        """Return the env name to sample from next."""
        ...


class WeightedRoundRobin(EnvMixStrategy):
    """Default env mix: weighted random choice over env names. Weights are the
    configured per-env ratios (when all set) or per-env dataset sizes.

    Draws from the caller's RNG so env selection stays in the same stream as
    ``TrainSource``'s dataset shuffles — the example sequence is unchanged.
    """

    def __init__(self, env_names: list[str], weights: list[float], *, rng: random.Random) -> None:
        if not env_names:
            raise ValueError("WeightedRoundRobin needs at least one env")
        self._rng = rng
        self._env_names = list(env_names)
        self._weights = list(weights)

    def pick(self) -> str:
        return self._rng.choices(self._env_names, weights=self._weights, k=1)[0]
