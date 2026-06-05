"""Sampling strategies for training rollouts.

Two seams sit between the train envs and the dispatcher:

- ``EnvMixStrategy`` (global) — decides *which* env to draw from next.
- ``SampleStrategy`` (per-env) — decides *what* example that env serves next,
  and (via ``observe``) can learn from finished, scored groups. Each env owns
  its own ``SampleStrategy`` instance, so it can hold dataset + per-env state
  (cursor today; curriculum / replay buffers later).

The defaults (``WeightedRoundRobin`` + ``ShuffledCursorSampler``) reproduce the
previous ``TrainSource`` behavior: a weighted round-robin over per-env datasets
that are each shuffled once and walked with a reshuffling cursor.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import TrainRollout


class SampleStrategy(ABC):
    """Per-env example selection. One stateful instance per env, alive for the
    whole run. ``next`` returns the next example dict (carrying ``env_name`` +
    ``example_id``); ``observe`` is the feedback hook for stateful strategies."""

    @abstractmethod
    def next(self) -> dict:
        """Return the next example for this env."""
        ...

    def observe(self, group: list[TrainRollout]) -> None:
        """Called with one finished, scored group of this env's rollouts (after
        advantages are assigned). Default is a no-op; stateful strategies
        (curriculum, replay) override this to learn from outcomes."""
        return


class ShuffledCursorSampler(SampleStrategy):
    """Default sampler: shuffle the env's rows once, walk a cursor, reshuffle on
    exhaustion (infinite pull)."""

    def __init__(self, rows: list[dict], *, seed: int | None) -> None:
        if not rows:
            raise ValueError("ShuffledCursorSampler needs at least one example")
        self._rng = random.Random(seed)
        self._rows = list(rows)
        self._rng.shuffle(self._rows)
        self._cursor = 0

    @property
    def dataset_size(self) -> int:
        return len(self._rows)

    def next(self) -> dict:
        if self._cursor >= len(self._rows):
            self._rng.shuffle(self._rows)
            self._cursor = 0
        row = self._rows[self._cursor]
        self._cursor += 1
        return row


class EnvMixStrategy(ABC):
    """Global: which env to draw from next. ``pick`` returns an env name."""

    @abstractmethod
    def pick(self) -> str:
        """Return the env name to sample from next."""
        ...


class WeightedRoundRobin(EnvMixStrategy):
    """Default env mix: weighted random choice over env names. Weights are the
    configured per-env ratios (when all set) or per-env dataset sizes."""

    def __init__(self, env_names: list[str], weights: list[float], *, seed: int | None) -> None:
        if not env_names:
            raise ValueError("WeightedRoundRobin needs at least one env")
        self._rng = random.Random(seed)
        self._env_names = list(env_names)
        self._weights = list(weights)

    def pick(self) -> str:
        return self._rng.choices(self._env_names, weights=self._weights, k=1)[0]
