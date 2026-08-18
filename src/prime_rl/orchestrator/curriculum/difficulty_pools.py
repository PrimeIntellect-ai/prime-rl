"""Difficulty-pool task sampling."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.base import CurriculumResult, TaskSampler


@dataclass(frozen=True)
class DifficultyPool:
    """An inclusive reward threshold and a relative per-task sampling weight."""

    threshold: float
    weight: float


DEFAULT_POOLS = {
    "hard": DifficultyPool(threshold=0.25, weight=0.2),
    "normal": DifficultyPool(threshold=0.75, weight=1.0),
    "easy": DifficultyPool(threshold=1.0, weight=0.2),
}


class DifficultyPools(TaskSampler):
    """Weight finite tasks by a pool derived from their latest group mean.

    Each pool's threshold is its inclusive maximum reward; the final pool is
    the catch-all. Unseen tasks have neutral weight, so observations affect
    sampling immediately without requiring a full taskset pass.
    """

    def __init__(
        self,
        tasks: Sequence[vf.Task] | Iterator[vf.Task],
        *,
        pools: Mapping[str, DifficultyPool | Mapping[str, float]] | None = None,
        seed: int = 42,
    ) -> None:
        if not isinstance(tasks, Sequence):
            raise ValueError("DifficultyPools requires a finite taskset")
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("DifficultyPools requires at least one task")
        keys = [task.key for task in self.tasks]
        duplicates = {key for key, count in Counter(keys).items() if count > 1}
        if duplicates:
            raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
        self.tasks_by_key = dict(zip(keys, self.tasks))
        self.rng = random.Random(seed)
        configured_pools = DEFAULT_POOLS if pools is None else pools
        self.pools = {
            name: pool if isinstance(pool, DifficultyPool) else DifficultyPool(**pool)
            for name, pool in configured_pools.items()
        }
        if not self.pools:
            raise ValueError("DifficultyPools requires at least one pool")
        if any(pool.weight <= 0 for pool in self.pools.values()):
            raise ValueError("Difficulty pool weights must be positive")
        ordered = sorted(self.pools.items(), key=lambda item: item[1].threshold)
        if len({pool.threshold for _, pool in ordered}) != len(ordered):
            raise ValueError("Difficulty pool thresholds must be unique")
        self._ordered_pools = tuple(ordered)
        self.task_rewards: dict[str, float] = {}

    def task_pool(self, task_key: str) -> str | None:
        """Return the task's current pool, or ``None`` until it has a score."""
        score = self.task_rewards.get(task_key)
        if score is None:
            return None
        for name, pool in self._ordered_pools:
            if score <= pool.threshold:
                return name
        return self._ordered_pools[-1][0]

    def sample(self) -> vf.Task:
        ceiling = max(pool.weight for pool in self.pools.values())
        if len(self.task_rewards) < len(self.tasks):
            ceiling = max(1.0, ceiling)
        while True:
            task = self.rng.choice(self.tasks)
            pool = self.task_pool(task.key)
            weight = 1.0 if pool is None else self.pools[pool].weight
            if self.rng.random() * ceiling < weight:
                return task

    def observe(self, result: CurriculumResult) -> None:
        rewards = [rollout.reward for rollout in result.rollouts if not rollout.has_error and rollout.agent.trainable]
        if not rewards:
            return
        self.task_rewards[result.task_key] = sum(rewards) / len(rewards)

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng": self.rng.getstate(),
            "task_rewards": dict(self.task_rewards),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng.setstate(state_dict["rng"])
        self.task_rewards = dict(state_dict["task_rewards"])

    def metrics(self) -> dict[str, float]:
        occupancy = dict.fromkeys(self.pools, 0)
        for task_key in self.task_rewards:
            pool = self.task_pool(task_key)
            if pool is not None:
                occupancy[pool] += 1
        return {
            "pool/unseen": float(len(self.tasks_by_key) - len(self.task_rewards)),
            **{f"pool/{name}": float(count) for name, count in occupancy.items()},
        }
