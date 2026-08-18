"""Difficulty-pool task sampling."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.samplers.base import TaskSampler

if TYPE_CHECKING:
    from prime_rl.configs.orchestrator import DifficultyPoolSamplerConfig
    from prime_rl.orchestrator.types import Rollout


class DifficultyPoolSampler(TaskSampler):
    """Weight finite tasks by a pool derived from their latest group mean.

    Each pool's threshold is its inclusive maximum reward; the final pool is
    the catch-all. Unseen tasks have neutral weight, so observations affect
    sampling immediately without requiring a full taskset pass.
    """

    def __init__(
        self,
        config: DifficultyPoolSamplerConfig,
        tasks: Sequence[vf.Task] | Iterator[vf.Task],
    ) -> None:
        if not isinstance(tasks, Sequence):
            raise ValueError("DifficultyPoolSampler requires a finite taskset")
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("DifficultyPoolSampler requires at least one task")
        keys = [task.key for task in self.tasks]
        duplicates = {key for key, count in Counter(keys).items() if count > 1}
        if duplicates:
            raise ValueError(f"Task keys must be unique within a taskset: {sorted(duplicates)}")
        self.tasks_by_key = dict(zip(keys, self.tasks))
        self.rng = random.Random(config.seed)
        self.pools = config.pools
        ordered = sorted(self.pools.items(), key=lambda item: item[1].threshold)
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

    def __next__(self) -> vf.Task:
        weights = []
        for task in self.tasks:
            pool = self.task_pool(task.key)
            weights.append(1.0 if pool is None else self.pools[pool].weight)
        if not any(weights):
            raise RuntimeError("DifficultyPoolSampler has no tasks in a pool with positive weight")
        return self.rng.choices(self.tasks, weights=weights, k=1)[0]

    def observe(self, group: list[Rollout]) -> None:
        rewards = [rollout.reward for rollout in group if not rollout.has_error and rollout.agent.trainable]
        if not rewards:
            return
        task_key = group[0].task.key
        if task_key is None:
            raise ValueError("A finalized group is missing Task.key")
        self.task_rewards[task_key] = sum(rewards) / len(rewards)

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
