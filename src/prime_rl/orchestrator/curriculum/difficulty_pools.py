"""Difficulty-pool task sampling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import verifiers.v1 as vf

from prime_rl.orchestrator.curriculum.base import CurriculumResult, TaskSampler


class DifficultyPools(TaskSampler):
    """Sample finite tasks through named pools based on the latest group mean.

    Unsampled tasks are chosen first. Afterwards, a nonempty pool is selected
    by its configured weight and a task is sampled uniformly from that pool.
    ``thresholds`` maps each pool name to its inclusive maximum mean reward;
    the final pool is the catch-all.
    """

    def __init__(
        self,
        tasks: Sequence[vf.Task] | Iterator[vf.Task],
        *,
        thresholds: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(tasks, seed=seed)
        if self.tasks is None:
            raise ValueError("DifficultyPools requires a finite taskset")
        self.thresholds = {"hard": 0.25, "medium": 0.75, "easy": 1.0} if thresholds is None else thresholds
        self.weights = dict.fromkeys(self.thresholds, 1.0) if weights is None else weights
        if not self.thresholds:
            raise ValueError("DifficultyPools requires at least one pool")
        if self.thresholds.keys() != self.weights.keys():
            raise ValueError("Difficulty pool thresholds and weights must name the same pools")
        if any(weight <= 0 for weight in self.weights.values()):
            raise ValueError("Difficulty pool weights must be positive")
        ordered = sorted(self.thresholds.items(), key=lambda item: item[1])
        if len({maximum for _, maximum in ordered}) != len(ordered):
            raise ValueError("Difficulty pool thresholds must be unique")
        self._ordered_pools = tuple(ordered)
        self.sampled_task_keys: set[str] = set()
        self.task_rewards: dict[str, float] = {}

    def _pool(self, task_key: str) -> str:
        score = self.task_rewards[task_key]
        for name, maximum in self._ordered_pools:
            if score <= maximum:
                return name
        return self._ordered_pools[-1][0]

    def sample(self) -> vf.Task:
        unsampled = [task for task in self.tasks_by_key.values() if task.key not in self.sampled_task_keys]
        if unsampled:
            task = self.rng.choice(unsampled)
            self.sampled_task_keys.add(task.key)
            return task

        if not self.task_rewards:
            return self.rng.choice(tuple(self.tasks_by_key.values()))

        tasks_by_pool: dict[str, list[vf.Task]] = defaultdict(list)
        for task_key, task in self.tasks_by_key.items():
            if task_key in self.task_rewards:
                tasks_by_pool[self._pool(task_key)].append(task)
        nonempty = [name for name in self.weights if tasks_by_pool[name]]
        pool = self.rng.choices(nonempty, weights=[self.weights[name] for name in nonempty], k=1)[0]
        return self.rng.choice(tasks_by_pool[pool])

    def observe(self, result: CurriculumResult) -> None:
        rewards = [rollout.reward for rollout in result.rollouts if not rollout.has_error and rollout.agent.trainable]
        if not rewards:
            return
        self.task_rewards[result.task_key] = sum(rewards) / len(rewards)

    def state_dict(self) -> dict[str, Any]:
        return super().state_dict() | {
            "sampled_task_keys": sorted(self.sampled_task_keys),
            "task_rewards": dict(self.task_rewards),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.task_rewards = dict(state_dict["task_rewards"])
        self.sampled_task_keys = set(state_dict.get("sampled_task_keys", self.task_rewards))

    def metrics(self) -> dict[str, float]:
        occupancy = dict.fromkeys(self.thresholds, 0)
        for task_key in self.task_rewards:
            occupancy[self._pool(task_key)] += 1
        return {
            "pool/unseen": float(len(self.tasks_by_key) - len(self.sampled_task_keys)),
            **{f"pool/{name}": float(count) for name, count in occupancy.items()},
        }
