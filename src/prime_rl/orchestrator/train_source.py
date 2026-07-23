"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). A v1 env serves the tasks the orchestrator loaded client-side: a
finite one as a shuffled table (reshuffled on cursor exhaustion), an
infinite one (``num_tasks is None``) straight off its generator — every
pull is a fresh task and there are no epochs to shuffle. A legacy env's
dataset lives on its server, so it serves shuffled task *indices*."""

from __future__ import annotations

import random
from collections.abc import Iterator

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example(available_permits)`` picks a weighted-RR env and
    returns its next example (or ``None`` when the env's per-call permit
    cost doesn't fit — the dispatch loop retries when permits free up).
    Returned dicts carry ``env_name`` + ``task_idx`` (+ ``task`` for v1 envs,
    whose data is shipped to the env server at dispatch)."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        # A finite env's shuffled example table; ``None`` for an infinite env,
        # whose generator (``self.iters``) is pulled per example.
        self.examples: dict[str, list[dict] | None] = {}
        self.iters: dict[str, Iterator[vf.Task]] = {}
        self.cursors: dict[str, int] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            if env.tasks is None:  # legacy: sample over the index range from info()
                rows: list[dict] = [{"task_idx": i, "env_name": env.name} for i in range(env.num_tasks)]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            elif env.num_tasks is None:  # infinite: pull the generator per example
                self.examples[env.name] = None
                self.iters[env.name] = env.tasks
            else:
                rows = [{"task_idx": task.data.idx, "task": task, "env_name": env.name} for task in env.tasks]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        self.weights: list[float] = [float(e.config.ratio) for e in self.envs]

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        if self.env_costs[env_name] > available_permits:
            return None
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if rows is None:  # infinite env: pull the next generated task
            task = next(self.iters[env_name])
            return {"task_idx": task.data.idx, "task": task, "env_name": env_name}
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
