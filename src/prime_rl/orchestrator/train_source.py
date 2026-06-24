"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights default to configured ``ratio`` (when every env sets one) or to
per-env dataset size. ``next_example`` reshuffles on cursor exhaustion."""

from __future__ import annotations

import random

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example(available_permits)`` picks a weighted-RR env and
    returns its next example (or ``None`` when the env's per-call permit
    cost doesn't fit — the dispatch loop retries when permits free up).
    Returned dicts carry ``env_name`` + ``task_idx``."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.examples: dict[str, list[dict]] = {}
        self.cursors: dict[str, int] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            # The orchestrator never loads the env: sample over the task-index
            # range the server reported via info() (num_tasks).
            rows: list[dict] = [{"task_idx": i, "env_name": env.name} for i in range(env.num_tasks)]
            self.rng.shuffle(rows)
            self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        configured_ratios = [e.config.ratio for e in self.envs]
        if all(r is not None for r in configured_ratios):
            self.weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            self.weights = [float(len(self.examples[name])) for name in self.env_names]

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        if self.env_costs[env_name] > available_permits:
            return None
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
