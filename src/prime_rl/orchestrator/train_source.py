"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). A finite env serves a shuffled task-index table, reshuffled on
cursor exhaustion; an infinite env (``num_tasks is None``) streams a
monotonic ``task_idx`` — the server generates tasks on demand, so every
pull is a fresh task and there are no epochs to shuffle."""

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

        # A finite env's shuffled index table; ``None`` for an infinite env,
        # whose cursor alone is the (monotonic) task_idx stream.
        self.examples: dict[str, list[dict] | None] = {}
        self.cursors: dict[str, int] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            # The orchestrator never loads the env: sample over the task-index
            # range the server reported via info() (num_tasks; None = infinite).
            if env.num_tasks is None:
                self.examples[env.name] = None
            else:
                rows: list[dict] = [{"task_idx": i, "env_name": env.name} for i in range(env.num_tasks)]
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
        if rows is None:  # infinite env: the cursor is the task_idx
            self.cursors[env_name] = cursor + 1
            return {"task_idx": cursor, "env_name": env_name}
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
