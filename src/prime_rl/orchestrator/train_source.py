"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights default to configured ``ratio`` (when every env sets one) or to
per-env dataset size. A finite env reshuffles its index range on cursor
exhaustion; an unbounded one (the server reports no ``num_tasks``) streams a
monotonically increasing ``task_idx`` instead."""

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
        self.bounded: dict[str, bool] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            # The orchestrator never loads the env. A finite env reports ``num_tasks``, so we
            # sample over its index range (shuffled, reshuffled each epoch). An unbounded one
            # (``num_tasks is None``) has no range — ``next_example`` streams a monotonically
            # increasing ``task_idx``; the taskset's own generator (e.g. RNG-seeded) varies it.
            self.bounded[env.name] = env.num_tasks is not None
            if env.num_tasks is not None:
                rows = [{"task_idx": i, "env_name": env.name} for i in range(env.num_tasks)]
                self.rng.shuffle(rows)
                self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        configured_ratios = [e.config.ratio for e in self.envs]
        if all(r is not None for r in configured_ratios):
            self.weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        elif all(self.bounded[name] for name in self.env_names):
            self.weights = [float(len(self.examples[name])) for name in self.env_names]
        else:
            unbounded = [n for n in self.env_names if not self.bounded[n]]
            raise ValueError(
                f"unbounded train env(s) {unbounded} have no dataset size to weight by; "
                "set an explicit `ratio` on every train env"
            )

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        if self.env_costs[env_name] > available_permits:
            return None
        cursor = self.cursors[env_name]
        if not self.bounded[env_name]:
            # Unbounded: hand out the next task_idx and advance forever (no epoch / reshuffle).
            self.cursors[env_name] = cursor + 1
            return {"task_idx": cursor, "env_name": env_name}
        rows = self.examples[env_name]
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
