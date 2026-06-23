"""TrainSource: weighted round-robin across train envs, infinite pull.

Env selection is delegated to a swappable ``EnvMixStrategy`` (default:
weighted round-robin by configured ``ratio`` when every env sets one, else by
per-env dataset size); example selection stays here (a reshuffling cursor per
env). ``next_example`` reshuffles on cursor exhaustion. Returned dicts carry
``env_name`` + ``example_id``.
"""

from __future__ import annotations

import random

from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.sampling import WeightedRoundRobin


class TrainSource:
    """``next_example(available_permits)`` picks an env via the mix strategy and
    returns its next example (or ``None`` when the env's per-call permit cost
    doesn't fit — the dispatch loop retries when permits free up). Returned
    dicts carry ``env_name`` + ``example_id``."""

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
            rows: list[dict] = []
            for row in env.get_dataset(seed=seed):
                ex = dict(row)
                ex["env_name"] = env.name
                rows.append(ex)
            self.rng.shuffle(rows)
            self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        env_names = [e.name for e in self.envs]
        configured_ratios = [e.config.ratio for e in self.envs]
        if all(r is not None for r in configured_ratios):
            weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            weights = [float(len(self.examples[name])) for name in env_names]
        # Shares ``self.rng`` so env selection draws from the same stream as the
        # dataset shuffles above — the example sequence matches the pre-seam path.
        self.env_mix = WeightedRoundRobin(env_names, weights, rng=self.rng)

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.env_mix.pick()
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
