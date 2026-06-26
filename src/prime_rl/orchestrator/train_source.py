"""TrainSource: weighted env picker, infinite pull.

The orchestrator never addresses tasks — the env server hands out the next (shuffled,
epoch-looped) task on each pull. TrainSource only chooses *which* env to pull from, weighted
by each env's ``ratio`` (default 1.0 → equal parts)."""

from __future__ import annotations

import random

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example(available_permits)`` picks a ``ratio``-weighted env and returns
    ``{"env_name": ...}`` (or ``None`` when the picked env's group doesn't fit the available
    permits — the dispatch loop retries when permits free up). The task itself is chosen by the
    env server on pull; its idx rides back on the returned Trace."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")
        self.env_names = [e.name for e in self.envs]
        # Each env's `ratio` is its relative weight (default 1.0 → equal parts); `random.choices`
        # normalizes to probabilities. No dataset-size weighting (an unbounded env has none).
        self.weights: list[float] = [float(e.config.ratio) for e in self.envs]
        # Permits to open a group: a group-scored env runs as one indivisible `run_group`
        # (group_size at once); a non-group env opens with one `run_rollout` (1) and fills the
        # rest per-rollout via the dispatcher's continue-group path.
        self.env_costs: dict[str, int] = {
            e.name: (e.config.group_size if e.requires_group_scoring else 1) for e in self.envs
        }

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        if self.env_costs[env_name] > available_permits:
            return None
        return {"env_name": env_name}
