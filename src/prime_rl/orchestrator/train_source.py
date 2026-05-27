"""TrainSource: infinite pull of training examples.

Weighted round-robin across train envs (configured ``ratio`` if every env
sets one, otherwise weight by dataset size). The dispatcher calls
``next_example(available_permits)`` whenever it has permits free and
``dispatcher.DispatcherMode`` is ``PREFER_TRAIN`` — there's no notion of
a finite epoch, the source just reshuffles per-env rows when it reaches
the end.

The dispatcher still owns scheduling priority (``dispatcher.DispatcherMode``)
and capacity (``max_inflight`` counter); this source owns the per-env
permit cost lookup and answers "what's the next training example to
schedule that fits in ``available_permits``?". Mirrors ``EvalSource
.next_example`` so the dispatcher hits both sources through a single
symmetric API.
"""

from __future__ import annotations

import random

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """Infinite source of training examples — weighted round-robin across envs.

    ``next_example(available_permits)`` picks a weighted-RR env, returns
    its next example (mutating its cursor + reshuffling on exhaustion),
    or ``None`` when the picked env's per-call permit cost doesn't fit in
    ``available_permits`` — the dispatch loop retries on the next
    iteration once permits free up. Returned dicts carry ``env_name`` and
    an ``example_id`` (the latter guaranteed by verifiers).
    """

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None, group_size: int) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.examples: dict[str, list[dict]] = {}
        self.cursors: dict[str, int] = {}
        # Per-env permit cost for opening a fresh group. Group-scoring envs
        # dispatch the whole group as a single task, so they need
        # ``group_size`` permits up front; per-rollout envs dispatch one at
        # a time and only need 1 permit to get going.
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
            self.env_costs[env.name] = group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        configured_ratios = [e.config.ratio for e in self.envs]
        if all(r is not None for r in configured_ratios):
            self.weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            # "ratio unset → weight by num examples" natural distribution.
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
