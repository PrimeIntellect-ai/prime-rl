"""TrainSource: infinite pull of training examples.

Weighted round-robin across train envs (configured ``ratio`` if every env
sets one, otherwise weight by dataset size). The dispatcher calls
``next_example()`` whenever it has permits free and ``dispatcher.SchedMode``
is ``PREFER_TRAIN`` — there's no notion of a finite epoch, the source just
reshuffles per-env rows when it reaches the end.

The dispatcher still owns scheduling priority (``dispatcher.SchedMode``), capacity
(``max_inflight`` counter), and per-env cost lookups; this source just answers "what's
the next training example to schedule?".
"""

from __future__ import annotations

import random

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """Infinite source of training examples — weighted round-robin across envs.

    ``next_example()`` always succeeds and returns a dict carrying the env name
    in ``env_name`` and a stable ``example_id`` (backfilled if the dataset row
    didn't already have one).
    """

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.examples: dict[str, list[dict]] = {}
        self.cursors: dict[str, int] = {}
        for env in self.envs:
            rows: list[dict] = []
            for row in env.get_dataset(seed=seed):
                ex = dict(row)
                ex["env_name"] = env.name
                rows.append(ex)
            self.rng.shuffle(rows)
            self.examples[env.name] = rows
            self.cursors[env.name] = 0

        self.env_names = [e.name for e in self.envs]
        configured_ratios = [e.config.ratio for e in self.envs]
        if all(r is not None for r in configured_ratios):
            self.weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            # "ratio unset → weight by num examples" natural distribution.
            self.weights = [float(len(self.examples[name])) for name in self.env_names]

    def next_example(self) -> dict:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
