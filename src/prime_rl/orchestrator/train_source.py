"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). An env serves the tasks the orchestrator loaded client-side: a
finite one as a shuffled table (reshuffled with ``seed=epoch`` on cursor
exhaustion), an infinite one (``num_tasks is None``) straight off its
generator — every pull is a fresh task and there are no epochs to shuffle."""

from __future__ import annotations

import random
from collections.abc import Iterator

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example()`` picks a weighted-RR env and returns its next
    example. Returned dicts carry ``env_name`` + ``task``, whose data is
    shipped to the env server at dispatch.

    The data position round-trips through a checkpoint via ``state_dict()``
    / ``load_state_dict()``: per-env ``{epoch, cursor}`` plus the env-choice
    RNG state, making the dispatch sequence reproducible across resumes. For
    a finite env, epochs are 1-indexed and seed that epoch's shuffle; for an
    infinite env the epoch stays 1 and the cursor counts generator pulls —
    restored as a counter only, never by fast-forwarding the generator, since
    an infinite taskset may be backed by live external state whose every pull
    is a real draw with side effects. Cursors advance at dispatch time (ahead
    of shipped batches), so a finite env's resume skips the tasks that were in
    flight at checkpoint time."""

    def __init__(self, train_envs: TrainEnvs) -> None:
        self.rng = random.Random(42)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        # A finite env's example table in canonical order (each epoch's shuffle
        # starts from this); ``None`` for an infinite env, whose generator
        # (``self.iters``) is pulled per example.
        self.base_rows: dict[str, list[dict] | None] = {}
        self.examples: dict[str, list[dict] | None] = {}
        self.iters: dict[str, Iterator[vf.Task]] = {}
        self.epochs: dict[str, int] = {}
        self.cursors: dict[str, int] = {}
        for env in self.envs:
            assert env.tasks is not None, f"env {env.name} not started"
            if env.num_tasks is None:  # infinite: pull the generator per example
                rows: list[dict] | None = None
                self.iters[env.name] = env.tasks
            else:
                rows = [{"task": task, "env_name": env.name} for task in env.tasks]
            self.base_rows[env.name] = rows
            self.epochs[env.name] = 1
            self.cursors[env.name] = 0
            self.examples[env.name] = self._shuffle(env.name)

        self.env_names = [e.name for e in self.envs]
        self.weights: list[float] = [float(e.config.ratio) for e in self.envs]

    def _shuffle(self, env_name: str) -> list[dict] | None:
        """The env's example table shuffled for its current epoch — a pure
        function of (canonical order, epoch), so a restored position replays
        the exact epoch permutation."""
        rows = self.base_rows[env_name]
        if rows is None:
            return None
        rows = rows.copy()
        random.Random(self.epochs[env_name]).shuffle(rows)
        return rows

    def state_dict(self) -> dict:
        """Env-choice RNG state + per-env ``{epoch, cursor}``."""
        return {
            "rng": self.rng.getstate(),
            "envs": {name: {"epoch": self.epochs[name], "cursor": self.cursors[name]} for name in self.epochs},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.rng.setstate(state_dict["rng"])
        for name, position in state_dict["envs"].items():
            if name not in self.base_rows:
                continue
            self.epochs[name] = position["epoch"]
            self.cursors[name] = position["cursor"]
            if self.base_rows[name] is not None:
                self.examples[name] = self._shuffle(name)
            # An infinite env restores its cursor as a counter only — never by
            # fast-forwarding the generator. A fresh iterator is not the old one:
            # replay is exact only for a pure deterministic generator, and for a
            # taskset backed by live external state (a task server, a curriculum
            # stream) every next() is a real draw with side effects, so replaying
            # a thousands-deep cursor leaks work and stalls resume for hours.

    def next_example(self) -> dict | None:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if rows is None:  # infinite env: pull the next generated task
            task = next(self.iters[env_name])
            self.cursors[env_name] = cursor + 1
            return {"task": task, "env_name": env_name}
        if cursor >= len(rows):
            self.epochs[env_name] += 1
            rows = self._shuffle(env_name)
            self.examples[env_name] = rows
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
