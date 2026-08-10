"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights are each env's configured ``ratio`` (default 1, i.e. equal weight
per env). A v1 env serves the tasks the orchestrator loaded client-side: a
finite one as a shuffled table (reshuffled with ``seed=epoch`` on cursor
exhaustion), an infinite one (``num_tasks is None``) straight off its
generator — every pull is a fresh task and there are no epochs to shuffle."""

from __future__ import annotations

import random
from collections.abc import Iterator

import verifiers.v1 as vf

from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.types import TaskItem


class TrainSource:
    """``next_task()`` picks a weighted-RR env and returns its next v1 task item.

    The data position round-trips through a checkpoint via ``state_dict()``
    / ``load_state_dict()``: per-env ``{epoch, cursor}`` plus the env-choice
    RNG state, making the dispatch sequence reproducible across resumes. For
    a finite env, epochs are 1-indexed and seed that epoch's shuffle; for an
    infinite env the epoch stays 1 and the cursor counts generator pulls,
    replayed on restore by fast-forwarding the generator (exact iff it's
    deterministic). Cursors advance at dispatch time (ahead of shipped
    batches), so a resume skips the tasks that were in flight at checkpoint
    time."""

    def __init__(self, train_envs: TrainEnvs) -> None:
        self.rng = random.Random(42)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        # A finite env's task table in canonical order (each epoch's shuffle
        # starts from this); ``None`` for an infinite env, whose generator
        # (``self.iters``) is pulled per task.
        self.base_rows: dict[str, list[TaskItem] | None] = {}
        self.task_rows: dict[str, list[TaskItem] | None] = {}
        self.iters: dict[str, Iterator[vf.TaskData]] = {}
        self.epochs: dict[str, int] = {}
        self.cursors: dict[str, int] = {}
        for env in self.envs:
            if env.num_tasks is None:  # infinite: pull the generator per task
                rows = None
                self.iters[env.name] = env.tasks
            else:
                rows = [TaskItem(env_name=env.name, data=data) for data in env.tasks]
                if not rows:
                    raise ValueError(f"Train env {env.name} has no tasks")
            self.base_rows[env.name] = rows
            self.epochs[env.name] = 1
            self.cursors[env.name] = 0
            self.task_rows[env.name] = self._shuffle(env.name)

        self.env_names = [e.name for e in self.envs]
        self.weights: list[float] = [float(e.config.ratio) for e in self.envs]

    def _shuffle(self, env_name: str) -> list[TaskItem] | None:
        """The env's task table shuffled for its current epoch — a pure
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
            if self.base_rows[name] is None:
                for _ in range(position["cursor"]):
                    next(self.iters[name])
            else:
                self.task_rows[name] = self._shuffle(name)

    def next_task(self) -> TaskItem:
        env_name = self.rng.choices(self.env_names, weights=self.weights, k=1)[0]
        rows = self.task_rows[env_name]
        cursor = self.cursors[env_name]
        if rows is None:  # infinite env: pull the next generated task
            data = next(self.iters[env_name])
            self.cursors[env_name] = cursor + 1
            return TaskItem(env_name=env_name, data=data)
        if cursor >= len(rows):
            self.epochs[env_name] += 1
            rows = self._shuffle(env_name)
            self.task_rows[env_name] = rows
            cursor = 0
        task = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return task
