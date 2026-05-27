"""Example sources the dispatcher pulls from.

Two abstractions, same spirit:

- ``TrainSource``: infinite pull. Yields the next training example forever,
  weighted-round-robin across train envs (or by dataset size if ratios are
  unset). The dispatcher pulls from it in ``PREFER_TRAIN`` mode.

- ``EvalSource``: trigger-driven finite-per-epoch pull. When the watcher
  advances ``policy.version`` to an eval interval, the dispatcher calls
  ``trigger(step)`` and the source enqueues every example for every eligible
  env at that step. The dispatcher pulls from it in ``PREFER_EVAL`` mode
  until the queue drains, then flips back. ``trigger_at_start()`` handles
  the ``skip_first_step`` startup-eval case.

The dispatcher still owns scheduling priority (``SchedMode``), capacity
(semaphore), and per-env cost lookups (``EvalEnvs.get(...).config``); these
sources just answer "what's the next example to schedule?".
"""

from __future__ import annotations

import random
from collections import deque

from prime_rl.configs.orchestrator import EvalConfig
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs


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
            dataset = env.get_dataset(seed=seed)
            column_names = getattr(dataset, "column_names", None)
            has_example_id = column_names is not None and "example_id" in column_names
            rows: list[dict] = []
            for i, row in enumerate(dataset):
                ex = dict(row)
                ex["env_name"] = env.name
                if not has_example_id and "example_id" not in ex:
                    ex["example_id"] = i
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


class EvalSource:
    """Finite-per-epoch source of eval examples.

    Holds the per-env example lists + intervals + the per-(env, step) pending
    queue. The dispatcher pokes it via ``trigger(step)`` when the watcher
    advances ``policy.version`` (or ``trigger_at_start()`` once at the very
    start when ``skip_first_step`` is set), then pulls examples via
    ``peek`` / ``pop`` until ``bool(source) == False``.

    Empty pool (``eval_envs is None`` or ``eval_config is None``) is a valid
    state — ``trigger`` and ``peek`` become no-ops, ``bool(source)`` stays
    False forever.
    """

    def __init__(
        self,
        eval_envs: EvalEnvs | None,
        eval_config: EvalConfig | None,
        *,
        resume_step: int | None = None,
    ) -> None:
        self.eval_envs = eval_envs
        self.eval_config = eval_config
        self.resume_step = resume_step

        self.examples_by_env: dict[str, list[dict]] = {}
        self.intervals: dict[str, int] = {}
        if eval_envs is not None and eval_config is not None:
            for env in eval_envs:
                rows: list[dict] = []
                for i, ex in enumerate(env.examples):
                    row = dict(ex)
                    row["env_name"] = env.name
                    if "example_id" not in row:
                        row["example_id"] = i
                    rows.append(row)
                self.examples_by_env[env.name] = rows
                self.intervals[env.name] = env.config.interval

        # (env_name, example, eval_step) FIFO. Each ``trigger`` extends it
        # with one batch per fired env.
        self.queue: deque[tuple[str, dict, int]] = deque()

        # Last step we fired this env at — prevents a single step crossing
        # the interval boundary twice. Initialized to ``resume_step`` so
        # resuming doesn't immediately re-fire the just-completed step.
        last = resume_step or 0
        self.last_eval_step: dict[str, int] = {name: last for name in self.examples_by_env}

    # ── trigger ────────────────────────────────────────────────────────────

    def trigger(self, step: int, *, force_all: bool = False) -> list[str]:
        """Fire eligible envs for ``step`` and return their names.

        Default gating fires an env iff ``step % interval == 0`` AND
        ``step > last_eval_step[env]``. ``skip_eval_on_resume`` blocks the
        very first trigger after a resume; ``step == 0`` is also gated off
        (the startup path uses ``trigger_at_start`` instead).

        ``force_all=True`` bypasses gating and fires every env — used by
        ``trigger_at_start`` for the ``skip_first_step`` startup epoch.
        """
        if self.eval_envs is None or self.eval_config is None:
            return []
        if not force_all:
            if step == 0:
                return []
            if step == self.resume_step and self.eval_config.skip_eval_on_resume:
                return []

        fired: list[str] = []
        for env_name, interval in self.intervals.items():
            if not force_all:
                if step % interval != 0:
                    continue
                if step <= self.last_eval_step.get(env_name, 0):
                    continue
            self.enqueue(env_name, step)
            fired.append(env_name)
        return fired

    def trigger_at_start(self) -> list[str]:
        """Fire all envs at step 0 if ``eval.skip_first_step`` is True and
        this is not a resume. Called once from ``Orchestrator.start``."""
        if self.eval_config is None or not self.eval_config.skip_first_step:
            return []
        if self.resume_step is not None:
            return []
        return self.trigger(0, force_all=True)

    def enqueue(self, env_name: str, step: int) -> None:
        for example in self.examples_by_env.get(env_name, []):
            self.queue.append((env_name, example, step))
        self.last_eval_step[env_name] = step

    # ── pull ───────────────────────────────────────────────────────────────

    def peek(self) -> tuple[str, dict, int] | None:
        return self.queue[0] if self.queue else None

    def pop(self) -> tuple[str, dict, int] | None:
        return self.queue.popleft() if self.queue else None

    def __bool__(self) -> bool:
        return bool(self.queue)

    def __len__(self) -> int:
        return len(self.queue)
