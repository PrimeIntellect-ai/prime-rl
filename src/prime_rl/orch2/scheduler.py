import random
from collections import deque
from dataclasses import dataclass
from typing import Iterator, Literal, TypeAlias

import verifiers as vf

from prime_rl.configs.orchestrator import EvalEnvConfig, TrainEnvConfig
from prime_rl.utils.logger import get_logger

Kind: TypeAlias = Literal["train", "eval"]


@dataclass
class Task:
    """What the engine needs to run a rollout. No dataset — that's the scheduler's job."""

    id: str
    env: vf.Environment
    sampling_args: dict
    kind: Kind
    rollouts_per_group: int


@dataclass
class Dispatch:
    """One unit of work handed to the engine."""

    task: Task
    example: dict
    eval_step: int | None = None  # trigger step for eval dispatches; None for train


class Scheduler:
    """Round-robin over train envs, interleaved with eval epochs.

    When `on_new_version(step)` fires at an eval-interval boundary (or at init
    for `eval_base_model`), the scheduler populates `_eval_queue` with the
    cartesian product of eval envs × eval examples. `next_task()` drains the
    eval queue exclusively before returning to train round-robin, so the engine
    sees an eval-only phase followed by a train phase — no mixing within the
    dispatch stream, though both may be in-flight simultaneously.
    """

    def __init__(
        self,
        train_envs: list[TrainEnvConfig],
        train_rollouts_per_example: int,
        eval_envs: list[EvalEnvConfig] | None = None,
        eval_interval: int | None = None,
        eval_at_zero: bool = False,
        seed: int | None = None,
    ):
        assert train_envs, "Scheduler requires at least one train env"
        logger = get_logger()

        self.tasks: list[Task] = []
        self._datasets: list[Iterator[dict]] = []
        for cfg in train_envs:
            env = vf.load_environment(cfg.stripped_id, **cfg.args)
            self.tasks.append(
                Task(
                    id=cfg.resolved_name,
                    env=env,
                    sampling_args=cfg.sampling.to_sampling_args(),
                    kind="train",
                    rollouts_per_group=train_rollouts_per_example,
                )
            )
            self._datasets.append(_cycle_forever(env.get_dataset()))
        self._idx = 0

        # Env selection: weighted random when all envs have a ratio set
        # (config validator enforces all-or-none), round-robin otherwise.
        # Local Random instance keeps env-selection determinism decoupled from
        # global random state.
        ratios = [cfg.ratio for cfg in train_envs]
        self._env_weights: list[float] | None = ratios if all(r is not None for r in ratios) else None  # type: ignore[assignment]
        self._rng = random.Random(seed)
        if self._env_weights is not None:
            named = ", ".join(f"{cfg.resolved_name}={cfg.ratio:.2f}" for cfg in train_envs)
            logger.info(f"Sampling train envs by ratio ({named})")
        else:
            logger.info(f"Sampling train envs round-robin ({len(train_envs)} env(s))")

        self.eval_tasks: list[Task] = []
        self._eval_datasets: list[list[dict]] = []
        if eval_envs:
            for cfg in eval_envs:
                env = vf.load_environment(cfg.stripped_id, **cfg.args)
                self.eval_tasks.append(
                    Task(
                        id=cfg.resolved_name,
                        env=env,
                        sampling_args=cfg.sampling.to_sampling_args(),
                        kind="eval",
                        rollouts_per_group=cfg.rollouts_per_example,
                    )
                )
                ds = env.get_eval_dataset() if hasattr(env, "get_eval_dataset") else env.get_dataset()
                rows = list(ds)
                if cfg.num_examples != -1:
                    rows = rows[: cfg.num_examples]
                self._eval_datasets.append([dict(r) for r in rows])

        self.eval_interval = eval_interval
        self._eval_queue: deque[tuple[Task, dict, int]] = deque()
        self._dispatching_eval_step: int | None = None
        self._eval_expected: dict[int, int] = {}
        # Last step an eval was triggered at. Used to fire at the next boundary
        # even when the weight watcher jumps over the exact interval step (e.g.
        # weights skip from 3 → 5 — we still want an eval around step 4/5).
        self.last_eval_step = 0

        if eval_at_zero and self.eval_tasks:
            self._start_eval_epoch(0)

    def _start_eval_epoch(self, step: int) -> None:
        entries: list[tuple[Task, dict, int]] = []
        for task, ds in zip(self.eval_tasks, self._eval_datasets):
            for i, row in enumerate(ds):
                ex = dict(row)
                ex["example_id"] = i
                entries.append((task, ex, step))
        self._eval_queue = deque(entries)
        self._dispatching_eval_step = step
        self._eval_expected[step] = len(entries)

    def expected_eval_count(self, step: int) -> int | None:
        """How many eval groups were dispatched for this eval epoch. None if
        there was no epoch triggered at that step."""
        return self._eval_expected.get(step)

    def next_task(self) -> Dispatch | None:
        if self._eval_queue:
            task, example, eval_step = self._eval_queue.popleft()
            if not self._eval_queue:
                self._dispatching_eval_step = None
            return Dispatch(task=task, example=example, eval_step=eval_step)

        if self._env_weights is not None:
            # Weighted: pick by ratio; loop only as a defensive guard if an
            # iterator is exhausted (shouldn't happen with _cycle_forever).
            for _ in range(len(self.tasks)):
                i = self._rng.choices(range(len(self.tasks)), weights=self._env_weights, k=1)[0]
                try:
                    example = next(self._datasets[i])
                    return Dispatch(task=self.tasks[i], example=example)
                except StopIteration:
                    continue
            return None

        for _ in range(len(self.tasks)):
            i = self._idx
            self._idx = (self._idx + 1) % len(self.tasks)
            try:
                example = next(self._datasets[i])
                return Dispatch(task=self.tasks[i], example=example)
            except StopIteration:
                continue
        return None

    async def on_new_version(self, step: int) -> None:
        if not self.eval_tasks or self.eval_interval is None:
            return
        # Watcher may jump over boundaries (e.g. 3 → 5) so we trigger as soon
        # as `step` is an interval past the last eval, not on exact multiples.
        if step < self.last_eval_step + self.eval_interval:
            return
        if self._dispatching_eval_step is not None:
            # Previous epoch still has items in the dispatch queue; skip.
            # Eval still in-flight (already dispatched) is fine — we'd just
            # start a new epoch next iteration. The skip here only applies
            # when the previous epoch hasn't even finished dispatching.
            return
        self._start_eval_epoch(step)
        self.last_eval_step = step


def _cycle_forever(ds) -> Iterator[dict]:
    i = 0
    while True:
        for row in ds:
            r = dict(row)
            r["example_id"] = i
            i += 1
            yield r
