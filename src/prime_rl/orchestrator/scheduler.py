import random
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Literal, TypeAlias

import verifiers as vf

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.utils.logger import get_logger

if TYPE_CHECKING:
    from prime_rl.orchestrator.buffer import DifficultyBuffer

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


@dataclass
class SchedulerInputs:
    """Pre-built inputs for the Scheduler. `setup_scheduler` produces this from
    config; tests construct it directly with handcrafted tasks/datasets."""

    train_tasks: list[Task]
    train_datasets: list[Iterator[dict]]
    eval_tasks: list[Task] = field(default_factory=list)
    eval_datasets: list[list[dict]] = field(default_factory=list)
    env_weights: list[float] | None = None
    eval_interval: int | None = None
    eval_at_zero: bool = False
    seed: int | None = None


class Scheduler:
    """Round-robin over train envs, interleaved with eval epochs.

    When `on_new_version(step)` fires at an eval-interval boundary (or at init
    for `eval_base_model`), the scheduler populates `_eval_queue` with the
    cartesian product of eval envs × eval examples. `next_task()` drains the
    eval queue exclusively before returning to train round-robin, so the engine
    sees an eval-only phase followed by a train phase — no mixing within the
    dispatch stream, though both may be in-flight simultaneously.
    """

    def __init__(self, inputs: SchedulerInputs):
        assert inputs.train_tasks, "Scheduler requires at least one train task"
        assert len(inputs.train_tasks) == len(inputs.train_datasets), (
            "train_tasks and train_datasets must have equal length"
        )
        if inputs.env_weights is not None:
            assert len(inputs.env_weights) == len(inputs.train_tasks), "env_weights length must match train_tasks"
        assert len(inputs.eval_tasks) == len(inputs.eval_datasets), (
            "eval_tasks and eval_datasets must have equal length"
        )

        self.tasks = inputs.train_tasks
        self._datasets = inputs.train_datasets
        self._idx = 0

        # Env selection: weighted random when env_weights is set, round-robin
        # otherwise. Local Random instance keeps env-selection determinism
        # decoupled from global random state.
        self._env_weights = inputs.env_weights
        self._rng = random.Random(inputs.seed)

        self.eval_tasks = inputs.eval_tasks
        self._eval_datasets = inputs.eval_datasets

        self.eval_interval = inputs.eval_interval
        self._eval_queue: deque[tuple[Task, dict, int]] = deque()
        self._dispatching_eval_step: int | None = None
        self._eval_expected: dict[int, int] = {}
        # Last step an eval was triggered at. Used to fire at the next boundary
        # even when the weight watcher jumps over the exact interval step (e.g.
        # weights skip from 3 → 5 — we still want an eval around step 4/5).
        self.last_eval_step = 0

        if inputs.eval_at_zero and self.eval_tasks:
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


def setup_scheduler(
    cfg: OrchestratorConfig, buffer: "DifficultyBuffer | None" = None, resume_step: int | None = None
) -> Scheduler:
    """Translate config → Scheduler. Loads verifiers envs, wraps datasets in
    `_cycle_forever`, and detects round-robin vs weighted env selection.

    Tests should construct `Scheduler(SchedulerInputs(...))` directly with
    handcrafted tasks/datasets — this function is the production callsite."""
    logger = get_logger()

    train_tasks: list[Task] = []
    train_datasets: list[Iterator[dict]] = []
    for env_cfg in cfg.train.env:
        env = vf.load_environment(env_cfg.stripped_id, **env_cfg.args)
        train_tasks.append(
            Task(
                id=env_cfg.resolved_name,
                env=env,
                sampling_args=env_cfg.sampling.to_sampling_args(),
                kind="train",
                rollouts_per_group=cfg.rollouts_per_example,
            )
        )
        train_datasets.append(_cycle_forever(env.get_dataset(), env_id=env_cfg.resolved_name, buffer=buffer))

    ratios = [env_cfg.ratio for env_cfg in cfg.train.env]
    env_weights: list[float] | None = ratios if all(r is not None for r in ratios) else None  # type: ignore[assignment]
    if env_weights is not None:
        named = ", ".join(f"{e.resolved_name}={e.ratio:.2f}" for e in cfg.train.env)
        logger.info(f"Sampling train envs by ratio ({named})")
    else:
        logger.info(f"Sampling train envs round-robin ({len(cfg.train.env)} env(s))")

    eval_tasks: list[Task] = []
    eval_datasets: list[list[dict]] = []
    if cfg.eval is not None:
        for env_cfg in cfg.eval.env:
            env = vf.load_environment(env_cfg.stripped_id, **env_cfg.args)
            eval_tasks.append(
                Task(
                    id=env_cfg.resolved_name,
                    env=env,
                    sampling_args=env_cfg.sampling.to_sampling_args(),
                    kind="eval",
                    rollouts_per_group=env_cfg.rollouts_per_example,
                )
            )
            ds = env.get_eval_dataset() if hasattr(env, "get_eval_dataset") else env.get_dataset()
            rows = list(ds)
            if env_cfg.num_examples != -1:
                rows = rows[: env_cfg.num_examples]
            eval_datasets.append([dict(r) for r in rows])

    eval_at_zero = cfg.eval is not None and cfg.eval.eval_base_model and resume_step is None

    return Scheduler(
        SchedulerInputs(
            train_tasks=train_tasks,
            train_datasets=train_datasets,
            eval_tasks=eval_tasks,
            eval_datasets=eval_datasets,
            env_weights=env_weights,
            eval_interval=cfg.eval.interval if cfg.eval else None,
            eval_at_zero=eval_at_zero,
            seed=cfg.seed,
        )
    )


def _cycle_forever(ds, env_id: str | None = None, buffer: "DifficultyBuffer | None" = None) -> Iterator[dict]:
    """Cycle the dataset forever, skipping examples evicted into the buffer's
    easy/hard pools. If every remaining example in a full pass is evicted we
    yield anyway as a fallback — better to over-train on hard ones than stall."""
    i = 0
    while True:
        yielded_this_pass = False
        for row in ds:
            r = dict(row)
            r["example_id"] = i
            i += 1
            if buffer is not None and env_id is not None and buffer.is_evicted(env_id, r):
                continue
            yielded_this_pass = True
            yield r
        if not yielded_this_pass:
            # Entire dataset is evicted; fall back to yielding without filtering
            # so the engine doesn't deadlock waiting for tasks. Single warning,
            # then we keep streaming raw rows until something un-evicts.
            get_logger().warning(
                f"Dataset for env={env_id!r} fully evicted; yielding evicted examples to avoid deadlock"
            )
            for row in ds:
                r = dict(row)
                r["example_id"] = i
                i += 1
                yield r
