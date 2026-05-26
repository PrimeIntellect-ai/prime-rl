"""RolloutDispatcher: the single thing that schedules rollouts in v2.

Owns:

- A shared ``asyncio.Semaphore(config.max_inflight_rollouts)`` across train + eval.
  Capacity is denominated in rollouts; a group-scoring task that runs N rollouts
  in one call acquires N permits.
- A shared ``AsyncLimiter(config.tasks_per_minute, 60)`` if rate limiting is on.
- The dispatch loop: pick "next work" based on ``sched_mode`` and fill capacity.
- Per-group accumulation: completed rollouts are gathered per
  ``(env_name, example_id)`` group; the group emits one ``Trajectory`` to ``out_q``
  once all dispatched rollouts come back (succeeded, failed, or cancelled).
- Off-policy cancellation: on each ``on_new_version`` from the watcher, train
  rollouts whose ``off_policy_steps`` exceed ``max_off_policy_steps`` get cancelled.
  Eval rollouts are exempt (snapshot at dispatch, never cancelled mid-flight).
- Eval triggers: per-env ``policy.version % env.interval == 0`` queues an epoch.
  ``SchedMode.PREFER_EVAL`` means we only schedule eval until the queue drains;
  ``PREFER_TRAIN`` only schedules train. Both transitions are level-triggered
  (never cancel-and-restart), so in-flight rollouts of the opposite kind drain
  naturally on each side of the eval boundary — that's where the overlap lives.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.advantage import compute_advantages
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator_v2.policy import Policy
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger

Kind = Literal["train", "eval"]


class SchedMode(Enum):
    """Which kind of work the dispatcher will schedule next.

    Transitions are level-triggered (driven by the eval queue's emptiness), so
    in-flight rollouts of the opposite kind drain naturally on both sides of
    every eval boundary — the overlap mechanism.
    """

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class Trajectory:
    """One assembled group of rollouts emitted to ``out_q``.

    ``policy_version`` is the snapshot at dispatch time — used by the batcher
    for per-rollout off-policy metrics. ``eval_step`` is set only for eval
    trajectories (the policy version at which the eval epoch was triggered)
    so the batcher can aggregate eval rollouts back to their trigger step.
    """

    kind: Kind
    env_name: str
    example_id: int
    rollouts: list[vf.RolloutOutput]
    policy_version: int
    eval_step: int | None = None


@dataclass
class RolloutMeta:
    """Per-task bookkeeping. One entry per in-flight ``run_rollout`` / ``run_group``."""

    kind: Kind
    env_name: str
    group_id: int
    policy_version: int
    rollout_count: int  # number of semaphore permits this task holds
    client_config: vf.ClientConfig | None = None
    off_policy_steps: int = 0  # incremented on every ``on_new_version``; train only
    eval_step: int | None = None


@dataclass
class GroupState:
    """Accumulator for one rollout group across N independent ``run_rollout`` tasks.

    For group-scoring envs ``rollouts_to_schedule`` collapses to 0 after the
    single ``run_group`` task is queued; otherwise it's decremented per rollout.
    """

    kind: Kind
    env_name: str
    example: dict
    rollouts_to_schedule: int
    target_rollouts: int  # total rollouts expected for this group
    completed_rollouts: list[vf.RolloutOutput] = field(default_factory=list)
    failed_rollouts: int = 0
    eval_step: int | None = None
    pinned_client: vf.ClientConfig | None = None
    policy_version_at_start: int = 0


class _TrainEnvCycle:
    """Round-robin / weighted iterator over training datasets.

    Replaces the legacy ``Buffer.sample_examples`` for env-ratio sampling
    without dragging in the difficulty-pool tracking the v2 design dropped.
    """

    def __init__(self, train_envs: TrainEnvs, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self._envs = list(train_envs)
        if not self._envs:
            raise ValueError("RolloutDispatcher needs at least one train env")

        self._examples: dict[str, list[dict]] = {}
        self._cursors: dict[str, int] = {}
        for env in self._envs:
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
            self._examples[env.name] = rows
            self._cursors[env.name] = 0

        self._env_names = [e.name for e in self._envs]
        configured_ratios = [e.config.ratio for e in self._envs]
        if all(r is not None for r in configured_ratios):
            self._weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            # Natural distribution by dataset size — matches legacy Buffer's
            # "ratio unset → weight by num_normal" fallback.
            self._weights = [float(len(self._examples[name])) for name in self._env_names]

    def next_example(self) -> dict:
        env_name = self.rng.choices(self._env_names, weights=self._weights, k=1)[0]
        rows = self._examples[env_name]
        cursor = self._cursors[env_name]
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self._cursors[env_name] = cursor + 1
        return example


class RolloutDispatcher:
    """The only scheduler in v2. Runs as a single ``asyncio.Task`` under the
    orchestrator's ``TaskGroup`` (``await dispatcher.run()``).

    Observers (notably the watcher) drive ``on_new_version`` to advance off-policy
    counters and trigger eval epochs. The batcher reads completed ``Trajectory``
    instances from ``self.out_q``.
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        student_inference: InferencePool,
        teacher_inference: InferencePool | None,
        policy: Policy,
        resume_step: int | None = None,
    ) -> None:
        self.logger = get_logger()
        self.config = config
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs

        # Rollouts go to teacher in sft mode, student otherwise.
        if config.training_mode == "sft":
            assert teacher_inference is not None, "sft mode requires teacher_inference"
            self.rollout_inference = teacher_inference
        else:
            self.rollout_inference = student_inference

        # Shared concurrency cap. ``config.max_inflight_rollouts`` is guaranteed
        # to be set by the OrchestratorConfig resolver (it falls back from
        # batch_size or token_batch_size and oversampling_factor).
        assert config.max_inflight_rollouts is not None, "max_inflight_rollouts must be resolved before dispatcher init"
        self.max_inflight = config.max_inflight_rollouts
        self.semaphore = asyncio.Semaphore(self.max_inflight)
        self._inflight_permits = 0  # mirror of ``max_inflight - semaphore._value``
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(config.tasks_per_minute, time_period=60) if config.tasks_per_minute else None
        )

        # Dataset cycle (train) + eval queue.
        self._train_cycle = _TrainEnvCycle(train_envs, seed=config.seed)
        self.group_size = config.group_size

        # In-flight tracking.
        self.inflight: dict[asyncio.Task, RolloutMeta] = {}
        self.groups: dict[int, GroupState] = {}
        self._next_group_id = 0

        # Output queue. Bounded so the dispatcher backpressures on a slow batcher.
        self.out_q: asyncio.Queue[Trajectory] = asyncio.Queue(maxsize=max(8, self.max_inflight))

        # Scheduling priority.
        self.sched_mode: SchedMode = SchedMode.PREFER_TRAIN

        # Eval state.
        self._eval_queue: deque[tuple[str, dict, int]] = deque()  # (env_name, example, eval_step)
        self._eval_examples: dict[str, list[dict]] = {}
        self._eval_intervals: dict[str, int] = {}
        if eval_envs is not None and config.eval is not None:
            for env in eval_envs:
                rows: list[dict] = []
                for i, ex in enumerate(env.examples):
                    row = dict(ex)
                    row["env_name"] = env.name
                    if "example_id" not in row:
                        row["example_id"] = i
                    rows.append(row)
                self._eval_examples[env.name] = rows
                self._eval_intervals[env.name] = env.config.interval

        self.expected_eval_counts: dict[int, int] = {}  # eval_step -> total rollouts expected
        self.eval_step_envs: dict[int, set[str]] = {}  # eval_step -> set of envs that fired (public)

        # First-step eval handling — mirror legacy ``eval_base_model`` / ``skip_eval_on_resume``.
        eval_at_zero = False
        if config.eval is not None and config.eval.eval_base_model and resume_step is None:
            eval_at_zero = True
        self._eval_at_zero_pending = eval_at_zero
        self._resume_step = resume_step
        # ``last_eval_step`` per env: prevents re-triggering at the resumed step.
        self._last_eval_step_per_env: dict[str, int] = {name: 0 for name in self._eval_examples}
        if resume_step is not None:
            for name in self._last_eval_step_per_env:
                self._last_eval_step_per_env[name] = resume_step

        # Metrics counters (reset every poll by IntervalLogger).
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.errored_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.errors_by_type: dict[str, int] = defaultdict(int)
        self.total_rollouts_by_env: dict[str, int] = defaultdict(int)
        self.dropped_groups_by_env: dict[str, int] = defaultdict(int)
        self.eval_epochs_started: int = 0
        self._mode_transitions: int = 0

        self._stopped = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def model_name(self) -> str:
        """Model name to send on rollout requests — follows ``policy.model_name``
        in non-sft modes, the teacher's model name in sft."""
        if self.config.training_mode == "sft":
            return self.rollout_inference.model_name
        return self.policy.model_name

    @property
    def inflight_train_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "train")

    @property
    def inflight_eval_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "eval")

    @property
    def inflight_total_permits(self) -> int:
        return self._inflight_permits

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self._inflight_permits

    @property
    def queued_eval_examples(self) -> int:
        return len(self._eval_queue)

    @property
    def max_off_policy_level(self) -> int:
        train_steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return max(train_steps) if train_steps else 0

    @property
    def mean_off_policy_level(self) -> float:
        train_steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return sum(train_steps) / len(train_steps) if train_steps else 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self._task = asyncio.current_task()
        if self._eval_at_zero_pending and self.eval_envs is not None:
            self._fire_eval_epoch(0, log_reason="eval_base_model=true at step 0")
            self._eval_at_zero_pending = False

        try:
            while not self._stopped.is_set():
                await self._fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly and retry. Eval triggers from the
                    # watcher (sync ``_fire_eval_epoch`` mutation) will wake the
                    # next iteration.
                    try:
                        await asyncio.wait_for(self._stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                done, _pending = await asyncio.wait(
                    list(self.inflight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (eval triggers, sched_mode flips)
                )
                for task in done:
                    await self._handle_completed_rollout(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self._stopped.set()
        await self.cancel_inflight_rollouts()
        if self._task is not None:
            await safe_cancel(self._task)
            self._task = None

    # ── observer hook (called by WeightWatcher.on_new_version) ─────────────

    async def on_new_version(self, step: int) -> None:
        """Bump off-policy counters, cancel stale train rollouts, fire eval triggers."""
        # 1) Increment off-policy steps for in-flight train rollouts.
        stale_group_ids: set[int] = set()
        for meta in self.inflight.values():
            if meta.kind != "train":
                continue
            meta.off_policy_steps += 1
            if meta.off_policy_steps > self.config.max_off_policy_steps:
                stale_group_ids.add(meta.group_id)

        # 2) Cancel stale train groups. Eval is exempt by construction (never queued here).
        if stale_group_ids:
            removed = 0
            for gid in stale_group_ids:
                removed += await self._drop_group(gid)
            self.cancelled_rollouts_count += removed
            if removed:
                self.logger.warning(
                    f"Cancelled {removed} train rollouts past max_off_policy_steps={self.config.max_off_policy_steps}. "
                    "Consider increasing it to avoid this."
                )

        # 3) Per-env eval trigger checks.
        if self.eval_envs is None or self.config.eval is None:
            return
        if step == 0:
            return  # the step-0 eval is handled by ``_eval_at_zero_pending`` at startup
        if step == self._resume_step and self.config.eval.skip_eval_on_resume:
            return  # explicit resume opt-out
        fired_envs: list[str] = []
        for env_name, interval in self._eval_intervals.items():
            if step % interval != 0:
                continue
            if step <= self._last_eval_step_per_env.get(env_name, 0):
                continue
            self._enqueue_eval_env(env_name, step)
            fired_envs.append(env_name)
        if fired_envs:
            self._switch_mode(SchedMode.PREFER_EVAL, reason=f"eval triggered for {','.join(fired_envs)} @ step={step}")
            self.eval_epochs_started += 1
            self.logger.info(
                f"Eval @ step={step} for env(s) {','.join(fired_envs)} "
                f"(queued {len(self._eval_queue)} example(s); expected {self.expected_eval_counts.get(step, 0)} rollouts)"
            )

    async def force_eval(self, step: int) -> None:
        """Force-fire an eval epoch at ``step`` for *all* configured eval envs.

        Used by the orchestrator's end-of-training "final eval" pass.
        """
        if self.eval_envs is None or self.config.eval is None:
            return
        fired_envs: list[str] = []
        for env_name in list(self._eval_examples):
            self._enqueue_eval_env(env_name, step, force=True)
            fired_envs.append(env_name)
        if fired_envs:
            self._switch_mode(SchedMode.PREFER_EVAL, reason=f"force_eval for {','.join(fired_envs)} @ step={step}")

    # ── eval epoch machinery ───────────────────────────────────────────────

    def _fire_eval_epoch(self, step: int, log_reason: str) -> None:
        """Queue every example × group_size for each eval env at this step."""
        if self.eval_envs is None or self.config.eval is None:
            return
        fired_envs: list[str] = []
        for env_name in list(self._eval_examples):
            self._enqueue_eval_env(env_name, step, force=True)
            fired_envs.append(env_name)
        if fired_envs:
            self._switch_mode(SchedMode.PREFER_EVAL, reason=log_reason)
            self.eval_epochs_started += 1
            self.logger.info(
                f"Eval @ step={step} ({log_reason}) — queued envs={','.join(fired_envs)}, "
                f"{self.expected_eval_counts.get(step, 0)} expected rollouts"
            )

    def _enqueue_eval_env(self, env_name: str, step: int, *, force: bool = False) -> None:
        examples = self._eval_examples.get(env_name)
        if not examples:
            return
        if not force and step <= self._last_eval_step_per_env.get(env_name, 0):
            return
        assert self.eval_envs is not None
        eval_env = self.eval_envs.get(env_name)
        group_size = eval_env.config.group_size
        per_example_rollouts = group_size
        added = 0
        for example in examples:
            self._eval_queue.append((env_name, example, step))
            added += per_example_rollouts
        self._last_eval_step_per_env[env_name] = step
        self.expected_eval_counts[step] = self.expected_eval_counts.get(step, 0) + added
        self.eval_step_envs.setdefault(step, set()).add(env_name)

    # ── fill ────────────────────────────────────────────────────────────────

    async def _fill_inflight(self) -> None:
        """Schedule new rollouts up to the global semaphore cap.

        Honors the current ``sched_mode``: in ``PREFER_EVAL`` we only enqueue
        eval work; in ``PREFER_TRAIN`` we only enqueue train. When eval queue
        empties we transition back to ``PREFER_TRAIN`` (the eval tail in-flight
        keeps draining naturally).
        """
        while True:
            # Cheap pre-check: avoid acquiring the semaphore if we're full.
            if self.available_permits <= 0:
                return

            if self.sched_mode == SchedMode.PREFER_EVAL:
                if not self._eval_queue and self.inflight_eval_count == 0:
                    # All eval examples dispatched and finished — switch back to train.
                    self._switch_mode(SchedMode.PREFER_TRAIN, reason="eval queue drained")
                    continue
                if not self._eval_queue:
                    # Eval queue empty but eval still in flight: don't schedule train here
                    # (we don't want a hard PREFER_TRAIN flip until eval is done). The
                    # condition above handles that on the next iteration.
                    return
                scheduled = await self._try_schedule_eval()
                if not scheduled:
                    return
            else:  # PREFER_TRAIN
                scheduled = await self._try_schedule_train()
                if not scheduled:
                    # No train work available right now (rare; only if buffer empty).
                    return

    def _switch_mode(self, new_mode: SchedMode, *, reason: str) -> None:
        if new_mode == self.sched_mode:
            return
        # INFO-level so the eval-overlap transitions are visible in steady-state
        # production logs without needing DEBUG verbosity.
        self.logger.info(
            f"Dispatcher mode: {self.sched_mode.name} → {new_mode.name} "
            f"(inflight_train={self.inflight_train_count}, inflight_eval={self.inflight_eval_count}, "
            f"reason={reason})"
        )
        self.sched_mode = new_mode
        self._mode_transitions += 1

    async def _try_schedule_train(self) -> bool:
        # If any existing train group has pending rollouts that fit, prefer them.
        for group_id, group in list(self.groups.items()):
            if group.kind != "train" or group.rollouts_to_schedule <= 0:
                continue
            env = self.train_envs.get(group.env_name)
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= self.available_permits:
                return await self._schedule_group_rollout(group_id, group)

        # Need a fresh group; reserve ``group_size`` permits for it.
        if self.available_permits < self.group_size:
            return False
        example = self._train_cycle.next_example()
        env_name = example["env_name"]
        group_id = self._new_group_id()
        self.groups[group_id] = GroupState(
            kind="train",
            env_name=env_name,
            example=example,
            rollouts_to_schedule=self.group_size,
            target_rollouts=self.group_size,
            policy_version_at_start=self.policy.version,
        )
        return await self._schedule_group_rollout(group_id, self.groups[group_id])

    async def _try_schedule_eval(self) -> bool:
        if not self._eval_queue:
            return False
        # Peek: do we have enough capacity for the next eval example's rollouts?
        env_name, example, eval_step = self._eval_queue[0]
        eval_env = self.eval_envs.get(env_name) if self.eval_envs is not None else None
        if eval_env is None:
            self._eval_queue.popleft()
            return False
        per_example_rollouts = eval_env.config.group_size
        cost = per_example_rollouts if eval_env.requires_group_scoring else 1
        if cost > self.available_permits:
            return False
        self._eval_queue.popleft()
        group_id = self._new_group_id()
        self.groups[group_id] = GroupState(
            kind="eval",
            env_name=env_name,
            example=example,
            rollouts_to_schedule=per_example_rollouts,
            target_rollouts=per_example_rollouts,
            eval_step=eval_step,
            policy_version_at_start=self.policy.version,
        )
        return await self._schedule_group_rollout(group_id, self.groups[group_id])

    def _new_group_id(self) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        return gid

    async def _schedule_group_rollout(self, group_id: int, group: GroupState) -> bool:
        """Dispatch one ``run_rollout`` / ``run_group`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Pick or pin a client for the group. Pinning keeps prefix-cache hits
        # within a group (matches the legacy scheduler).
        if group.pinned_client is None:
            client = await self._select_least_loaded_client()
            if group_id not in self.groups:
                return False
            group.pinned_client = client
        else:
            client = group.pinned_client

        env_collection = self.train_envs if group.kind == "train" else self.eval_envs
        if env_collection is None:
            return False
        env = env_collection.get(group.env_name)
        cache_salt = str(group.policy_version_at_start)
        model_name = self.model_name

        if env.requires_group_scoring:
            permits = group.rollouts_to_schedule
            group.rollouts_to_schedule = 0
            await self._acquire(permits)
            task: asyncio.Task = asyncio.create_task(
                env.run_group(
                    client=client,
                    example=group.example,
                    model_name=model_name,
                    group_size=permits,
                    cache_salt=cache_salt,
                )
            )
        else:
            permits = 1
            group.rollouts_to_schedule -= 1
            await self._acquire(permits)
            task = asyncio.create_task(
                env.run_rollout(
                    client=client,
                    example=group.example,
                    model_name=model_name,
                    cache_salt=cache_salt,
                )
            )

        self.inflight[task] = RolloutMeta(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            policy_version=group.policy_version_at_start,
            rollout_count=permits,
            client_config=client,
            eval_step=group.eval_step,
        )
        return True

    async def _acquire(self, n: int) -> None:
        """Acquire ``n`` permits from the shared semaphore. Optionally rate-limit each one."""
        for _ in range(n):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            await self.semaphore.acquire()
            self._inflight_permits += 1

    def _release(self, n: int) -> None:
        for _ in range(n):
            self.semaphore.release()
            self._inflight_permits -= 1

    @staticmethod
    def _client_identity(c: vf.ClientConfig) -> tuple[str, str | None]:
        return (c.api_base_url, c.extra_headers.get("X-data-parallel-rank"))

    async def _select_least_loaded_client(self) -> vf.ClientConfig:
        clients = self.rollout_inference.train_clients
        while not clients:
            await asyncio.sleep(0.5)
            clients = self.rollout_inference.train_clients
        load = Counter(
            self._client_identity(m.client_config) for m in self.inflight.values() if m.client_config is not None
        )
        return min(clients, key=lambda c: load[self._client_identity(c)])

    # ── completion handling ────────────────────────────────────────────────

    async def _handle_completed_rollout(self, task: asyncio.Task) -> None:
        meta = self.inflight.pop(task, None)
        if meta is None:
            return
        self._release(meta.rollout_count)

        group = self.groups.get(meta.group_id)
        if group is None:
            # Group was dropped (off-policy cancel, etc.) — discard the result.
            return

        try:
            result = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.logger.warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            self.errored_rollouts_by_env[meta.env_name] += meta.rollout_count
            self.errors_by_type[type(exc).__name__] += 1
            group.failed_rollouts += meta.rollout_count
            await self._maybe_finalize_group(meta.group_id)
            return

        rollouts: list[vf.RolloutOutput] = result if isinstance(result, list) else [result]
        self.total_rollouts_by_env[meta.env_name] += len(rollouts)

        env_collection = self.train_envs if meta.kind == "train" else self.eval_envs
        assert env_collection is not None
        env = env_collection.get(meta.env_name)

        valid: list[vf.RolloutOutput] = []
        for r in rollouts:
            if r.get("error") is not None:
                self.errored_rollouts_by_env[meta.env_name] += 1
                self.errors_by_type[r["error"]["error"]] += 1
                self.logger.warning(
                    f"Rollout failed in group {meta.group_id} ({meta.env_name}) — "
                    f"{r['error'].get('error_chain_repr', r['error'].get('error'))}"
                )
            elif len(r.get("trajectory") or []) == 0:
                self.empty_rollouts_by_env[meta.env_name] += 1
                self.logger.warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            else:
                r["env_name"] = meta.env_name
                valid.append(r)

        num_failed = len(rollouts) - len(valid)
        group.failed_rollouts += num_failed

        # Group-scoring envs: any failure means surviving rollouts carry scores
        # computed against the (now-missing) failed ones — unsafe to keep. Drop
        # the whole group.
        if num_failed > 0 and env.requires_group_scoring:
            self.dropped_groups_by_env[meta.env_name] += 1
            self.logger.warning(f"Dropping group-scored group {meta.group_id} ({meta.env_name}) after rollout failure")
            await self._drop_group(meta.group_id)
            return

        group.completed_rollouts.extend(valid)
        await self._maybe_finalize_group(meta.group_id)

    async def _maybe_finalize_group(self, group_id: int) -> None:
        """Emit a ``Trajectory`` once every dispatched rollout has come back.

        Partial groups (some failed) are still emitted with the surviving
        rollouts — matches legacy behavior. All-failed groups are dropped and
        counted in ``dropped_groups_by_env``.
        """
        group = self.groups.get(group_id)
        if group is None:
            return
        if len(group.completed_rollouts) + group.failed_rollouts < group.target_rollouts:
            return

        # Group complete (or partial-complete).
        self.groups.pop(group_id, None)
        if not group.completed_rollouts:
            self.dropped_groups_by_env[group.env_name] += 1
            self.logger.warning(
                f"Dropping {group.kind} group {group_id} ({group.env_name}) — all "
                f"{group.target_rollouts} rollouts failed"
            )
            return

        if group.failed_rollouts > 0:
            self.logger.warning(
                f"Partial {group.kind} group {group_id} ({group.env_name}) — "
                f"{len(group.completed_rollouts)}/{group.target_rollouts} valid "
                f"({group.failed_rollouts} failed)"
            )

        # Compute advantages per-group for train trajectories. Done here so
        # pre_batch_filters (e.g. ZeroAdvantageFilter) can fire in the batcher
        # without re-grouping rollouts.
        if group.kind == "train" and self.config.advantage is not None:
            await asyncio.to_thread(compute_advantages, group.completed_rollouts, self.config.advantage)
        elif group.kind == "train" and self.config.advantage is None:
            for r in group.completed_rollouts:
                r["advantage"] = r.get("reward", 0.0)

        example_id = group.example.get("example_id")
        if example_id is None:
            example_id = -1
        traj = Trajectory(
            kind=group.kind,
            env_name=group.env_name,
            example_id=int(example_id),
            rollouts=group.completed_rollouts,
            policy_version=group.policy_version_at_start,
            eval_step=group.eval_step,
        )
        await self.out_q.put(traj)

    async def _drop_group(self, group_id: int) -> int:
        """Cancel any remaining in-flight tasks for this group and forget it.

        Returns the number of rollouts cancelled (for off-policy metrics).
        """
        tasks_to_cancel: list[asyncio.Task] = []
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            self.inflight.pop(task, None)
            self._release(meta.rollout_count)
            tasks_to_cancel.append(task)
            cancelled += meta.rollout_count
        self.groups.pop(group_id, None)
        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        return cancelled

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollouts (used on shutdown only)."""
        for meta in self.inflight.values():
            self._release(meta.rollout_count)
        tasks = list(self.inflight.keys())
        self.cancelled_rollouts_count += sum(m.rollout_count for m in self.inflight.values())
        self.inflight.clear()
        self.groups.clear()
        if tasks:
            await safe_cancel_all(tasks)

    # ── metrics ────────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        """Instantaneous gauges sampled by ``IntervalLogger``. Read-only."""
        return {
            "dispatcher/inflight_train": float(self.inflight_train_count),
            "dispatcher/inflight_eval": float(self.inflight_eval_count),
            "dispatcher/inflight_total_permits": float(self._inflight_permits),
            "dispatcher/available_permits": float(self.available_permits),
            "dispatcher/queued_eval_examples": float(self.queued_eval_examples),
            "dispatcher/sched_mode": float(self.sched_mode == SchedMode.PREFER_EVAL),
            "dispatcher/mode_transitions": float(self._mode_transitions),
            "dispatcher/groups_in_flight": float(len(self.groups)),
            "dispatcher/off_policy_level_max": float(self.max_off_policy_level),
            "dispatcher/off_policy_level_mean": self.mean_off_policy_level,
            "dispatcher/eval_epochs_started": float(self.eval_epochs_started),
        }

    def drain_metrics(self) -> dict[str, float]:
        """Counters that reset each call (mirrors ``Scheduler.get_metrics``)."""
        total_rollouts = sum(self.total_rollouts_by_env.values())
        out: dict[str, float] = {
            "dispatcher/cancelled_rollouts": float(self.cancelled_rollouts_count),
            "dispatcher/dropped_groups": float(sum(self.dropped_groups_by_env.values())),
            "rollouts/empty_rate": (sum(self.empty_rollouts_by_env.values()) / max(total_rollouts, 1)),
            "rollouts/errored_rate": (sum(self.errored_rollouts_by_env.values()) / max(total_rollouts, 1)),
        }
        for env_name, count in self.dropped_groups_by_env.items():
            out[f"dispatcher/dropped_groups/{env_name}"] = float(count)
        for error_type, count in self.errors_by_type.items():
            out[f"dispatcher/errors/{error_type}"] = float(count)
        for env_name in self.total_rollouts_by_env:
            env_total = max(self.total_rollouts_by_env[env_name], 1)
            out[f"rollouts/{env_name}/empty_rate"] = self.empty_rollouts_by_env.get(env_name, 0) / env_total
            out[f"rollouts/{env_name}/errored_rate"] = self.errored_rollouts_by_env.get(env_name, 0) / env_total
        out.update(self.rollout_inference.get_metrics())
        self.cancelled_rollouts_count = 0
        self.empty_rollouts_by_env.clear()
        self.errored_rollouts_by_env.clear()
        self.errors_by_type.clear()
        self.total_rollouts_by_env.clear()
        self.dropped_groups_by_env.clear()
        return out

    # Make accessors palatable for IntervalLogger.
    def __iter_train_envs__(self) -> Iterator[str]:
        return iter(self.train_envs.names)
