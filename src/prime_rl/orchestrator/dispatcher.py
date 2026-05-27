"""RolloutDispatcher: the only thing that schedules rollouts.

Owns:

- A shared ``asyncio.Semaphore(config.max_inflight_rollouts)`` across train + eval.
  Capacity is denominated in rollouts; a group-scoring task that runs N rollouts
  in one call acquires N permits.
- A shared ``AsyncLimiter(config.tasks_per_minute, 60)`` if rate limiting is on.
- The dispatch loop: pick "next work" based on ``sched_mode`` and fill capacity.
- Emit-everything invariant: every dispatched rollout (one permit) eventually
  reaches ``out_q`` exactly once as a ``Rollout`` — successful, env-errored,
  empty-trajectory, task-exception, or off-policy-cancelled. Failures carry
  ``raw["error"]`` set; the sinks decide drop / partial-train policy.
- Off-policy cancellation: on each ``on_new_version`` from the watcher, train
  rollouts whose ``off_policy_steps`` exceed ``max_off_policy_steps`` get
  cancelled, and a synthetic "Cancelled" rollout is emitted in their place so
  the sink can finalize the partial group. Eval rollouts are exempt (snapshot
  at dispatch, never cancelled mid-flight).
- Eval triggers: per-env ``policy.version % env.interval == 0`` queues an epoch.
  ``SchedMode.PREFER_EVAL`` means we only schedule eval until the queue drains;
  ``PREFER_TRAIN`` only schedules train. Both transitions are level-triggered
  (never cancel-and-restart), so in-flight rollouts of the opposite kind drain
  naturally on each side of the eval boundary — that's where the overlap lives.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from typing import Literal

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.sources import EvalSource, TrainSource
from prime_rl.orchestrator.types import (
    DispatcherMetrics,
    GroupState,
    Kind,
    Policy,
    Rollout,
    RolloutMeta,
    SchedMode,
)
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool, client_identity
from prime_rl.utils.logger import get_logger


class RolloutDispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.

    The orchestrator owns the example sources (``TrainSource``,
    ``EvalSource``) and the policy; the dispatcher is purely the scheduler
    that pulls from them, enforces concurrency / off-policy caps, and emits
    completed ``Rollout``\\ s to ``out_q``.

    Observers (notably the ``WeightWatcher``) drive ``on_new_version`` to
    advance off-policy counters and cancel stale rollouts. Eval epoch
    triggering is the orchestrator's job (one trigger per training step),
    not the dispatcher's.
    """

    def __init__(
        self,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource,
        eval_source: EvalSource,
        inference: InferencePool,
        policy: Policy,
        max_inflight_rollouts: int,
        tasks_per_minute: float | None,
        group_size: int,
        max_off_policy_steps_train: int,
        max_off_policy_steps_eval: int | None,
        training_mode: Literal["rl", "opd", "sft"],
    ) -> None:
        self.logger = get_logger()
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.inference = inference
        self.train_source = train_source
        self.eval_source = eval_source
        self.group_size = group_size
        self.training_mode = training_mode
        self.max_off_policy_steps_train = max_off_policy_steps_train
        self.max_off_policy_steps_eval = max_off_policy_steps_eval

        # Shared concurrency cap across train + eval.
        self.max_inflight = max_inflight_rollouts
        self.semaphore = asyncio.Semaphore(self.max_inflight)
        self.inflight_permits = 0  # mirror of ``max_inflight - semaphore._value``
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )

        # In-flight tracking. Group IDs are UUIDs so dispatcher restarts /
        # resumed runs don't accidentally collide on a stale counter.
        self.inflight: dict[asyncio.Task, RolloutMeta] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Output queue. Bounded so the dispatcher backpressures on a slow sink.
        self.out_q: asyncio.Queue[Rollout] = asyncio.Queue(maxsize=max(8, self.max_inflight))

        # Scheduling priority.
        self.sched_mode: SchedMode = SchedMode.PREFER_TRAIN

        # All per-poll counters live on this dataclass — see ``DispatcherMetrics``.
        self.metrics = DispatcherMetrics()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    @property
    def model_name(self) -> str:
        """Model name to send on rollout requests — follows ``policy.model_name``
        in non-sft modes, the inference pool's model name in sft (where the
        pool points at the teacher and the policy is irrelevant)."""
        if self.training_mode == "sft":
            return self.inference.model_name
        return self.policy.model_name

    @property
    def inflight_train_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "train")

    @property
    def inflight_eval_count(self) -> int:
        return sum(m.rollout_count for m in self.inflight.values() if m.kind == "eval")

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self.inflight_permits

    @property
    def queued_eval_examples(self) -> int:
        return len(self.eval_source)

    @property
    def max_off_policy_level(self) -> int:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return max(steps) if steps else 0

    @property
    def mean_off_policy_level(self) -> float:
        steps = [m.off_policy_steps for m in self.inflight.values() if m.kind == "train"]
        return sum(steps) / len(steps) if steps else 0.0

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat. Runs until ``stop()``."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly and retry. Eval triggers from
                    # the orchestrator (sync ``eval_source.trigger`` +
                    # ``switch_mode`` mutation) will wake the next iteration.
                    try:
                        await asyncio.wait_for(self.stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                done, _pending = await asyncio.wait(
                    list(self.inflight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (eval triggers, sched_mode flips)
                )
                for task in done:
                    await self.handle_completed_rollout(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_inflight_rollouts()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    # ── observer hook (called by WeightWatcher.on_new_version) ─────────────

    async def on_new_version(self, step: int) -> None:
        """Bump off-policy counters and cancel stale rollouts.

        Both train and eval rollouts get an off-policy tick — train rollouts
        are capped at ``max_off_policy_steps_train``; eval rollouts at
        ``max_off_policy_steps_eval`` (``None`` = uncapped, the default).
        Cancelled rollouts surface to the sink as ``Cancelled`` error markers
        so the group still finalizes.

        Eval epoch *triggering* is the orchestrator's job (after each
        training step) — see ``Orchestrator.process_one_step``.
        """
        stale_groups: dict[uuid.UUID, Kind] = {}
        cancelled_by_kind: dict[Kind, int] = {"train": 0, "eval": 0}
        for meta in self.inflight.values():
            meta.off_policy_steps += 1
            limit = self.max_off_policy_steps_train if meta.kind == "train" else self.max_off_policy_steps_eval
            if limit is not None and meta.off_policy_steps > limit:
                stale_groups[meta.group_id] = meta.kind

        for gid, kind in stale_groups.items():
            removed = await self.drop_group(gid)
            cancelled_by_kind[kind] += removed

        for kind in ("train", "eval"):
            n = cancelled_by_kind[kind]
            if n:
                self.metrics.record_cancellation(kind=kind, n=n)
                limit = self.max_off_policy_steps_train if kind == "train" else self.max_off_policy_steps_eval
                self.logger.warning(
                    f"Cancelled {n} {kind} rollouts past max_off_policy_steps={limit}. "
                    "Consider increasing it to avoid this."
                )

    # ── fill ────────────────────────────────────────────────────────────────

    async def fill_inflight(self) -> None:
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
                if not self.eval_source and self.inflight_eval_count == 0:
                    # All eval examples dispatched and finished — switch back to train.
                    self.switch_mode(SchedMode.PREFER_TRAIN, reason="eval queue drained")
                    continue
                if not self.eval_source:
                    # Eval queue empty but eval still in flight: don't schedule
                    # train here (no hard PREFER_TRAIN flip until eval drains).
                    return
                scheduled = await self.try_schedule("eval")
                if not scheduled:
                    return
            else:  # PREFER_TRAIN
                scheduled = await self.try_schedule("train")
                if not scheduled:
                    return

    def switch_mode(self, new_mode: SchedMode, *, reason: str) -> None:
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
        self.metrics.mode_transitions += 1

    async def try_schedule(self, kind: Kind) -> bool:
        """Schedule one rollout of ``kind``. Same algorithm for train + eval:

        1. Prefer continuing an existing group of this kind that still has
           rollouts to schedule and fits in the available permits (keeps
           prefix-cache hits within a group).
        2. Otherwise open a fresh group from the corresponding source (train:
           ``next_example``, eval: ``peek`` + ``pop``).

        Returns False when nothing could be scheduled (no permits, source
        empty for eval, missing env, etc).
        """
        envs = self.train_envs if kind == "train" else self.eval_envs
        if envs is None:
            return False

        # 1. Continue an existing group of this kind.
        for gid, group in list(self.groups.items()):
            if group.kind != kind or group.rollouts_to_schedule <= 0:
                continue
            env = envs.get(group.env_name)
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= self.available_permits:
                return await self.schedule_group_rollout(gid, group)

        # 2. Open a fresh group.
        fresh = self.next_fresh_group(kind, envs)
        if fresh is None:
            return False
        gid = uuid.uuid4()
        self.groups[gid] = fresh
        return await self.schedule_group_rollout(gid, fresh)

    def next_fresh_group(self, kind: Kind, envs) -> GroupState | None:
        """Resolve the next example to schedule for ``kind`` and reserve a
        ``GroupState`` for it. Returns ``None`` if nothing can be scheduled
        right now (no permits / source empty / missing env)."""
        if kind == "train":
            # Pre-check: a fresh train group needs ``group_size`` permits up
            # front. Avoids pulling an example we can't immediately dispatch.
            if self.available_permits < self.group_size:
                return None
            example = self.train_source.next_example()
            env_name = example["env_name"]
            group_size = self.group_size
            eval_step: int | None = None
        else:
            item = self.eval_source.peek()
            if item is None:
                return None
            env_name, example, eval_step = item
            env = envs.get(env_name)
            if env is None:
                # Stale entry (env removed mid-run) — drop it.
                self.eval_source.pop()
                return None
            group_size = env.config.group_size
            cost = group_size if env.requires_group_scoring else 1
            if cost > self.available_permits:
                return None
            self.eval_source.pop()

        return GroupState(
            kind=kind,
            env_name=env_name,
            example=example,
            rollouts_to_schedule=group_size,
            target_rollouts=group_size,
            eval_step=eval_step,
            policy_version_at_start=self.policy.version,
        )

    async def schedule_group_rollout(self, group_id: uuid.UUID, group: GroupState) -> bool:
        """Dispatch one ``run_rollout`` / ``run_group`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Pick or pin a client for the group. Pinning keeps prefix-cache hits
        # within a group.
        if group.pinned_client is None:
            load = Counter(
                client_identity(m.client_config) for m in self.inflight.values() if m.client_config is not None
            )
            client = await self.inference.select_train_client(load)
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
            await self.acquire(permits)
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
            await self.acquire(permits)
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

    async def acquire(self, n: int) -> None:
        """Acquire ``n`` permits from the shared semaphore. Optionally rate-limit each one."""
        for _ in range(n):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            await self.semaphore.acquire()
            self.inflight_permits += 1

    def release(self, n: int) -> None:
        for _ in range(n):
            self.semaphore.release()
            self.inflight_permits -= 1

    # ── completion handling ────────────────────────────────────────────────

    async def handle_completed_rollout(self, task: asyncio.Task) -> None:
        """Emit every dispatched rollout exactly once to ``out_q``.

        - Successful rollouts → emit as-is.
        - Env-reported errors (``r["error"] is not None``) → emit as-is; the
          sink filters them out.
        - Empty trajectories → annotate ``r["error"] = EmptyTrajectory`` and
          emit; sink treats them the same as any other failure.
        - Task exceptions → synthesize ``meta.rollout_count`` error rollouts
          and emit (a group-scored task that would have produced N rollouts
          surfaces N error markers, so the sink's count-to-``group_size``
          finalization still triggers).

        Cancellations are handled by ``drop_group`` directly (it emits its
        own markers before cancelling the tasks); when the cancelled tasks
        eventually complete with ``CancelledError`` we discard.
        """
        meta = self.inflight.pop(task, None)
        if meta is None:
            return  # already handled by drop_group / cancel_inflight_rollouts
        self.release(meta.rollout_count)
        group = self.groups.get(meta.group_id)

        is_synth_exception = False
        try:
            result = task.result()
            rollouts: list[vf.RolloutOutput] = result if isinstance(result, list) else [result]
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.logger.warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            rollouts = [
                self.error_rollout_output(error_type=type(exc).__name__, error_repr=repr(exc))
                for _ in range(meta.rollout_count)
            ]
            is_synth_exception = True

        self.metrics.record_arrivals(meta.env_name, len(rollouts))

        for r in rollouts:
            if r.get("error") is None and len(r.get("trajectory") or []) == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # can filter it uniformly with other failures.
                r["error"] = {
                    "error": "EmptyTrajectory",
                    "error_chain_repr": "Rollout returned with no trajectory steps",
                    "error_chain_str": "",
                }
                self.metrics.record_empty(meta.env_name)
                self.logger.warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            if r.get("error") is not None:
                err_type = r["error"].get("error", "Unknown")
                self.metrics.record_error(meta.env_name, err_type)
                if not is_synth_exception:
                    self.logger.warning(
                        f"Rollout failed in group {meta.group_id} ({meta.env_name}) — "
                        f"{r['error'].get('error_chain_repr', err_type)}"
                    )
            r["env_name"] = meta.env_name
            await self.emit_rollout(meta, group, r)

    async def emit_rollout(self, meta: RolloutMeta, group: GroupState | None, raw: vf.RolloutOutput) -> None:
        """Put one ``Rollout`` on ``out_q`` and bump per-group emit count.

        Pops the group from ``self.groups`` once every member has been
        emitted, so the dispatcher's group bookkeeping stays bounded.
        """
        example_id = -1
        eval_step = meta.eval_step
        policy_version = meta.policy_version
        if group is not None:
            ex_id = group.example.get("example_id")
            if ex_id is not None:
                example_id = int(ex_id)
            eval_step = group.eval_step
            policy_version = group.policy_version_at_start
            group.emitted += 1
            if group.emitted >= group.target_rollouts:
                self.groups.pop(meta.group_id, None)

        await self.out_q.put(
            Rollout(
                kind=meta.kind,
                env_name=meta.env_name,
                example_id=example_id,
                raw=raw,
                policy_version=policy_version,
                eval_step=eval_step,
            )
        )

    @staticmethod
    def error_rollout_output(*, error_type: str, error_repr: str) -> vf.RolloutOutput:
        """Synthesize a minimal ``vf.RolloutOutput`` carrying just an error.

        Used for rollouts that never produced a real output (task exception,
        off-policy cancellation). The sink filters anything with ``error``
        set out of the trainable pool.
        """
        out: vf.RolloutOutput = vf.RolloutOutput()
        out["error"] = {
            "error": error_type,
            "error_chain_repr": error_repr,
            "error_chain_str": error_repr,
        }
        out["trajectory"] = []
        out["completion"] = None
        out["reward"] = 0.0
        out["is_truncated"] = False
        out["metrics"] = {}
        out["stop_condition"] = None
        return out

    async def drop_group(self, group_id: uuid.UUID) -> int:
        """Cancel any remaining in-flight tasks for this group and emit a
        cancellation marker for each, so the sink can finalize the partial
        group.

        Returns the number of rollouts cancelled (for off-policy metrics).
        """
        group = self.groups.pop(group_id, None)
        tasks_to_cancel: list[asyncio.Task] = []
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            self.inflight.pop(task, None)
            self.release(meta.rollout_count)
            tasks_to_cancel.append(task)
            cancelled += meta.rollout_count
            # Emit a marker per rollout this task would have produced so the
            # sink sees ``group_size`` arrivals overall and finalizes.
            for _ in range(meta.rollout_count):
                raw = self.error_rollout_output(error_type="Cancelled", error_repr="Off-policy cancel")
                raw["env_name"] = meta.env_name
                example_id = -1
                eval_step = meta.eval_step
                policy_version = meta.policy_version
                if group is not None:
                    ex_id = group.example.get("example_id")
                    if ex_id is not None:
                        example_id = int(ex_id)
                    eval_step = group.eval_step
                    policy_version = group.policy_version_at_start
                await self.out_q.put(
                    Rollout(
                        kind=meta.kind,
                        env_name=meta.env_name,
                        example_id=example_id,
                        raw=raw,
                        policy_version=policy_version,
                        eval_step=eval_step,
                    )
                )
        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        return cancelled

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollouts (used on shutdown only).

        Doesn't emit markers — sinks are being torn down anyway and would
        just discard them.
        """
        cancelled_by_kind: dict[Kind, int] = {"train": 0, "eval": 0}
        for meta in self.inflight.values():
            cancelled_by_kind[meta.kind] += meta.rollout_count
            self.release(meta.rollout_count)
        tasks = list(self.inflight.keys())
        for kind in ("train", "eval"):
            if cancelled_by_kind[kind]:
                self.metrics.record_cancellation(kind=kind, n=cancelled_by_kind[kind])
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
            "dispatcher/inflight_permits": float(self.inflight_permits),
            "dispatcher/available_permits": float(self.available_permits),
            "dispatcher/queued_eval_examples": float(len(self.eval_source)),
            "dispatcher/sched_mode": float(self.sched_mode == SchedMode.PREFER_EVAL),
            "dispatcher/mode_transitions": float(self.metrics.mode_transitions),
            "dispatcher/groups_in_flight": float(len(self.groups)),
            "dispatcher/off_policy_level_max": float(self.max_off_policy_level),
            "dispatcher/off_policy_level_mean": self.mean_off_policy_level,
            "dispatcher/eval_epochs_started": float(self.metrics.eval_epochs_started),
        }

    def drain_metrics(self) -> dict[str, float]:
        """Per-poll drain counters (cancellations, errors, etc) — delegated
        to ``DispatcherMetrics.drained`` which clears them on read. Also
        merges in the inference pool's own drained metrics."""
        out = self.metrics.drained()
        out.update(self.inference.get_metrics())
        return out
