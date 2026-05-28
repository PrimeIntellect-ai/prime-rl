"""RolloutDispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight_rollouts``) is shared across train + eval.
  A group-scoring task that runs N rollouts in one call reserves N permits.
- Optional rate limiting via ``AsyncLimiter(tasks_per_minute, 60)``.
- Emit-everything invariant: every dispatched rollout eventually reaches
  ``out_q`` exactly once as a ``TrainRollout`` / ``EvalRollout``. Failures
  (env error, empty trajectory, task exception, off-policy cancel) carry
  ``raw["error"]`` set; sinks decide drop / partial-train policy.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to
  schedule next. Transitions are level-triggered (driven by the eval
  source's emptiness), so in-flight rollouts of the opposite kind drain
  naturally on either side of an eval boundary.
- ``on_new_version`` (called by the watcher) bumps ``off_policy_steps`` on
  every in-flight rollout and drops groups past ``max_off_policy_steps``.
  Cancellations surface as synthetic ``Cancelled`` markers so the sink's
  count-to-``group_size`` finalization still fires.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import verifiers as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    EvalRollout,
    FinishedRollout,
    GroupState,
    InflightRollout,
    Kind,
    Policy,
    TrainRollout,
)
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool, client_identity
from prime_rl.utils.logger import get_logger


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class DispatcherMetrics:
    """Per-tick drain counters for the orchestrator's periodic log.
    ``drained()`` returns the current values and clears them; point-in-time
    gauges live on ``RolloutDispatcher.gauges`` instead."""

    cancelled_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    errored_by_kind_env: dict[tuple[Literal["train", "eval"], str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_cancellation(self, *, kind: Literal["train", "eval"], env_name: str, n: int = 1) -> None:
        self.cancelled_by_kind_env[(kind, env_name)] += n

    def record_error(self, *, kind: Literal["train", "eval"], env_name: str) -> None:
        self.errored_by_kind_env[(kind, env_name)] += 1

    def drained(self, *, train_envs: set[str], eval_envs: set[str]) -> dict[str, float]:
        """Return per-tick counters and clear them. Emits the full pre-
        registered key set every tick (zero when no activity) so the wandb
        time axis stays dense and ``define_metric`` lines up."""
        out: dict[str, float] = {}
        for kind in ("train", "eval"):
            envs = train_envs if kind == "train" else eval_envs
            cancelled_total = sum(self.cancelled_by_kind_env.get((kind, e), 0) for e in envs)
            errored_total = sum(self.errored_by_kind_env.get((kind, e), 0) for e in envs)
            out[f"dispatcher/cancelled/{kind}"] = float(cancelled_total)
            out[f"dispatcher/errored/{kind}"] = float(errored_total)
        for env in train_envs | eval_envs:
            out[f"dispatcher/cancelled/{env}"] = float(
                self.cancelled_by_kind_env.get(("train", env), 0) + self.cancelled_by_kind_env.get(("eval", env), 0)
            )
            out[f"dispatcher/errored/{env}"] = float(
                self.errored_by_kind_env.get(("train", env), 0) + self.errored_by_kind_env.get(("eval", env), 0)
            )
        self.cancelled_by_kind_env.clear()
        self.errored_by_kind_env.clear()
        return out

    @staticmethod
    def drain_keys(*, train_envs: set[str], eval_envs: set[str]) -> list[str]:
        """Full set of keys ``drained`` may emit; used by the periodic
        logger for ``wandb.define_metric``."""
        keys = [
            "dispatcher/cancelled/train",
            "dispatcher/cancelled/eval",
            "dispatcher/errored/train",
            "dispatcher/errored/eval",
        ]
        for env in train_envs | eval_envs:
            keys.append(f"dispatcher/cancelled/{env}")
            keys.append(f"dispatcher/errored/{env}")
        return keys


class RolloutDispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules
    rollouts under shared capacity, and emits ``FinishedRollout``\\ s to
    ``out_q``. The watcher drives ``on_new_version`` for off-policy
    cancellation; the orchestrator triggers eval epochs."""

    def __init__(
        self,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource,
        eval_source: EvalSource | None,
        inference: InferencePool,
        policy: Policy,
        max_inflight_rollouts: int,
        tasks_per_minute: float | None,
        max_off_policy_steps: int,
        training_mode: Literal["rl", "opd", "sft"],
    ) -> None:
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.inference = inference
        self.train_source = train_source
        self.eval_source = eval_source
        self.training_mode = training_mode
        self.max_off_policy_steps = max_off_policy_steps

        self.max_inflight = max_inflight_rollouts
        self.inflight_permits = 0
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )

        self.inflight: dict[asyncio.Task, InflightRollout] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Bounded so the dispatcher backpressures on a slow sink.
        self.out_q: asyncio.Queue[FinishedRollout] = asyncio.Queue(maxsize=max(8, self.max_inflight))

        self.mode: DispatcherMode = DispatcherMode.PREFER_TRAIN
        # Set by the orchestrator after the final train step; pipeline then
        # winds down without scheduling new train rollouts.
        self.train_scheduling_disabled: bool = False
        self.metrics = DispatcherMetrics()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    @property
    def model_name(self) -> str:
        """In SFT mode the policy is irrelevant — rollouts go to the teacher
        pool, so use its model name. Otherwise follow the live policy."""
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
    def inflight_by_env(self) -> dict[tuple[Kind, str], int]:
        counts: dict[tuple[Kind, str], int] = defaultdict(int)
        for meta in self.inflight.values():
            counts[(meta.kind, meta.env_name)] += meta.rollout_count
        return dict(counts)

    @property
    def queued_eval_examples(self) -> int:
        return len(self.eval_source) if self.eval_source is not None else 0

    @property
    def is_idle(self) -> bool:
        """True once nothing is in flight, no eval queued, and ``out_q`` is
        empty — the pipeline has fully drained."""
        eval_drained = self.eval_source is None or not self.eval_source
        return not self.inflight and eval_drained and self.out_q.empty()

    def disable_train_scheduling(self) -> None:
        """Stop scheduling new train rollouts; in-flight train + any
        triggered eval drain naturally."""
        self.train_scheduling_disabled = True

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
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly. Eval triggers from the
                    # orchestrator wake the next iteration via a mode flip.
                    try:
                        await asyncio.wait_for(self.stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue

                done, _pending = await asyncio.wait(
                    list(self.inflight.keys()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (mode flips)
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

    async def on_new_version(self, step: int) -> None:
        """Bump off-policy counters and drop groups past
        ``max_off_policy_steps`` (drop_group emits ``Cancelled`` markers so
        the sink still finalizes the partial group)."""
        stale_groups: dict[uuid.UUID, Kind] = {}
        cancelled_by_kind: dict[Kind, int] = {"train": 0, "eval": 0}
        for meta in self.inflight.values():
            meta.off_policy_steps += 1
            if meta.off_policy_steps > self.max_off_policy_steps:
                stale_groups[meta.group_id] = meta.kind

        for gid, kind in stale_groups.items():
            removed = await self.drop_group(gid)
            cancelled_by_kind[kind] += removed

        for kind in ("train", "eval"):
            n = cancelled_by_kind[kind]
            if n:
                get_logger().warning(
                    f"Cancelled {n} {kind} rollouts past max_off_policy_steps={self.max_off_policy_steps}. "
                    "Consider increasing it to avoid this."
                )

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring
        ``self.mode``. When ``PREFER_EVAL``'s source exhausts we flip back
        to ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            if self.available_permits <= 0:
                return

            if self.mode == DispatcherMode.PREFER_EVAL:
                # PREFER_EVAL is only entered when the orchestrator triggers
                # eval, which requires ``eval_source`` to be configured.
                assert self.eval_source is not None
                eval_has_work = bool(self.eval_source) or any(
                    g.kind == "eval" and g.rollouts_to_schedule > 0 for g in self.groups.values()
                )
                if not eval_has_work:
                    # Eval source + all eval groups fully dispatched. Flip
                    # to PREFER_TRAIN so any remaining permits go to train
                    # while the in-flight eval tail completes naturally.
                    self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                    continue
                scheduled = await self.try_schedule("eval")
                if not scheduled:
                    return
            else:  # PREFER_TRAIN
                scheduled = await self.try_schedule("train")
                if not scheduled:
                    return

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} rollouts because {reason}")
        self.mode = new_mode

    async def try_schedule(self, kind: Kind) -> bool:
        """Schedule one rollout of ``kind``: prefer continuing an existing
        group (keeps prefix-cache hits); otherwise open a fresh group from
        the corresponding source. Returns False if nothing could be
        scheduled."""
        if kind == "train" and self.train_scheduling_disabled:
            return False
        envs = self.train_envs if kind == "train" else self.eval_envs
        if envs is None:
            return False

        for gid, group in list(self.groups.items()):
            if group.kind != kind or group.rollouts_to_schedule <= 0:
                continue
            env = envs.get(group.env_name)
            cost = group.rollouts_to_schedule if env.requires_group_scoring else 1
            if cost <= self.available_permits:
                return await self.schedule_group_rollout(gid, group)

        fresh = self.next_fresh_group(kind, envs)
        if fresh is None:
            return False
        gid = uuid.uuid4()
        self.groups[gid] = fresh
        return await self.schedule_group_rollout(gid, fresh)

    def next_fresh_group(self, kind: Kind, envs) -> GroupState | None:
        """Pop the next example from the corresponding source and wrap it in
        a ``GroupState``. Returns ``None`` if the source is empty or the
        picked env's permit cost doesn't fit."""
        if kind == "train":
            source = self.train_source
        else:
            assert self.eval_source is not None
            source = self.eval_source
        example = source.next_example(self.available_permits)
        if example is None:
            return None

        env_name = example["env_name"]
        group_size = envs.get(env_name).config.group_size
        eval_step: int | None = example.get("eval_step") if kind == "eval" else None

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
        # Pin a single client per group to keep prefix-cache hits.
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

        self.inflight[task] = InflightRollout(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            policy_version=group.policy_version_at_start,
            rollout_count=permits,
            client_config=client,
            eval_step=group.eval_step,
        )
        get_logger().debug(
            f"dispatch {group.kind} | group={str(group_id)[:8]} env={group.env_name} "
            f"ex={group.example['example_id']} | permits={permits} v={group.policy_version_at_start}"
        )
        return True

    async def acquire(self, n: int) -> None:
        """Reserve ``n`` permits + rate-limit each one. Caller must precheck
        ``available_permits >= n``; this is not a blocking acquire."""
        for _ in range(n):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()
            self.inflight_permits += 1

    def release(self, n: int) -> None:
        self.inflight_permits -= n

    async def handle_completed_rollout(self, task: asyncio.Task) -> None:
        """Emit every dispatched rollout exactly once to ``out_q``. Task
        exceptions synthesize ``meta.rollout_count`` error markers so the
        sink's count-to-``group_size`` finalization still triggers.
        Cancelled tasks (popped by ``drop_group``) raise ``CancelledError``
        and are discarded — ``drop_group`` already emitted their markers.
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
            get_logger().warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            rollouts = [
                self.error_rollout_output(error_type=type(exc).__name__, error_repr=repr(exc))
                for _ in range(meta.rollout_count)
            ]
            is_synth_exception = True

        for r in rollouts:
            if r.get("error") is None and len(r.get("trajectory") or []) == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # treats it like any other failure.
                r["error"] = {
                    "error": "EmptyTrajectory",
                    "error_chain_repr": "Rollout returned with no trajectory steps",
                    "error_chain_str": "",
                }
                get_logger().warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            if r.get("error") is not None:
                err_type = r["error"].get("error", "Unknown")
                self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
                if not is_synth_exception:
                    get_logger().warning(
                        f"Rollout failed in group {meta.group_id} ({meta.env_name}) — "
                        f"{r['error'].get('error_chain_repr', err_type)}"
                    )
            await self.emit_rollout(meta, group, r)

        n_errored = sum(1 for r in rollouts if r.get("error") is not None)
        get_logger().debug(
            f"complete {meta.kind} | group={str(meta.group_id)[:8]} env={meta.env_name} | "
            f"rollouts={len(rollouts)} errored={n_errored}"
        )

    async def emit_rollout(self, meta: InflightRollout, group: GroupState | None, raw: vf.RolloutOutput) -> None:
        """Build a ``TrainRollout`` / ``EvalRollout`` and put it on ``out_q``.
        Pops the group from ``self.groups`` once every member has been emitted."""
        eval_step = meta.eval_step
        policy_version = meta.policy_version
        example_id = raw.get("example_id")
        if group is not None:
            eval_step = group.eval_step
            policy_version = group.policy_version_at_start
            example_id = group.example["example_id"]
            group.emitted += 1
            if group.emitted >= group.target_rollouts:
                self.groups.pop(meta.group_id, None)

        common = dict(
            raw=raw,
            env_name=meta.env_name,
            example_id=example_id if example_id is not None else -1,
            group_id=meta.group_id,
            policy_version=policy_version,
            off_policy_steps=meta.off_policy_steps,
        )
        rollout: FinishedRollout
        if meta.kind == "train":
            rollout = TrainRollout(**common)
        else:
            assert eval_step is not None, "eval rollout missing eval_step"
            rollout = EvalRollout(**common, eval_step=eval_step)
        await self.out_q.put(rollout)

    @staticmethod
    def error_rollout_output(*, error_type: str, error_repr: str) -> vf.RolloutOutput:
        """Minimal ``vf.RolloutOutput`` for rollouts that never produced
        real output (task exception, off-policy cancel)."""
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
        out["token_usage"] = {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "final_input_tokens": 0.0,
            "final_output_tokens": 0.0,
        }
        return out

    async def drop_group(self, group_id: uuid.UUID) -> int:
        """Cancel remaining in-flight tasks for this group and emit a
        ``Cancelled`` marker for every rollout it still owes the sink
        (both in-flight and not-yet-scheduled). Returns the count for
        off-policy metrics."""
        group = self.groups.pop(group_id, None)
        tasks_to_cancel: list[asyncio.Task] = []
        inflight_cancelled = 0
        last_meta: InflightRollout | None = None
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            self.inflight.pop(task, None)
            self.release(meta.rollout_count)
            tasks_to_cancel.append(task)
            inflight_cancelled += meta.rollout_count
            last_meta = meta
            for _ in range(meta.rollout_count):
                raw = self.error_rollout_output(error_type="Cancelled", error_repr="Off-policy cancel")
                await self.emit_rollout(meta, group, raw)

        # For non-group-scoring envs, the group may have rollouts that
        # were never dispatched (``rollouts_to_schedule > 0``). Emit
        # markers for those too so the sink hits ``target_rollouts``.
        #
        # ``last_meta`` can be ``None`` if the only inflight task for this
        # group completed naturally between ``on_new_version``'s snapshot
        # and us reaching it — synthesize a stand-in from the group state.
        unscheduled_cancelled = 0
        if group is not None and group.rollouts_to_schedule > 0:
            fallback_meta = last_meta or InflightRollout(
                kind=group.kind,
                env_name=group.env_name,
                group_id=group_id,
                policy_version=group.policy_version_at_start,
                rollout_count=1,
                eval_step=group.eval_step,
            )
            unscheduled_cancelled = group.rollouts_to_schedule
            for _ in range(unscheduled_cancelled):
                raw = self.error_rollout_output(error_type="Cancelled", error_repr="Off-policy cancel")
                await self.emit_rollout(fallback_meta, group, raw)

        cancelled = inflight_cancelled + unscheduled_cancelled
        if cancelled > 0:
            meta_for_log = last_meta or (
                InflightRollout(
                    kind=group.kind,
                    env_name=group.env_name,
                    group_id=group_id,
                    policy_version=group.policy_version_at_start if group else 0,
                    rollout_count=1,
                    eval_step=group.eval_step,
                )
                if group is not None
                else None
            )
            if meta_for_log is not None:
                self.metrics.record_cancellation(kind=meta_for_log.kind, env_name=meta_for_log.env_name, n=cancelled)
                get_logger().debug(
                    f"drain {meta_for_log.kind} | group={str(group_id)[:8]} env={meta_for_log.env_name} | "
                    f"cancelled={cancelled} (inflight={inflight_cancelled} unscheduled={unscheduled_cancelled})"
                )

        if tasks_to_cancel:
            await safe_cancel_all(tasks_to_cancel)
        return cancelled

    async def cancel_inflight_rollouts(self) -> None:
        """Cancel all in-flight rollouts. Used on shutdown — doesn't emit
        markers since the sinks are being torn down anyway."""
        for meta in self.inflight.values():
            self.metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name, n=meta.rollout_count)
            self.release(meta.rollout_count)
        tasks = list(self.inflight.keys())
        self.inflight.clear()
        self.groups.clear()
        if tasks:
            await safe_cancel_all(tasks)

    async def cancel_inflight_train_rollouts(self) -> int:
        """Cancel in-flight train rollouts, leaving eval alone. Used by the
        orchestrator at ``max_steps`` so triggered eval can still complete
        through the pipeline while wasted train inference is short-circuited."""
        train_tasks: list[asyncio.Task] = []
        train_group_ids: set[uuid.UUID] = set()
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.kind != "train":
                continue
            self.inflight.pop(task, None)
            self.release(meta.rollout_count)
            self.metrics.record_cancellation(kind="train", env_name=meta.env_name, n=meta.rollout_count)
            cancelled += meta.rollout_count
            train_tasks.append(task)
            train_group_ids.add(meta.group_id)
        for gid in train_group_ids:
            self.groups.pop(gid, None)
        if train_tasks:
            await safe_cancel_all(train_tasks)
        return cancelled

    # ── metrics ────────────────────────────────────────────────────────────

    def gauges(self) -> dict[str, float]:
        """Instantaneous, read-only gauges sampled by the periodic logger."""
        return {
            "dispatcher/inflight_train": float(self.inflight_train_count),
            "dispatcher/inflight_eval": float(self.inflight_eval_count),
            "dispatcher/queued/eval": float(self.queued_eval_examples),
            "dispatcher/mode": float(self.mode == DispatcherMode.PREFER_EVAL),
            "dispatcher/groups_in_flight": float(len(self.groups)),
            "dispatcher/off_policy_level_max": float(self.max_off_policy_level),
            "dispatcher/off_policy_level_mean": self.mean_off_policy_level,
        }
