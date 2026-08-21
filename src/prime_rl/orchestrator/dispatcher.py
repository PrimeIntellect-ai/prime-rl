"""Dispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight``) is shared across train + eval. One permit is
  one episode: one ``run`` request against an env server. The cap is dynamic —
  the concurrency controller moves it via ``set_limit``; refills are
  burst-capped so a raised (or drained) cap never lands all its prefills at
  once.
- Optional rate limiting via ``AsyncLimiter(tasks_per_minute, 60)``.
- Emit-everything invariant: every dispatched episode eventually reaches
  ``out_q`` exactly once — as a native episode, or covered by its group's
  single ``GroupCancellation`` message when the group is dropped. Failures
  remain episode errors; sinks decide drop / partial-train policy.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to
  schedule next. Transitions are level-triggered (driven by the eval
  source's emptiness), so in-flight episodes of the opposite kind drain
  naturally on either side of an eval boundary.
- ``on_version_pending`` (called by the watcher before the engines pause for
  the weight update) drops train groups already past ``max_off_policy_steps`` — a
  compute-saving early cancel; the sink's queue sweep is what guarantees the
  bound. Eval episodes are measurements for the policy version they started
  with, so they are allowed to finish even if training advances. Train
  episodes sampled from a frozen model never go stale — their generation
  source doesn't change with policy updates.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import verifiers.v1 as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    CancelReason,
    DispatchResult,
    GroupCancellation,
    GroupState,
    InflightEpisode,
    Policy,
    Progress,
    WorkKind,
)
from prime_rl.orchestrator.utils import min_fresh_version
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferenceClients
from prime_rl.utils.logger import get_logger


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class DispatcherMetrics:
    """Per-tick drain counters for the orchestrator's periodic log.
    ``drained()`` returns the current values and clears them; point-in-time
    gauges live on ``Dispatcher.gauges`` instead."""

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


def _trace_task(task: vf.Task) -> vf.TraceTask:
    return vf.TraceTask(
        type=type(task).__name__,
        data=task.data,
        key=task.key,
        hash=task.hash,
    )


def _validate_episode_task(episode: vf.WireEpisode, task: vf.Task) -> None:
    expected = (task.key, task.hash)
    actual = (episode.task.key, episode.task.hash)
    if actual != expected:
        raise ValueError(f"Episode task provenance {actual} does not match dispatched task {expected}")


class Dispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules episodes
    under shared capacity, and emits native verifier episodes to ``out_q``.
    The watcher drives ``on_version_pending`` for staleness
    cancellation; the orchestrator triggers eval epochs."""

    def __init__(
        self,
        *,
        train_envs: TrainEnvs | None,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource | None,
        eval_source: EvalSource | None,
        policy_pool: InferenceClients,
        policy: Policy,
        progress: Progress | None,
        initial_max_inflight: int,
        max_inflight_ceiling: int | None,
        tasks_per_minute: float | None,
        max_off_policy_steps: int,
        run_id: str,
        run_name: str | None,
        on_episode_complete: Callable[[str, str, int, float], None] | None = None,
    ) -> None:
        self.policy = policy
        self.progress = progress
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        # Train rollouts go to the env's generation source; eval always
        # evaluates the policy.
        self.policy_pool = policy_pool
        self.train_source = train_source
        self.eval_source = eval_source
        self.max_off_policy_steps = max_off_policy_steps
        self.run_id = run_id
        self.run_name = run_name
        # ``(env_name, kind, total_tokens, duration_s)`` per completed episode
        self.on_episode_complete = on_episode_complete

        # Starting value of the dynamic cap (the concurrency controller moves
        # it); ``max_inflight_ceiling`` is the configured hard maximum, used to
        # bound ``out_q``
        self.max_inflight = initial_max_inflight
        self.current_inflight = 0
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )
        # Admission smoothing: the pool may only GROW by ``burst_cap`` per
        # window. Replacing a completed episode is always free (each natural
        # completion refunds one admission); only net expansion is metered.
        # This keeps a raised cap or post-drain refill from landing a wall of
        # prefills at once, without rate-limiting fast-turning workloads at
        # steady state.
        self.admission_window_s = 5.0
        self.admission_window_start = time.monotonic()
        self.admissions_in_window = 0
        self.min_burst = max((env.config.group_size for env in train_envs or ()), default=8)

        self.inflight: dict[asyncio.Task, InflightEpisode] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Bounded so the dispatcher backpressures on a slow sink (unbounded
        # when no hard ceiling is configured — the dynamic cap still bounds
        # in-flight work). One entry per episode — the sinks count episodes,
        # never loose traces — plus one ``GroupCancellation`` per dropped group.
        maxsize = max(8, max_inflight_ceiling) if max_inflight_ceiling is not None else 0
        self.out_q: asyncio.Queue[DispatchResult] = asyncio.Queue(maxsize=maxsize)

        self.mode: DispatcherMode = DispatcherMode.PREFER_TRAIN
        # Set by the orchestrator after the final train step; pipeline then
        # winds down without scheduling new train rollouts
        self.train_scheduling_disabled: bool = False
        self.metrics = DispatcherMetrics()

        # Orchestrator-owned gate. When clear, ``fill_inflight`` returns
        # without scheduling new groups. The dispatcher itself doesn't know
        # *why* — the orchestrator toggles this based on step / policy lead.
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    def _train_generation_for(self, env_name: str) -> tuple[InferenceClients, str, bool]:
        """``(pool, model_name, is_live)`` for *train* rollouts of this env —
        eval always uses the policy."""
        assert self.train_envs is not None  # train groups only exist when train is configured
        source = self.train_envs.get(env_name).generation_source
        if source.uses_live_policy:
            return source.pool, self.policy.model_name, True
        return source.pool, source.pool.model_name, False

    @property
    def inflight_train_count(self) -> int:
        return sum(1 for m in self.inflight.values() if m.kind == "train")

    @property
    def inflight_eval_count(self) -> int:
        return sum(1 for m in self.inflight.values() if m.kind == "eval")

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self.current_inflight

    def set_limit(self, max_inflight: int) -> None:
        """Move the in-flight cap (concurrency controller hook). A cap below
        the current in-flight count sheds nothing — admissions just stay
        blocked until enough episodes finish."""
        self.max_inflight = max_inflight

    def cancel_inflight(self, n: int) -> None:
        """Cancel roughly ``n`` in-flight train episodes, youngest groups
        first (least inference spend so far). Called on an overload cut so
        the working set shrinks immediately instead of waiting for the
        over-admitted episodes to finish at thrashed throughput."""
        asyncio.create_task(self._cancel_inflight(n))

    async def _cancel_inflight(self, n: int) -> None:
        # A group's age is its OLDEST member's start: max() would make a
        # long-running group look young the moment it schedules another
        # member, cancelling the most sunk cost instead of the least
        group_age: dict[uuid.UUID, float] = {}
        for meta in self.inflight.values():
            if meta.kind != "train":
                continue
            age = group_age.get(meta.group_id)
            group_age[meta.group_id] = meta.started_at if age is None else min(age, meta.started_at)
        shed = 0
        for group_id in sorted(group_age, key=lambda gid: group_age[gid], reverse=True):
            if shed >= n:
                break
            # Count only live cancellations toward the excess: drop_group's
            # return includes never-dispatched episodes, which free no permits
            live = sum(1 for meta in self.inflight.values() if meta.group_id == group_id)
            await self.drop_group(group_id, reason="overload")
            shed += live
        if shed:
            get_logger().warning(f"Cancelled {shed} youngest in-flight episodes after overload cut")

    def admission_budget(self) -> int:
        """Admissions still allowed in the current burst window."""
        now = time.monotonic()
        if now - self.admission_window_start >= self.admission_window_s:
            self.admission_window_start = now
            self.admissions_in_window = 0
        burst_cap = max(self.min_burst, self.max_inflight // 10)
        return burst_cap - self.admissions_in_window

    @property
    def inflight_by_env(self) -> dict[tuple[WorkKind, str], int]:
        counts: dict[tuple[WorkKind, str], int] = defaultdict(int)
        for meta in self.inflight.values():
            counts[(meta.kind, meta.env_name)] += 1
        return dict(counts)

    @property
    def queued_eval_examples(self) -> int:
        return len(self.eval_source) if self.eval_source is not None else 0

    @property
    def eval_has_work(self) -> bool:
        """Eval has work while its source queue is non-empty OR any opened eval group still has
        rollouts to schedule. An example leaves ``eval_source`` when its group opens
        (``next_fresh_group``), but its ``group_size`` rollouts dispatch one at a time across
        ``fill_inflight`` passes — so the queue can be empty while a group is still mid-schedule."""
        return bool(self.eval_source) or any(
            g.kind == "eval" and g.episodes_to_schedule > 0 for g in self.groups.values()
        )

    @property
    def is_idle(self) -> bool:
        """True once nothing is in flight, no eval work remains (queued *or* a partly-scheduled eval
        group), and ``out_q`` is empty — the pipeline has fully drained."""
        return not self.inflight and not self.eval_has_work and self.out_q.empty()

    def disable_train_scheduling(self) -> None:
        """Stop scheduling new train rollouts; in-flight train + any
        triggered eval drain naturally."""
        self.train_scheduling_disabled = True

    def inflight_staleness(self) -> list[int]:
        """Current staleness of each in-flight live-sourced train episode: the
        version the batch being collected trains on (v{step-1}) minus the
        episode's dispatch version."""
        if self.train_envs is None or self.progress is None:
            return []
        return [
            (self.progress.step - 1) - meta.policy_version
            for meta in self.inflight.values()
            if meta.kind == "train" and self.train_envs.get(meta.env_name).generation_source.uses_live_policy
        ]

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self.task = asyncio.current_task()
        try:
            while not self.stopped.is_set():
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly. Eval triggers from the
                    # orchestrator wake the next iteration via a mode flip
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
                    await self.handle_completed_episode(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_inflight_episodes()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def on_version_pending(self, step: int) -> None:
        """Drop train groups past ``max_off_policy_steps``: a group dispatched at
        v{k} ships at earliest in the batch currently collecting, at staleness
        ``(progress.step - 1) - k`` — beyond the bound it can never train, so
        cut it before more inference sinks in. This is a compute saver; the
        sink's queue sweep is what guarantees no stale episode ships. Eval
        groups never go stale (they are tied to their start-time policy
        version), nor do frozen-sourced train groups (their generation source
        doesn't change with policy updates).

        Runs *before* the inference engines are paused for the weight update so
        the resulting aborts are processed while the engine is still stepping —
        otherwise the orphaned KV transfers crash the decode engine on resume
        (see ``WeightWatcher.apply_policy_update``)."""
        if self.train_envs is None or self.progress is None:
            return
        min_version = min_fresh_version(self.progress.step, self.max_off_policy_steps)
        stale_groups = [
            gid
            for gid, group in self.groups.items()
            if group.kind == "train"
            and self.train_envs.get(group.env_name).generation_source.uses_live_policy
            and group.policy_version_at_start < min_version
        ]
        cancelled = 0
        for gid in stale_groups:
            cancelled += await self.drop_group(gid, reason="stale")

        if cancelled:
            get_logger().warning(
                f"Cancelled {cancelled} train episodes past max_off_policy_steps={self.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    async def on_new_version(self, step: int) -> None:
        """No-op: the dispatcher drains in ``on_version_pending`` (pre-pause)."""

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring
        ``self.mode``. Eval scheduling ignores the orchestrator's dispatch
        gate (evals are version-pinned measurements); only train scheduling
        respects it. When ``PREFER_EVAL``'s source exhausts we flip back to
        ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            if self.available_permits <= 0 or self.admission_budget() <= 0:
                return

            if self.mode == DispatcherMode.PREFER_EVAL:
                # PREFER_EVAL is only entered when the orchestrator triggers
                # eval, which requires ``eval_source`` to be configured
                assert self.eval_source is not None
                if not self.eval_has_work:
                    # Eval source + all eval groups fully dispatched. Flip
                    # to PREFER_TRAIN so any remaining permits go to train
                    # while the in-flight eval tail completes naturally
                    self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                    continue
                scheduled = await self.try_schedule("eval")
                if not scheduled:
                    return
            else:  # PREFER_TRAIN — respects the orchestrator's dispatch gate
                if not self.dispatch_allowed.is_set():
                    return
                scheduled = await self.try_schedule("train")
                if not scheduled:
                    return

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} episodes because {reason}")
        self.mode = new_mode

    async def try_schedule(self, kind: WorkKind) -> bool:
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
            if group.kind != kind or group.episodes_to_schedule <= 0:
                continue
            return await self.schedule_group_episode(gid, group)

        fresh = self.next_fresh_group(kind, envs)
        if fresh is None:
            return False
        gid = uuid.uuid4()
        self.groups[gid] = fresh
        return await self.schedule_group_episode(gid, fresh)

    def next_fresh_group(self, kind: WorkKind, envs) -> GroupState | None:
        """Pop the next task from the corresponding source and wrap it in
        a ``GroupState``. Returns ``None`` if the source is empty."""
        if kind == "train":
            assert self.train_source is not None
            if self.progress is None:
                raise RuntimeError("Train dispatch requires progress state")
            request = self.train_source.next_task(step=self.progress.step)
        else:
            assert self.eval_source is not None
            request = self.eval_source.next_task()
        if request is None:
            return None

        env_name = request.env_name
        group_size = envs.get(env_name).config.group_size

        return GroupState(
            kind=kind,
            env_name=env_name,
            task=request.task,
            step=request.step,
            episodes_to_schedule=group_size,
            target_episodes=group_size,
            policy_version_at_start=self.policy.version,
        )

    async def schedule_group_episode(self, group_id: uuid.UUID, group: GroupState) -> bool:
        """Dispatch one ``run`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Train rollouts use the env's generation source via the
        # renderer/token train client. Eval always evaluates the policy and
        # goes through the eval client (chat-completions) so eval scores stay
        # comparable.
        if group.kind == "eval":
            pool, model_name = self.policy_pool, self.policy.model_name
            live_sourced = True
        else:
            pool, model_name, live_sourced = self._train_generation_for(group.env_name)

        client = pool.eval_client if group.kind == "eval" else pool.train_client

        env_collection = self.train_envs if group.kind == "train" else self.eval_envs
        if env_collection is None:
            return False
        env = env_collection.get(group.env_name)
        # Frozen-sourced train rollouts hit a frozen pool; salting per policy
        # version would invalidate its prefix cache every weight update for
        # no reason.
        if live_sourced:
            cache_salt = str(group.policy_version_at_start)
        else:
            cache_salt = None

        group.episodes_to_schedule -= 1
        await self.acquire()
        self.admissions_in_window += 1
        task = asyncio.create_task(
            env.run(
                client=client,
                model_name=model_name,
                cache_salt=cache_salt,
                task_data=group.task.data.model_dump(mode="json"),
            )
        )

        self.inflight[task] = InflightEpisode(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            task=group.task,
            policy_version=group.policy_version_at_start,
            step=group.step,
            client_config=client,
            started_at=time.monotonic(),
        )
        return True

    async def acquire(self) -> None:
        """Reserve one permit + rate-limit it. Caller must precheck
        ``available_permits >= 1``; this is not a blocking acquire."""
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        self.current_inflight += 1

    def release(self, *, refund_admission: bool = False) -> None:
        """Free one permit. ``refund_admission`` only on natural completions:
        refunding cancelled episodes would hand a mass shed's worth of burst
        budget to the refill while the overload is still draining."""
        self.current_inflight -= 1
        if refund_admission:
            self.admissions_in_window = max(0, self.admissions_in_window - 1)

    async def handle_completed_episode(self, task: asyncio.Task) -> None:
        """Emit every dispatched episode exactly once to ``out_q``."""
        meta = self.inflight.pop(task, None)
        if meta is None:
            return  # already handled by drop_group / cancel_inflight_episodes
        self.release(refund_admission=True)
        group = self.groups.get(meta.group_id)

        is_synth_exception = False
        try:
            episode: vf.WireEpisode = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            get_logger().warning(f"Episode task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            episode = vf.WireEpisode(
                env=vf.EnvInfo(id=meta.env_name),
                task=_trace_task(meta.task),
                ok=False,
                errors=[
                    vf.Error(
                        type=type(exc).__name__,
                        message=str(exc),
                        traceback="".join(traceback.format_exception(exc)),
                    )
                ],
            )
            is_synth_exception = True

        if not episode.traces and episode.ok:
            episode.ok = False
            episode.errors.append(vf.Error(type="EmptyEpisode", message="Episode returned with no traces"))

        for trace in episode.traces:
            if not trace.has_error and trace.num_turns == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # treats it like any other failure (``has_error`` reads ``ok``)
                trace.errors.append(vf.Error(type="EmptyTrajectory", message="Trace returned with no trajectory steps"))
                trace.ok = False
                episode.ok = False
                get_logger().warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            if trace.has_error:
                self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
                if not is_synth_exception and trace.last_error is not None:
                    get_logger().warning(
                        f"Trace failed in group {meta.group_id} ({meta.env_name}) — "
                        f"{trace.last_error.type}: {trace.last_error.message}"
                    )
        if not episode.ok and not episode.traces:
            self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
        if not is_synth_exception and self.on_episode_complete is not None and meta.started_at > 0:
            self.on_episode_complete(
                meta.env_name, meta.kind, episode.num_total_tokens, time.monotonic() - meta.started_at
            )
        await self.emit_episode(meta, group, episode)

    async def emit_episode(
        self,
        meta: InflightEpisode,
        group: GroupState | None,
        episode: vf.WireEpisode,
    ) -> None:
        """Stamp one completed episode with its dispatch provenance and emit it."""
        _validate_episode_task(episode, meta.task)
        policy_version = meta.policy_version
        if group is not None:
            policy_version = group.policy_version_at_start
            group.emitted += 1
            if group.emitted >= group.target_episodes:
                self.groups.pop(meta.group_id, None)

        episode.env.name = meta.env_name
        episode.group = vf.GroupInfo(id=str(meta.group_id))
        live_policy = meta.kind == "eval"
        if meta.kind == "train":
            assert self.train_envs is not None
            live_policy = self.train_envs.get(meta.env_name).generation_source.uses_live_policy
        policy = vf.PolicySpan(start=policy_version, end=self.policy.version) if live_policy else None
        work: vf.WorkInfo = (
            vf.EvalWorkInfo(step=meta.step, policy=policy)
            if meta.kind == "eval"
            else vf.TrainWorkInfo(step=meta.step, policy=policy)
        )
        run = vf.TrainRunInfo(id=self.run_id, name=self.run_name, work=work)
        episode.record_run(run)
        await self.out_q.put(episode)

    async def drop_group(self, group_id: uuid.UUID, *, reason: CancelReason) -> int:
        """Cancel this group's remaining in-flight tasks and emit one
        ``GroupCancellation`` covering every episode it still owes the sink (both
        in-flight and never-dispatched), so count-to-``group_size``
        finalization still fires. Returns the owed count for metrics."""
        group = self.groups.pop(group_id, None)
        # Sync claim phase: pop matching tasks from ``self.inflight`` and
        # release their permits in one non-yielding sweep. After this loop
        # the dropped tasks are no longer reachable from ``self.inflight``,
        # so ``handle_completed_episode``'s existing None-guard makes the
        # subsequent async emit phase race-free.
        claimed: list[tuple[asyncio.Task, InflightEpisode]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            del self.inflight[task]
            self.release()
            claimed.append((task, meta))

        inflight_cancelled = len(claimed)
        unscheduled_cancelled = group.episodes_to_schedule if group is not None else 0
        cancelled = inflight_cancelled + unscheduled_cancelled

        if cancelled > 0:
            # ``group`` can be ``None`` only if every episode was already
            # emitted — then nothing is in flight or unscheduled, so kind/env
            # always resolve from the group or a claimed meta.
            kind = group.kind if group is not None else claimed[-1][1].kind
            env_name = group.env_name if group is not None else claimed[-1][1].env_name
            self.metrics.record_cancellation(kind=kind, env_name=env_name, n=cancelled)
            get_logger().debug(
                f"drop {kind} | group={str(group_id)[:8]} env={env_name} reason={reason} | "
                f"cancelled={cancelled} (inflight={inflight_cancelled} unscheduled={unscheduled_cancelled})"
            )
            await self.out_q.put(
                GroupCancellation(
                    kind=kind,
                    env_name=env_name,
                    group_id=str(group_id),
                    count=cancelled,
                    reason=reason,
                )
            )

        if claimed:
            await safe_cancel_all([task for task, _ in claimed])
        return cancelled

    async def cancel_inflight_episodes(self) -> None:
        """Cancel all in-flight episodes. Used on shutdown — doesn't emit
        markers since the sinks are being torn down anyway."""
        for meta in self.inflight.values():
            self.metrics.record_cancellation(kind=meta.kind, env_name=meta.env_name)
            self.release()
        tasks = list(self.inflight.keys())
        self.inflight.clear()
        self.groups.clear()
        if tasks:
            await safe_cancel_all(tasks)

    async def cancel_inflight_train_episodes(self) -> int:
        """Cancel in-flight train episodes, leaving eval alone. Used by the
        orchestrator at ``max_steps`` so triggered eval can still complete
        through the pipeline while wasted train inference is short-circuited."""
        train_tasks: list[asyncio.Task] = []
        train_group_ids: set[uuid.UUID] = set()
        cancelled = 0
        for task, meta in list(self.inflight.items()):
            if meta.kind != "train":
                continue
            self.inflight.pop(task, None)
            self.release()
            self.metrics.record_cancellation(kind="train", env_name=meta.env_name)
            cancelled += 1
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
        staleness = self.inflight_staleness()
        return {
            "dispatcher/inflight/train": float(self.inflight_train_count),
            "dispatcher/inflight/eval": float(self.inflight_eval_count),
            "dispatcher/queued/eval": float(self.queued_eval_examples),
            "dispatcher/mode": float(self.mode == DispatcherMode.PREFER_EVAL),
            "dispatcher/off_policy/max": float(max(staleness, default=0)),
            "dispatcher/off_policy/mean": sum(staleness) / len(staleness) if staleness else 0.0,
        }
