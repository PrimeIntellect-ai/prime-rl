"""Dispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight``) is shared across train + eval. One permit is one
  episode: one ``run`` request against an env server. The cap is dynamic — the
  concurrency controller moves it via ``set_limit``; refills are burst-capped so a
  raised (or drained) cap never lands all its prefills at once.
- Optional rate limiting via ``AsyncLimiter(dispatch_per_minute, 60)``, one token per episode.
- Every dispatched attempt reaches the bound consumer exactly once, as one
  ``vf.Episode``: the one the environment returned, or one synthesized here for a
  request that produced none (failed, or cancelled with the group). Results queue
  through a bounded buffer so a slow consumer backpressures scheduling; ``on_train``
  and ``on_eval`` receive them by kind.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to schedule
  next. Transitions are level-triggered (driven by the eval source's emptiness), so
  in-flight episodes of the opposite kind drain naturally on either side of an eval
  boundary.
- ``on_version_pending`` (called by the watcher before the engines pause for the
  weight update) drops train groups already past ``max_off_policy_steps`` — a
  compute-saving early cancel; the queue's sweep is what guarantees the bound. Eval
  episodes are measurements for the policy version they started with; online evals
  may cancel them when a newer checkpoint is ready. Train episodes sampled from a
  frozen model never go stale.
"""

from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import verifiers.v1 as vf
from aiolimiter import AsyncLimiter

from prime_rl import monitors as default_monitors
from prime_rl.orchestrator.annotations import stamp_arrival
from prime_rl.orchestrator.clients import InferenceClient
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.live import LiveStream
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import CANCELLED, InflightEpisode, Kind, is_cancelled
from prime_rl.utils.async_utils import safe_cancel, safe_cancel_all
from prime_rl.utils.logger import get_logger

# Admission smoothing: per window the in-flight pool may grow by at most this
# share of its cap (floored at MIN_BURST, or a train env's group_size).
ADMISSION_WINDOW = 5.0
ADMISSION_FRACTION = 0.1
MIN_BURST = 8


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


@dataclass
class GroupState:
    """A group being scheduled: its task and how many rollouts it still owes."""

    kind: Kind
    env_name: str
    task: vf.Task
    step: int
    version: int
    to_schedule: int


EpisodeHook = Callable[[vf.Episode], Awaitable[None]]


class Dispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules episodes
    under shared capacity, and hands each result to the consumer bound for its
    kind. The watcher drives ``on_version_pending`` for staleness cancellation;
    the evaluator switches the mode when an eval epoch fires."""

    def __init__(
        self,
        *,
        dispatch_per_minute: int | None,
        train_envs: TrainEnvs | None,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource | None,
        eval_source: EvalSource | None,
        policy_clients: InferenceClient,
        initial_max_inflight: int,
        max_inflight_ceiling: int | None,
        max_off_policy_steps: int = 0,
        run_id: str,
        run_name: str | None,
    ) -> None:
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        # Train rollouts go to the env's generation source; eval always evaluates the policy.
        self.policy_clients = policy_clients
        self.train_source = train_source
        self.eval_source = eval_source
        self.max_off_policy_steps = max_off_policy_steps
        self.run_id = run_id
        self.run_name = run_name

        # Outbound hooks, bound by the owner. ``step`` is the batch being collected,
        # ``version`` the policy inference serves.
        self._step: Callable[[], int] = lambda: 1
        self._version: Callable[[], int] = lambda: 0
        self._on_train: EpisodeHook | None = None
        self._on_eval: EpisodeHook | None = None
        # ``(env_name, kind, total_tokens, duration_s)`` per completed episode
        self._on_episode_complete: Callable[[str, str, int, float], None] | None = None
        self.monitors: Any = default_monitors
        self.live = LiveStream()

        # Starting value of the dynamic cap (the concurrency controller moves it);
        # ``max_inflight_ceiling`` is the configured hard maximum, used to bound the result buffer
        self.max_inflight = initial_max_inflight
        self.current_inflight = 0
        self.rate_limiter = AsyncLimiter(dispatch_per_minute, time_period=60) if dispatch_per_minute else None
        # Admission smoothing: the pool may only GROW by ``burst_cap`` per window.
        # Replacing a completed episode is always free (each natural completion refunds
        # one admission); only net expansion is metered, so a raised cap or post-drain
        # refill never lands a wall of prefills at once.
        self.admission_window_start = time.monotonic()
        self.admissions_in_window = 0
        self.min_burst = max((env.config.group_size for env in train_envs or ()), default=MIN_BURST)

        self.inflight: dict[asyncio.Task, InflightEpisode] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}

        # Bounded so the dispatcher backpressures on a slow consumer (unbounded when no
        # hard ceiling is configured — the dynamic cap still bounds in-flight work).
        maxsize = max(8, max_inflight_ceiling) if max_inflight_ceiling is not None else 0
        self.results: asyncio.Queue[vf.Episode] = asyncio.Queue(maxsize=maxsize)
        self.undelivered = 0
        self.deliver_task: asyncio.Task | None = None

        self.mode = DispatcherMode.PREFER_TRAIN
        # Set once the final train step ships; the pipeline then winds down without
        # scheduling new train rollouts
        self.train_scheduling_disabled = False
        # Externally driven gate: when closed, no new train groups are scheduled.
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()
        self.policy_update_pending = False
        self.scheduling_lock = asyncio.Lock()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    def bind(
        self,
        *,
        step: Callable[[], int] | None = None,
        version: Callable[[], int] | None = None,
        on_train: EpisodeHook | None = None,
        on_eval: EpisodeHook | None = None,
        on_episode_complete: Callable[[str, str, int, float], None] | None = None,
        monitors: Any = None,
    ) -> None:
        if step is not None:
            self._step = step
        if version is not None:
            self._version = version
        if on_train is not None:
            self._on_train = on_train
        if on_eval is not None:
            self._on_eval = on_eval
        if on_episode_complete is not None:
            self._on_episode_complete = on_episode_complete
        if monitors is not None:
            self.monitors = monitors
            self.live.monitors = monitors

    # ── state others read ──────────────────────────────────────────────────

    def inflight_count(self, kind: Kind) -> int:
        return sum(1 for meta in self.inflight.values() if meta.kind == kind)

    @property
    def available_permits(self) -> int:
        return self.max_inflight - self.current_inflight

    @property
    def eval_has_work(self) -> bool:
        """Eval has work while its source queue is non-empty or any opened eval group
        still has rollouts to schedule (a group's rollouts dispatch one at a time)."""
        return bool(self.eval_source) or any(g.kind == "eval" and g.to_schedule > 0 for g in self.groups.values())

    @property
    def is_idle(self) -> bool:
        """Nothing in flight, no eval work left, every result delivered: fully drained."""
        return not self.inflight and not self.eval_has_work and self.undelivered == 0

    def inflight_staleness(self) -> list[int]:
        """Current staleness of each in-flight live-sourced train episode: the version
        the batch being collected trains on (v{step-1}) minus the dispatch version."""
        return [
            (self._step() - 1) - meta.policy_version
            for meta in self.inflight.values()
            if meta.kind == "train" and self.uses_live_policy(meta.kind, meta.env_name)
        ]

    def uses_live_policy(self, kind: Kind, env_name: str) -> bool:
        if kind == "eval":
            return True
        assert self.train_envs is not None
        return self.train_envs.get(env_name).generation_source.uses_live_policy

    # ── inbound ────────────────────────────────────────────────────────────

    def set_limit(self, max_inflight: int) -> None:
        """Move the in-flight cap (concurrency controller hook). A cap below the current
        in-flight count sheds nothing — admissions just stay blocked until enough finish."""
        self.max_inflight = max_inflight

    def gate(self, open: bool) -> None:
        """Open or close train scheduling; eval ignores the gate."""
        if open:
            self.dispatch_allowed.set()
        else:
            self.dispatch_allowed.clear()

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} episodes because {reason}")
        self.mode = new_mode

    def cancel_inflight(self, n: int) -> None:
        """Cancel roughly ``n`` in-flight train episodes, youngest groups first (least
        inference spend so far), so an overload cut shrinks the working set at once."""
        asyncio.create_task(self._cancel_inflight(n))

    async def _cancel_inflight(self, n: int) -> None:
        # A group's age is its OLDEST member's start: max() would make a long-running
        # group look young the moment it schedules another member
        group_age: dict[uuid.UUID, float] = {}
        for meta in self.inflight.values():
            if meta.kind == "train":
                group_age[meta.group_id] = min(group_age.get(meta.group_id, meta.started_at), meta.started_at)
        shed = 0
        for group_id in sorted(group_age, key=lambda gid: group_age[gid], reverse=True):
            if shed >= n:
                break
            # Only live cancellations count toward the excess: never-dispatched
            # episodes free no permits
            shed += sum(1 for meta in self.inflight.values() if meta.group_id == group_id)
            await self.drop_group(group_id, reason="overload")
        if shed:
            get_logger().warning(f"Cancelled {shed} youngest in-flight episodes after overload cut")

    async def on_version_pending(self, step: int) -> None:
        """Drop train groups past ``max_off_policy_steps`` before the engines pause for
        the weight update: a group dispatched at v{k} ships at earliest in the batch
        currently collecting, at staleness ``(step - 1) - k`` — beyond the bound it can
        never train, so cut it before more inference sinks in. Runs before the pause so
        the aborts are processed while the engine is still stepping; aborts after resume
        race the flush of KV transfers and crash the decode engine."""
        self.policy_update_pending = True
        # Wait for a scheduling call that started before the pending update: no
        # rollout can cross the inference weight swap after this barrier.
        async with self.scheduling_lock:
            pass
        min_version = (self._step() - 1) - self.max_off_policy_steps
        stale = [
            gid
            for gid, group in self.groups.items()
            if group.kind == "train" and self.uses_live_policy("train", group.env_name) and group.version < min_version
        ]
        cancelled = 0
        for gid in stale:
            cancelled += await self.drop_group(gid, reason="stale")
        if cancelled:
            get_logger().warning(
                f"Cancelled {cancelled} train episodes past max_off_policy_steps={self.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    async def on_new_version(self, step: int) -> None:
        """Resume rollout scheduling after inference applies the new policy."""
        self.policy_update_pending = False

    async def drain_train(self, reason: str) -> None:
        """Stop scheduling train work and cancel what is in flight; triggered eval
        epochs still run to completion."""
        self.train_scheduling_disabled = True
        cancelled = await self.cancel_all("train")
        get_logger().info(
            f"{reason} — draining pipeline (cancelled {cancelled} in-flight train episode(s); "
            "any in-flight evals will complete)"
        )

    async def cancel_eval_step(self, step: int) -> int:
        """Cancel queued and active eval groups for a superseded checkpoint. Scheduling
        stays paused until ``on_new_version`` runs after the replacement weights are live."""
        if self.eval_source is None or self.eval_envs is None:
            return 0
        self.policy_update_pending = True
        async with self.scheduling_lock:
            queued = self.eval_source.cancel_step(step)
            group_ids = [gid for gid, group in self.groups.items() if group.kind == "eval" and group.step == step]
        cancelled = 0
        for group_id in group_ids:
            cancelled += await self.drop_group(group_id, reason="superseded")
        for request in queued:
            count = request.rollouts or self.eval_envs.get(request.env_name).config.group_size
            group = GroupState("eval", request.env_name, request.task, request.step, self._version(), count)
            for _ in range(count):
                await self.emit(self.cancelled_episode(uuid.uuid4(), group, "superseded"))
            cancelled += count
        return cancelled

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Single dispatch loop: schedule, wait, collect, repeat."""
        self.task = asyncio.current_task()
        self.live.start()
        self.deliver_task = asyncio.create_task(self.deliver(), name="dispatcher_deliver")
        try:
            while not self.stopped.is_set():
                self._raise_if_deliver_failed()
                await self.fill_inflight()
                if not self.inflight:
                    # No work — sleep briefly; an eval trigger wakes the next
                    # iteration via a mode flip
                    try:
                        await asyncio.wait_for(self.stopped.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
                    continue
                done, _ = await asyncio.wait(
                    list(self.inflight),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.5,  # wake periodically to re-check fill (mode flips)
                )
                for task in done:
                    await self.handle_completed(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_all(None)
        if self.deliver_task is not None:
            await safe_cancel(self.deliver_task)
            self.deliver_task = None
        await self.live.stop()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def cancel_all(self, kind: Kind | None) -> int:
        """Cancel every in-flight episode of ``kind`` (all when None) without a marker:
        the pipeline is draining or being torn down, so nothing waits for them."""
        tasks = []
        for task, meta in list(self.inflight.items()):
            if kind is not None and meta.kind != kind:
                continue
            del self.inflight[task]
            self.release()
            self.live.retired(meta)
            self.groups.pop(meta.group_id, None)
            tasks.append(task)
        if kind is None:
            self.groups.clear()
        if tasks:
            await safe_cancel_all(tasks)
        return len(tasks)

    # ── delivery ───────────────────────────────────────────────────────────

    def _raise_if_deliver_failed(self) -> None:
        task = self.deliver_task
        if task is not None and task.done() and not task.cancelled() and task.exception() is not None:
            raise task.exception()

    async def emit(self, episode: vf.Episode) -> None:
        """Queue one episode for delivery; blocks while the buffer is full. A consumer
        that died surfaces here instead of leaving the producer parked on a full buffer."""
        self._raise_if_deliver_failed()
        self.undelivered += 1
        put = asyncio.ensure_future(self.results.put(episode))
        waiting = {put, self.deliver_task} if self.deliver_task is not None else {put}
        done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        if put not in done:
            await safe_cancel(put)
            self.undelivered -= 1
            self._raise_if_deliver_failed()
            raise RuntimeError("dispatcher result consumer exited")
        put.result()

    async def deliver(self) -> None:
        """Hand queued episodes to the consumer bound for their kind, in order. Every
        episode the environment returned also lands in the ``all`` trace stream the
        moment it arrives, so it survives crashes and drains. Train episodes belong to
        the batch window currently collecting, eval episodes to the step whose eval
        triggered them."""
        while True:
            episode = await self.results.get()
            try:
                if not isinstance(episode.run, vf.TrainRunInfo):
                    raise ValueError("Orchestrated episode is missing training-run provenance")
                kind = episode.run.work.type
                if episode.traces or not is_cancelled(episode):
                    step = episode.run.work.step if kind == "eval" else self._step()
                    stamp_arrival([episode], kind, step)
                    await self.monitors.log([episode], step, kind, "all")
                hook = self._on_eval if kind == "eval" else self._on_train
                if hook is None:
                    raise RuntimeError(f"Dispatcher has no consumer bound for {kind} results")
                await hook(episode)
            finally:
                self.undelivered -= 1

    # ── scheduling ─────────────────────────────────────────────────────────

    def admission_budget(self) -> int:
        """Admissions still allowed in the current burst window."""
        now = time.monotonic()
        if now - self.admission_window_start >= ADMISSION_WINDOW:
            self.admission_window_start = now
            self.admissions_in_window = 0
        burst_cap = max(self.min_burst, int(self.max_inflight * ADMISSION_FRACTION))
        return burst_cap - self.admissions_in_window

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring ``self.mode``. Eval
        scheduling ignores the dispatch gate (evals are version-pinned measurements);
        only train scheduling respects it. When ``PREFER_EVAL``'s source exhausts we flip
        back to ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            if self.policy_update_pending:
                return
            if self.available_permits <= 0 or self.admission_budget() <= 0:
                return
            async with self.scheduling_lock:
                if self.policy_update_pending:
                    return
                if self.mode == DispatcherMode.PREFER_EVAL:
                    if not self.eval_has_work:
                        self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                        continue
                    kind: Kind = "eval"
                else:
                    if not self.dispatch_allowed.is_set() or self.train_scheduling_disabled:
                        return
                    kind = "train"
                if not await self.try_schedule(kind):
                    return

    async def try_schedule(self, kind: Kind) -> bool:
        """Schedule one rollout of ``kind``: prefer continuing an existing group (keeps
        prefix-cache hits); otherwise open a fresh group from the corresponding source.
        Returns False if nothing could be scheduled."""
        for gid, group in self.groups.items():
            if group.kind == kind and group.to_schedule > 0:
                await self.schedule_episode(gid, group)
                return True
        fresh = self.next_group(kind)
        if fresh is None:
            return False
        gid, group = fresh
        self.groups[gid] = group
        await self.schedule_episode(gid, group)
        return True

    def next_group(self, kind: Kind) -> tuple[uuid.UUID, GroupState] | None:
        """Pop the next task from the corresponding source; None when it is empty."""
        if kind == "train":
            if self.train_source is None or self.train_envs is None:
                return None
            request = self.train_source.next_task(step=self._step())
            envs = self.train_envs
        else:
            if self.eval_source is None or self.eval_envs is None:
                return None
            request = self.eval_source.next_task()
            envs = self.eval_envs
        if request is None:
            return None
        rollouts = request.rollouts or envs.get(request.env_name).config.group_size
        gid = uuid.UUID(request.group_id) if request.group_id else uuid.uuid4()
        return gid, GroupState(kind, request.env_name, request.task, request.step, self._version(), rollouts)

    async def schedule_episode(self, group_id: uuid.UUID, group: GroupState) -> None:
        """Dispatch one ``run`` task for this group. Train rollouts use the env's
        generation source via the renderer/token train client; eval always evaluates
        the policy through the chat-completions eval client so scores stay comparable."""
        if group.kind == "eval":
            assert self.eval_envs is not None
            env = self.eval_envs.get(group.env_name)
            clients, model_name, client = (
                self.policy_clients,
                self.policy_clients.model_name,
                self.policy_clients.eval_client,
            )
        else:
            assert self.train_envs is not None
            env = self.train_envs.get(group.env_name)
            source = env.generation_source
            clients, client = source.clients, source.clients.train_client
            model_name = self.policy_clients.model_name if source.uses_live_policy else source.clients.model_name
        # Frozen-sourced train rollouts hit a frozen pool; salting per policy version
        # would invalidate its prefix cache every weight update for no reason.
        cache_salt = str(group.version) if self.uses_live_policy(group.kind, group.env_name) else None

        group.to_schedule -= 1
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()
        self.current_inflight += 1
        self.admissions_in_window += 1
        meta = InflightEpisode(
            kind=group.kind,
            env_name=group.env_name,
            group_id=group_id,
            task=group.task,
            policy_version=group.version,
            step=group.step,
            started_at=time.monotonic(),
        )
        session_ids: set[str] = set()

        def on_delta(delta: dict) -> None:
            session_ids.add(delta["trace"])
            self.live.delta(meta, delta)

        async def run_episode() -> vf.Episode:
            try:
                episode = await env.run(
                    client=client,
                    model_name=model_name,
                    cache_salt=cache_salt,
                    task_data=group.task.data.model_dump(mode="json"),
                    on_delta=on_delta,
                )
                session_ids.update(trace.id for trace in episode.traces)
                return episode
            finally:
                cleanup = asyncio.create_task(clients.finish_sessions(sorted(session_ids)))
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    await cleanup
                    raise

        task = asyncio.create_task(run_episode())
        self.inflight[task] = meta
        self.live.dispatched(meta)

    def release(self, *, refund_admission: bool = False) -> None:
        """Free one permit. ``refund_admission`` only on natural completions: refunding
        cancelled episodes would hand a mass shed's worth of burst budget to the refill
        while the overload is still draining."""
        self.current_inflight -= 1
        if refund_admission:
            self.admissions_in_window = max(0, self.admissions_in_window - 1)

    # ── completion ─────────────────────────────────────────────────────────

    async def handle_completed(self, task: asyncio.Task) -> None:
        """Emit the terminal episode of one dispatched environment request."""
        meta = self.inflight.pop(task, None)
        if meta is None:
            return  # already handled by drop_group / cancel_all
        self.live.retired(meta)
        self.release(refund_admission=True)
        group = self.groups.get(meta.group_id)
        try:
            episode: vf.Episode = task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            get_logger().warning(f"Environment request failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            error = vf.Error(
                type=type(exc).__name__, message=str(exc), traceback="".join(traceback.format_exception(exc))
            )
            episode = self.blank_episode(meta.task, error)
        else:
            if (episode.task.key, episode.task.hash) != (meta.task.key, meta.task.hash):
                raise ValueError(
                    f"Episode task provenance {(episode.task.key, episode.task.hash)} does not match "
                    f"dispatched task {(meta.task.key, meta.task.hash)}"
                )
            if not episode.traces and episode.ok:
                episode.ok = False
                episode.errors.append(vf.Error(type="EmptyEpisode", message="Episode returned with no traces"))
            for trace in episode.traces:
                if not trace.has_error and trace.num_turns == 0:
                    # Promote an empty trajectory to an explicit error so the sinks treat
                    # it like any other failure (``has_error`` reads ``ok``)
                    trace.errors.append(
                        vf.Error(type="EmptyTrajectory", message="Trace returned with no trajectory steps")
                    )
                    trace.ok = False
                    episode.ok = False
                    get_logger().warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
                elif trace.has_error and trace.last_error is not None:
                    get_logger().warning(
                        f"Trace failed in group {meta.group_id} ({meta.env_name}) — "
                        f"{trace.last_error.type}: {trace.last_error.message}"
                    )
            if self._on_episode_complete is not None and meta.started_at > 0:
                self._on_episode_complete(
                    meta.env_name, meta.kind, episode.num_total_tokens, time.monotonic() - meta.started_at
                )
        self.stamp(episode, meta.kind, meta.env_name, meta.group_id, meta.step, meta.policy_version)
        if (
            group is not None
            and group.to_schedule <= 0
            and not any(m.group_id == meta.group_id for m in self.inflight.values())
        ):
            self.groups.pop(meta.group_id, None)
        await self.emit(episode)

    async def drop_group(self, group_id: uuid.UUID, *, reason: str) -> int:
        """Cancel this group's remaining in-flight tasks and emit one cancelled episode
        for every attempt it still owes the sink (in flight and never dispatched), so
        count-to-``group_size`` finalization still fires. Returns the owed count."""
        group = self.groups.pop(group_id, None)
        # Claim the tasks in one non-yielding sweep: once they are out of ``inflight``,
        # ``handle_completed``'s None-guard makes the async emit phase race-free.
        claimed: list[tuple[asyncio.Task, InflightEpisode]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id == group_id:
                del self.inflight[task]
                self.release()
                self.live.retired(meta)
                claimed.append((task, meta))
        unscheduled = group.to_schedule if group is not None else 0
        cancelled = len(claimed) + unscheduled
        if cancelled:
            if group is None:
                # Every episode was emitted or is claimed here, so the last claimed one
                # describes the group.
                meta = claimed[-1][1]
                group = GroupState(meta.kind, meta.env_name, meta.task, meta.step, meta.policy_version, 0)
            get_logger().debug(
                f"Dropped {group.kind} group | group={str(group_id)[:8]} env={group.env_name} reason={reason} | "
                f"cancelled={cancelled} (inflight={len(claimed)} unscheduled={unscheduled})"
            )
            for _ in range(cancelled):
                await self.emit(self.cancelled_episode(group_id, group, reason))
        if claimed:
            await safe_cancel_all([task for task, _ in claimed])
        return cancelled

    def blank_episode(self, task: vf.Task, error: vf.Error) -> vf.Episode:
        """An episode for an attempt that returned none, carrying the reason."""
        trace_task = vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash)
        return vf.Episode(task=trace_task, ok=False, errors=[error])

    def cancelled_episode(self, group_id: uuid.UUID, group: GroupState, reason: str) -> vf.Episode:
        episode = self.blank_episode(group.task, vf.Error(type=CANCELLED, message=reason))
        self.stamp(episode, group.kind, group.env_name, group_id, group.step, group.version)
        return episode

    def stamp(
        self, episode: vf.Episode, kind: Kind, env_name: str, group_id: uuid.UUID, step: int, version: int
    ) -> None:
        """Record the dispatch provenance every consumer reads."""
        episode.env.name = env_name
        episode.group = vf.GroupInfo(id=str(group_id))
        policy = vf.PolicySpan(start=version, end=self._version()) if self.uses_live_policy(kind, env_name) else None
        work: vf.WorkInfo = (
            vf.EvalWorkInfo(step=step, policy=policy) if kind == "eval" else vf.TrainWorkInfo(step=step, policy=policy)
        )
        episode.record_run(vf.TrainRunInfo(id=self.run_id, name=self.run_name, work=work))

    # ── observability ──────────────────────────────────────────────────────

    def status(self) -> str:
        """``N inflight episodes (train=.., eval=..)``, per env when a kind has several."""
        train, eval_ = self.inflight_count("train"), self.inflight_count("eval")
        part = f"{train + eval_} inflight episodes (train={train}, eval={eval_}"
        envs = [("train", env.name) for env in self.train_envs or ()] + [
            ("eval", env.name) for env in self.eval_envs or ()
        ]
        if len(envs) > 2 or (len(envs) == 2 and envs[0][0] == envs[1][0]):
            counts = {key: 0 for key in envs}
            for meta in self.inflight.values():
                counts[(meta.kind, meta.env_name)] = counts.get((meta.kind, meta.env_name), 0) + 1
            part += " | " + ", ".join(f"{name}={counts[(kind, name)]}" for kind, name in envs)
        return part + ")"

    def gauges(self) -> dict[str, float]:
        staleness = self.inflight_staleness()
        return {
            "dispatcher/inflight/train": float(self.inflight_count("train")),
            "dispatcher/inflight/eval": float(self.inflight_count("eval")),
            "dispatcher/queued/eval": float(len(self.eval_source) if self.eval_source is not None else 0),
            "dispatcher/queued/results": float(self.undelivered),
            "dispatcher/mode": float(self.mode == DispatcherMode.PREFER_EVAL),
            "dispatcher/off_policy/max": float(max(staleness, default=0)),
            "dispatcher/off_policy/mean": sum(staleness) / len(staleness) if staleness else 0.0,
        }
