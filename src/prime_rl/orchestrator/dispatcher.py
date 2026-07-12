"""RolloutDispatcher: schedules rollouts under a shared permit counter.

- Capacity (``max_inflight_rollouts``) is shared across train + eval.
  A group-scoring task that runs N rollouts in one call reserves N permits.
- Emit-everything invariant: every dispatched rollout reaches ``out_q`` once.
  Failures carry ``trace.error``; sinks decide drop / partial-train policy.
- ``DispatcherMode.PREFER_TRAIN`` / ``PREFER_EVAL`` controls which kind to
  schedule next. Transitions are level-triggered (driven by the eval
  source's emptiness), so in-flight rollouts of the opposite kind drain
  naturally on either side of an eval boundary.
- Dynamo fences and settles eval/live-policy work before worker pause; other
  backends retain the off-policy window. Frozen-pool rollouts survive.
  Cancellations surface as synthetic ``Cancelled`` markers so the sink's
  count-to-``group_size`` finalization still fires.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter, defaultdict
from enum import Enum, auto

import verifiers.v1 as vf
from aiolimiter import AsyncLimiter

from prime_rl.orchestrator.dispatcher_metrics import DispatcherMetrics
from prime_rl.orchestrator.dispatcher_transactions import (
    EmissionRecord,
    EmissionTracker,
    emit_policy_cancellation_markers,
    settle_transaction_cleanup,
)
from prime_rl.orchestrator.envs import EvalEnvs, TrainEnvs
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.policy_gate import MutablePolicyGate, PolicyUpdateToken, SchedulingEpoch
from prime_rl.orchestrator.pool_identity import client_may_alias_pool, pools_may_alias
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    GroupState,
    InflightRollout,
    Policy,
    Rollout,
    RolloutKind,
)
from prime_rl.utils.async_utils import gather_shielded, safe_cancel, safe_cancel_all
from prime_rl.utils.client import InferencePool, client_identity
from prime_rl.utils.logger import get_logger


class DispatcherMode(Enum):
    """Which kind of work the dispatcher schedules next."""

    PREFER_TRAIN = auto()
    PREFER_EVAL = auto()


class RolloutDispatcher:
    """``await dispatcher.start()`` runs the dispatch loop until ``stop()``.
    Pulls examples from ``TrainSource`` / ``EvalSource``, schedules
    rollouts under shared capacity, and emits ``Rollout``\\ s to
    ``out_q``. The watcher drives ``on_version_pending`` for the policy-update
    barrier; the orchestrator triggers eval epochs."""

    def __init__(
        self,
        *,
        train_envs: TrainEnvs,
        eval_envs: EvalEnvs | None,
        train_source: TrainSource,
        eval_source: EvalSource | None,
        policy_pool: InferencePool,
        policy: Policy,
        max_inflight_rollouts: int,
        tasks_per_minute: float | None,
        max_off_policy_steps: int,
        enforce_policy_update_barrier: bool,
        policy_gate: MutablePolicyGate | None = None,
    ) -> None:
        self.policy = policy
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        # Train rollouts go to the env sampler's pool; eval always
        # evaluates the policy.
        self.policy_pool = policy_pool
        self.train_source = train_source
        self.eval_source = eval_source
        self.max_off_policy_steps = max_off_policy_steps
        self.enforce_policy_update_barrier = enforce_policy_update_barrier
        self.policy_gate = policy_gate or MutablePolicyGate(policy, enabled=enforce_policy_update_barrier)
        self._policy_update: tuple[int, PolicyUpdateToken] | None = None

        self.max_inflight = max_inflight_rollouts
        self.inflight_permits = 0
        self.rate_limiter: AsyncLimiter | None = (
            AsyncLimiter(tasks_per_minute, time_period=60) if tasks_per_minute else None
        )

        self.inflight: dict[asyncio.Task, InflightRollout] = {}
        self.groups: dict[uuid.UUID, GroupState] = {}
        self._emissions = EmissionTracker()

        # Bounded so the dispatcher backpressures on a slow sink
        self.out_q: asyncio.Queue[Rollout] = asyncio.Queue(maxsize=max(8, self.max_inflight))

        self.mode: DispatcherMode = DispatcherMode.PREFER_TRAIN
        # Set after the final train step to wind down new scheduling.
        self.train_scheduling_disabled: bool = False
        self.metrics = DispatcherMetrics()

        # Orchestrator-owned step/policy-lead gate.
        self.dispatch_allowed = asyncio.Event()
        self.dispatch_allowed.set()

        self.stopped = asyncio.Event()
        self.task: asyncio.Task | None = None

    def _train_pool_for(self, env_name: str) -> tuple[InferencePool, str, bool]:
        """``(pool, model_name, is_live)`` for *train* rollouts of this env —
        the env sampler's pool. (Eval always uses the policy.)"""
        sampler = self.train_envs.get(env_name).sampler
        if sampler.samples_from_live_policy:
            return sampler.pool, self.policy.model_name, True
        return sampler.pool, sampler.pool.model_name, False

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
    def inflight_by_env(self) -> dict[tuple[RolloutKind, str], int]:
        counts: dict[tuple[RolloutKind, str], int] = defaultdict(int)
        for meta in self.inflight.values():
            counts[(meta.kind, meta.env_name)] += meta.rollout_count
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
            g.kind == "eval" and g.rollouts_to_schedule > 0 for g in self.groups.values()
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
                    await self.handle_completed_rollout(task)
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        self.stopped.set()
        await self.cancel_inflight_rollouts()
        if self.task is not None:
            await safe_cancel(self.task)
            self.task = None

    async def on_version_pending(self, step: int) -> None:
        """Prepare for mutation: Dynamo fences/drains mutable-policy requests
        before worker pause; other backends retain their off-policy window.
        Pre-pause cancellation lets P/D abort and connector cleanup settle."""
        if not self.enforce_policy_update_barrier:
            await self._advance_off_policy_window()
            return

        token = await self.policy_gate.begin_update(step=step)
        self._policy_update = (step, token)
        try:
            claimed_groups, claimed_tasks, claimed_emissions = self._claim_mutable_policy_work()
            results, cancellation = await gather_shielded(
                self._settle_policy_requests(claimed_groups, claimed_tasks, claimed_emissions),
                self.policy_gate.wait_idle(),
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            primary: BaseException | None = cancellation or (failures[0] if failures else None)
            if primary is not None:
                siblings = failures if cancellation is not None else failures[1:]
                for sibling in siblings:
                    primary.add_note(f"Another policy barrier drain failed: {sibling!r}")
                raise primary
        except BaseException as primary_error:
            await settle_transaction_cleanup(
                self._reopen_policy_admission(step, token),
                primary_error,
                "Policy transition rollback",
            )
            raise

    async def on_new_version(self, step: int) -> None:
        """Reopen admission after the new policy is live."""
        if self.enforce_policy_update_barrier:
            await self._reopen_policy_admission(step)

    async def on_version_update_failed(self, step: int, error: BaseException) -> None:
        """Roll back only the transition fence; the policy stays unchanged."""
        if self.enforce_policy_update_barrier:
            await self._reopen_policy_admission(step)

    @property
    def policy_update_pending(self) -> bool:
        return self.policy_gate.pending

    async def _reopen_policy_admission(self, step: int, token: PolicyUpdateToken | None = None) -> None:
        owned = self._policy_update
        if owned is None or owned[0] != step or (token is not None and owned[1] is not token):
            raise RuntimeError(f"Dispatcher does not own the pending policy transition for step {step}")
        await self.policy_gate.finish_update(owned[1])
        self._policy_update = None

    async def _advance_off_policy_window(self) -> None:
        """Retain the established non-Dynamo in-flight tolerance policy."""
        stale_groups: set[uuid.UUID] = set()
        for meta in self.inflight.values():
            if meta.kind != "train":
                continue
            meta.uses_mutable_policy = meta.uses_mutable_policy or self._uses_mutable_policy(meta.kind, meta.env_name)
            if not meta.uses_mutable_policy:
                continue
            meta.off_policy_steps += 1
            if meta.off_policy_steps > self.max_off_policy_steps:
                stale_groups.add(meta.group_id)

        cancelled = 0
        for group_id in stale_groups:
            cancelled += await self.drop_group(group_id)
        if cancelled:
            get_logger().warning(
                f"Cancelled {cancelled} train rollouts past max_off_policy_steps={self.max_off_policy_steps}. "
                "Consider increasing it to avoid this."
            )

    def _uses_mutable_policy(self, kind: RolloutKind, env_name: str) -> bool:
        if kind == "eval":
            return True
        pool, _model_name, samples_from_live_policy = self._train_pool_for(env_name)
        # A separately-constructed "frozen" pool is safe only when its model
        # and request/admin endpoints do not alias the mutable policy service.
        return samples_from_live_policy or pools_may_alias(pool, self.policy_pool)

    def _claim_mutable_policy_work(
        self,
    ) -> tuple[
        dict[uuid.UUID, GroupState],
        list[tuple[asyncio.Task, InflightRollout]],
        list[EmissionRecord],
    ]:
        group_ids: set[uuid.UUID] = set()
        for group_id, group in self.groups.items():
            group.uses_mutable_policy = group.uses_mutable_policy or self._uses_mutable_policy(
                group.kind, group.env_name
            )
            if group.uses_mutable_policy:
                group_ids.add(group_id)
        for meta in self.inflight.values():
            meta.uses_mutable_policy = meta.uses_mutable_policy or self._uses_mutable_policy(meta.kind, meta.env_name)
            if meta.uses_mutable_policy:
                group_ids.add(meta.group_id)

        claimed_groups = {
            group_id: group for group_id in group_ids if (group := self.groups.pop(group_id, None)) is not None
        }
        claimed_tasks: list[tuple[asyncio.Task, InflightRollout]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id not in group_ids:
                continue
            del self.inflight[task]
            self.release(meta.rollout_count)
            claimed_tasks.append((task, meta))
        claimed_emissions = self._emissions.claim(group_ids)
        return claimed_groups, claimed_tasks, claimed_emissions

    async def _settle_policy_requests(
        self,
        groups: dict[uuid.UUID, GroupState],
        claimed: list[tuple[asyncio.Task, InflightRollout]],
        emissions: list[EmissionRecord],
    ) -> None:
        tasks = [task for task, _meta in claimed]
        already_settled = {task for task in tasks if task.done()}
        for task in tasks:
            task.cancel()

        results: list[object] = []
        cancellation: asyncio.CancelledError | None = None
        if tasks:
            results, cancellation = await gather_shielded(*tasks)

        emission_results: list[object] = []
        emission_cancellation: asyncio.CancelledError | None = None
        if emissions:
            emission_results, emission_cancellation = await gather_shielded(
                *(record.done.wait() for record in emissions)
            )

        metadata_by_group: dict[uuid.UUID, InflightRollout] = {}
        for _task, meta in claimed:
            metadata_by_group.setdefault(meta.group_id, meta)

        marker_results, marker_cancellation = await gather_shielded(
            emit_policy_cancellation_markers(
                groups,
                metadata_by_group,
                out_q=self.out_q,
                stopped=self.stopped,
                metrics=self.metrics,
            )
        )

        failures = [
            result
            for task, result in zip(tasks, results, strict=True)
            if task not in already_settled
            and isinstance(result, BaseException)
            and not isinstance(result, asyncio.CancelledError)
        ]
        failures.extend(result for result in marker_results if isinstance(result, BaseException))
        failures.extend(record.error for record in emissions if record.error is not None)
        failures.extend(result for result in emission_results if isinstance(result, BaseException))
        primary: BaseException | None = (
            cancellation or emission_cancellation or marker_cancellation or (failures[0] if failures else None)
        )
        if primary is not None:
            siblings = (
                failures
                if cancellation is not None or emission_cancellation is not None or marker_cancellation is not None
                else failures[1:]
            )
            for sibling in siblings:
                primary.add_note(f"Another policy barrier cleanup failed: {sibling!r}")
            raise primary

    async def fill_inflight(self) -> None:
        """Schedule new rollouts up to ``max_inflight``, honoring
        ``self.mode``. Eval scheduling ignores the orchestrator's dispatch
        gate (evals are version-pinned measurements); only train scheduling
        respects it. When ``PREFER_EVAL``'s source exhausts we flip back to
        ``PREFER_TRAIN`` so the eval tail drains alongside fresh train."""
        while True:
            epoch = await self.policy_gate.scheduling_epoch()
            if epoch is None or self.available_permits <= 0:
                return

            if self.mode == DispatcherMode.PREFER_EVAL:
                # PREFER_EVAL implies a configured eval source.
                assert self.eval_source is not None
                if not self.eval_has_work:
                    # Fill remaining permits with train while eval drains.
                    self.switch_mode(DispatcherMode.PREFER_TRAIN, reason="the eval queue drained")
                    continue
                scheduled = await self.try_schedule("eval", epoch=epoch)
                if not scheduled:
                    return
            else:  # PREFER_TRAIN — respects the orchestrator's dispatch gate
                if not self.dispatch_allowed.is_set():
                    return
                scheduled = await self.try_schedule("train", epoch=epoch)
                if not scheduled:
                    return

    def switch_mode(self, new_mode: DispatcherMode, *, reason: str) -> None:
        if new_mode == self.mode:
            return
        prefer = "eval" if new_mode == DispatcherMode.PREFER_EVAL else "train"
        get_logger().info(f"Switching dispatcher mode to prefer {prefer} rollouts because {reason}")
        self.mode = new_mode

    async def try_schedule(self, kind: RolloutKind, *, epoch: SchedulingEpoch) -> bool:
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
                return await self.schedule_group_rollout(gid, group, epoch=epoch)

        # Pop and publish a fresh group inside the short scheduling commit.
        # An update either sees the group in its claim snapshot or invalidates
        # this epoch before the source is consumed.
        async with self.policy_gate.scheduling_commit(epoch) as admitted:
            if not admitted or self.available_permits <= 0:
                return False
            fresh = self.next_fresh_group(kind, envs)
            if fresh is None:
                return False
            gid = uuid.uuid4()
            self.groups[gid] = fresh
        return await self.schedule_group_rollout(gid, fresh, epoch=epoch)

    def next_fresh_group(self, kind: RolloutKind, envs) -> GroupState | None:
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
            task_idx=example["task_idx"],
            rollouts_to_schedule=group_size,
            target_rollouts=group_size,
            eval_step=eval_step,
            policy_version_at_start=self.policy.version,
            uses_mutable_policy=self._uses_mutable_policy(kind, env_name),
        )

    async def schedule_group_rollout(
        self,
        group_id: uuid.UUID,
        group: GroupState,
        *,
        epoch: SchedulingEpoch,
    ) -> bool:
        """Dispatch one ``run_rollout`` / ``run_group`` task for this group.

        Returns False only if we couldn't even schedule one rollout (no clients
        ready, no permits). Returns True after issuing one task — the caller
        loops to keep scheduling.
        """
        # Train rollouts use the env sampler's pool via the
        # renderer/token train client. Eval always evaluates the policy and
        # goes through the eval client (chat-completions) — the same path the
        # legacy orchestrator used, so eval scores stay comparable.
        if group.kind == "eval":
            pool, model_name = self.policy_pool, self.policy.model_name
            live_sourced = True
        else:
            pool, model_name, live_sourced = self._train_pool_for(group.env_name)

        # Resolve a client and rate-limit outside the gate. Both operations may
        # wait indefinitely for elastic discovery or quota replenishment.
        if group.pinned_client is None:
            if group.kind == "eval":
                client = await pool.get_eval_client()
            else:
                load = Counter(
                    client_identity(m.client_config) for m in self.inflight.values() if m.client_config is not None
                )
                client = await pool.select_train_client(load)
            if group_id not in self.groups:
                return False
        else:
            client = group.pinned_client

        env_collection = self.train_envs if group.kind == "train" else self.eval_envs
        if env_collection is None:
            return False
        env = env_collection.get(group.env_name)
        if env.requires_group_scoring:
            permits = group.rollouts_to_schedule
        else:
            permits = 1
        # Snapshot selected identity before elastic churn can occur at the next await.
        group.uses_mutable_policy = (
            group.uses_mutable_policy
            or live_sourced
            or pools_may_alias(pool, self.policy_pool)
            or client_may_alias_pool(client, self.policy_pool)
        )
        await self._wait_for_rate_limit(permits)

        async with self.policy_gate.scheduling_commit(epoch) as admitted:
            if (
                not admitted
                or self.groups.get(group_id) is not group
                or permits > self.available_permits
                or group.rollouts_to_schedule < permits
                or (group.kind == "train" and (self.train_scheduling_disabled or not self.dispatch_allowed.is_set()))
            ):
                return False

            group.pinned_client = client
            cache_salt = str(group.policy_version_at_start) if group.uses_mutable_policy else None
            if env.requires_group_scoring:
                group.rollouts_to_schedule = 0
                task: asyncio.Task = asyncio.create_task(
                    env.run_group(
                        client=client,
                        task_idx=group.task_idx,
                        model_name=model_name,
                        group_size=permits,
                        cache_salt=cache_salt,
                    )
                )
            else:
                group.rollouts_to_schedule -= 1
                task = asyncio.create_task(
                    env.run_rollout(
                        client=client,
                        task_idx=group.task_idx,
                        model_name=model_name,
                        cache_salt=cache_salt,
                    )
                )

            self.inflight_permits += permits
            self.inflight[task] = InflightRollout(
                kind=group.kind,
                env_name=group.env_name,
                group_id=group_id,
                policy_version=group.policy_version_at_start,
                rollout_count=permits,
                client_config=client,
                eval_step=group.eval_step,
                uses_mutable_policy=group.uses_mutable_policy,
            )
            return True

    async def _wait_for_rate_limit(self, n: int) -> None:
        for _ in range(n):
            if self.rate_limiter is not None:
                await self.rate_limiter.acquire()

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
        async with self._emissions.track(meta.group_id):
            await self._emit_completed_task(task, meta, group)

    async def _emit_completed_task(
        self,
        task: asyncio.Task,
        meta: InflightRollout,
        group: GroupState | None,
    ) -> None:
        """Convert one settled task into its complete output transaction."""

        is_synth_exception = False
        try:
            result = task.result()
            rollouts: list[Rollout] = result if isinstance(result, list) else [result]
        except asyncio.CancelledError:
            return
        except Exception as exc:
            get_logger().warning(f"Rollout task failed in group {meta.group_id} ({meta.env_name}): {exc!r}")
            task_idx = group.task_idx if group is not None else -1
            rollouts = [
                Rollout(task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)))
                for _ in range(meta.rollout_count)
            ]
            for r in rollouts:
                r.capture_error(exc)
            is_synth_exception = True

        for r in rollouts:
            if not r.has_error and r.num_turns == 0:
                # Empty trajectory: promote to an explicit error so the sink
                # treats it like any other failure
                r.errors.append(vf.Error(type="EmptyTrajectory", message="Rollout returned with no trajectory steps"))
                get_logger().warning(f"Empty trajectory in group {meta.group_id} ({meta.env_name})")
            if r.has_error:
                self.metrics.record_error(kind=meta.kind, env_name=meta.env_name)
                if not is_synth_exception:
                    get_logger().warning(
                        f"Rollout failed in group {meta.group_id} ({meta.env_name}) — {r.error.type}: {r.error.message}"
                    )
            await self.emit_rollout(meta, group, r)

    async def emit_rollout(self, meta: InflightRollout, group: GroupState | None, rollout: Rollout) -> None:
        """Stamp prime-rl metadata onto the completed rollout and put it on ``out_q``.
        Pops the group from ``self.groups`` once every member has been emitted."""
        eval_step = meta.eval_step
        policy_version = meta.policy_version
        if group is not None:
            eval_step = group.eval_step
            policy_version = group.policy_version_at_start

        rollout.kind = meta.kind
        rollout.env_name = meta.env_name
        rollout.group_id = meta.group_id
        rollout.policy_version = policy_version
        rollout.off_policy_steps = meta.off_policy_steps
        if meta.kind == "eval":
            assert eval_step is not None, "eval rollout missing eval_step"
            rollout.eval_step = eval_step
        await self.out_q.put(rollout)
        if group is not None:
            group.emitted += 1
            if group.emitted >= group.target_rollouts:
                self.groups.pop(meta.group_id, None)

    async def drop_group(self, group_id: uuid.UUID) -> int:
        """Cancel remaining in-flight tasks for this group and emit a
        ``Cancelled`` marker for every rollout it still owes the sink
        (both in-flight and not-yet-scheduled). Returns the count for
        off-policy metrics."""
        group = self.groups.pop(group_id, None)
        task_idx = group.task_idx if group is not None else -1

        # Sync claim phase: pop matching tasks from ``self.inflight`` and
        # release their permits in one non-yielding sweep. After this loop
        # the dropped tasks are no longer reachable from ``self.inflight``,
        # so ``handle_completed_rollout``'s existing None-guard makes the
        # subsequent async emit phase race-free.
        claimed: list[tuple[asyncio.Task, InflightRollout]] = []
        for task, meta in list(self.inflight.items()):
            if meta.group_id != group_id:
                continue
            del self.inflight[task]
            self.release(meta.rollout_count)
            claimed.append((task, meta))

        tasks_to_cancel = [task for task, _ in claimed]
        inflight_cancelled = sum(meta.rollout_count for _, meta in claimed)
        last_meta: InflightRollout | None = claimed[-1][1] if claimed else None
        for _, meta in claimed:
            for _ in range(meta.rollout_count):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_rollout(meta, group, trace)

        # For non-group-scoring envs, the group may have rollouts that
        # were never dispatched (``rollouts_to_schedule > 0``). Emit
        # markers for those too so the sink hits ``target_rollouts``
        #
        # ``last_meta`` can be ``None`` if the only inflight task for this
        # group completed naturally between ``on_version_pending``'s snapshot
        # and us reaching it — synthesize a stand-in from the group state
        unscheduled_cancelled = 0
        if group is not None and group.rollouts_to_schedule > 0:
            fallback_meta = last_meta or InflightRollout(
                kind=group.kind,
                env_name=group.env_name,
                group_id=group_id,
                policy_version=group.policy_version_at_start,
                rollout_count=1,
                eval_step=group.eval_step,
                uses_mutable_policy=group.uses_mutable_policy,
            )
            unscheduled_cancelled = group.rollouts_to_schedule
            for _ in range(unscheduled_cancelled):
                trace = Rollout(
                    task=vf.TraceTask(type="Task", data=vf.TaskData(idx=task_idx, prompt=None)),
                    errors=[vf.Error(type="Cancelled", message="Off-policy cancel")],
                    stop_condition="error",
                )
                await self.emit_rollout(fallback_meta, group, trace)

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
                    uses_mutable_policy=group.uses_mutable_policy,
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
