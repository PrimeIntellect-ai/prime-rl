import asyncio
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.dispatcher import DispatcherMode, RolloutDispatcher
from prime_rl.orchestrator.policy_gate import MutablePolicyGate, PolicyRequestRejected
from prime_rl.orchestrator.types import GroupState, InflightRollout, Policy, Rollout
from prime_rl.orchestrator.watcher import WeightWatcher
from prime_rl.utils.pathing import get_broadcast_dir, get_step_path


class _TrainEnvs:
    def __init__(self, *, live: bool, pool: object) -> None:
        self._live = live
        self._pool = pool

    def get(self, _name: str):
        return SimpleNamespace(sampler=SimpleNamespace(samples_from_live_policy=self._live, pool=self._pool))


def _dispatcher(
    *,
    max_inflight: int = 1,
    live_train: bool = True,
    frozen_uses_policy_pool: bool = False,
    frozen_aliases_policy_pool: bool = False,
    enforce_policy_update_barrier: bool = True,
    max_off_policy_steps: int = 8,
    policy_gate: MutablePolicyGate | None = None,
) -> RolloutDispatcher:
    def pool(model_name: str, request_url: str, admin_url: str):
        return SimpleNamespace(
            model_name=model_name,
            train_clients=[SimpleNamespace(base_url=request_url, headers={})],
            admin_clients=[SimpleNamespace(base_url=admin_url)],
        )

    policy_pool = pool("policy", "http://policy/v1", "http://policy-worker:8081")
    if live_train or frozen_uses_policy_pool:
        train_pool = policy_pool
    elif frozen_aliases_policy_pool:
        # A separately-constructed pool can still address the exact mutable
        # serving resource. Object identity is not a topology identity.
        train_pool = pool("policy", "http://policy/v1", "http://policy-worker:8081")
    else:
        train_pool = pool("frozen", "http://frozen/v1", "http://frozen-worker:8081")
    return RolloutDispatcher(
        train_envs=_TrainEnvs(live=live_train, pool=train_pool),
        eval_envs=object(),
        train_source=object(),
        eval_source=object(),
        policy_pool=policy_pool,
        policy=Policy(version=0, model_name="policy"),
        max_inflight_rollouts=max_inflight,
        tasks_per_minute=None,
        max_off_policy_steps=max_off_policy_steps,
        enforce_policy_update_barrier=enforce_policy_update_barrier,
        policy_gate=policy_gate,
    )


@pytest.mark.asyncio
async def test_policy_barrier_invalidates_slow_scheduling_before_short_commit():
    dispatcher = _dispatcher()
    dispatcher.mode = DispatcherMode.PREFER_EVAL
    scheduling_started = asyncio.Event()
    allow_schedule_to_commit = asyncio.Event()
    request_started = asyncio.Event()

    async def active_request() -> None:
        request_started.set()
        await asyncio.Future()

    async def schedule_one(_kind: str, *, epoch) -> bool:
        scheduling_started.set()
        await allow_schedule_to_commit.wait()
        async with dispatcher.policy_gate.scheduling_commit(epoch) as admitted:
            if not admitted:
                return False
            group_id = uuid.uuid4()
            request = asyncio.create_task(active_request())
            dispatcher.groups[group_id] = GroupState(
                kind="eval",
                env_name="eval",
                task_idx=0,
                rollouts_to_schedule=0,
                target_rollouts=1,
                eval_step=1,
                policy_version_at_start=0,
            )
            dispatcher.inflight[request] = InflightRollout(
                kind="eval",
                env_name="eval",
                group_id=group_id,
                policy_version=0,
                rollout_count=1,
                eval_step=1,
            )
            dispatcher.inflight_permits += 1
            return True

    dispatcher.try_schedule = schedule_one  # type: ignore[method-assign]

    fill = asyncio.create_task(dispatcher.fill_inflight())
    await scheduling_started.wait()
    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await asyncio.sleep(0)

    await barrier
    assert dispatcher.policy_update_pending

    allow_schedule_to_commit.set()
    await fill
    assert not request_started.is_set()
    assert not dispatcher.inflight

    # The transition fence remains closed until the watcher reports either
    # success or failure.
    await dispatcher.fill_inflight()
    assert not dispatcher.inflight

    await dispatcher.on_new_version(1)
    assert not dispatcher.policy_update_pending


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("frozen_uses_policy_pool", "frozen_aliases_policy_pool", "cancelled_by_barrier"),
    [(False, False, False), (True, False, True), (False, True, True)],
)
async def test_policy_barrier_only_preserves_frozen_requests_on_a_distinct_pool(
    frozen_uses_policy_pool: bool,
    frozen_aliases_policy_pool: bool,
    cancelled_by_barrier: bool,
):
    dispatcher = _dispatcher(
        live_train=False,
        frozen_uses_policy_pool=frozen_uses_policy_pool,
        frozen_aliases_policy_pool=frozen_aliases_policy_pool,
    )
    group_id = uuid.uuid4()
    request_started = asyncio.Event()

    async def request() -> None:
        request_started.set()
        await asyncio.Future()

    task = asyncio.create_task(request())
    await request_started.wait()
    dispatcher.groups[group_id] = GroupState(
        kind="train",
        env_name="train",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        policy_version_at_start=0,
    )
    dispatcher.inflight[task] = InflightRollout(
        kind="train",
        env_name="train",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
    )
    dispatcher.inflight_permits = 1

    await dispatcher.on_version_pending(1)

    assert task.cancelled() is cancelled_by_barrier
    assert (task in dispatcher.inflight) is not cancelled_by_barrier

    await dispatcher.on_new_version(1)
    await dispatcher.cancel_inflight_rollouts()


@pytest.mark.asyncio
async def test_failed_policy_barrier_skips_engine_mutation_and_reopens_admission(tmp_path: Path):
    dispatcher = _dispatcher()
    group_id = uuid.uuid4()
    cleanup_attempted = asyncio.Event()

    async def request_with_failed_cleanup() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_attempted.set()
            raise RuntimeError("connector cleanup failed")

    request = asyncio.create_task(request_with_failed_cleanup())
    await asyncio.sleep(0)
    dispatcher.groups[group_id] = GroupState(
        kind="eval",
        env_name="eval",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        eval_step=1,
        policy_version_at_start=0,
    )
    dispatcher.inflight[request] = InflightRollout(
        kind="eval",
        env_name="eval",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
        eval_step=1,
    )
    dispatcher.inflight_permits = 1

    weight_path = get_step_path(get_broadcast_dir(tmp_path), 1)
    weight_path.mkdir(parents=True)
    (weight_path / "STABLE").touch()

    class _Inference:
        def __init__(self) -> None:
            self.update_calls = 0

        async def update_weights(self, *_args, **_kwargs) -> None:
            self.update_calls += 1

    inference = _Inference()
    policy = dispatcher.policy
    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=policy,
        inference=inference,
        observers=[dispatcher],
        lora_name=None,
    )

    with pytest.raises(RuntimeError, match="connector cleanup failed"):
        await watcher.apply_policy_update(1)

    assert cleanup_attempted.is_set()
    assert inference.update_calls == 0
    assert watcher.ckpt_step == 0
    assert policy.version == 0
    assert not dispatcher.policy_update_pending


@pytest.mark.asyncio
async def test_policy_barrier_accepts_request_error_that_was_already_settled():
    dispatcher = _dispatcher()
    group_id = uuid.uuid4()

    async def failed_request() -> None:
        raise RuntimeError("ordinary rollout failure")

    request = asyncio.create_task(failed_request())
    await asyncio.wait({request})
    dispatcher.groups[group_id] = GroupState(
        kind="eval",
        env_name="eval",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        eval_step=1,
        policy_version_at_start=0,
    )
    dispatcher.inflight[request] = InflightRollout(
        kind="eval",
        env_name="eval",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
        eval_step=1,
    )
    dispatcher.inflight_permits = 1

    await dispatcher.on_version_pending(1)

    assert dispatcher.policy_update_pending
    assert not dispatcher.inflight
    await dispatcher.on_new_version(1)


@pytest.mark.asyncio
async def test_policy_barrier_settles_cleanup_before_propagating_cancellation():
    dispatcher = _dispatcher()
    group_id = uuid.uuid4()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def request() -> None:
        try:
            await asyncio.Future()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()

    task = asyncio.create_task(request())
    await asyncio.sleep(0)
    dispatcher.groups[group_id] = GroupState(
        kind="eval",
        env_name="eval",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        eval_step=1,
        policy_version_at_start=0,
    )
    dispatcher.inflight[task] = InflightRollout(
        kind="eval",
        env_name="eval",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
        eval_step=1,
    )
    dispatcher.inflight_permits = 1

    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await cleanup_started.wait()
    barrier.cancel()
    await asyncio.sleep(0)
    barrier.cancel()
    await asyncio.sleep(0)

    assert not barrier.done()
    assert not cleanup_finished.is_set()

    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await barrier

    assert cleanup_finished.is_set()
    assert not dispatcher.policy_update_pending


@pytest.mark.asyncio
async def test_policy_barrier_shields_marker_enqueue_and_commits_accounting_after_put():
    dispatcher = _dispatcher()
    group_id = uuid.uuid4()
    request_started = asyncio.Event()
    marker_put_started = asyncio.Event()

    class _ObservedQueue(asyncio.Queue):
        async def put(self, item) -> None:
            marker_put_started.set()
            await super().put(item)

    dispatcher.out_q = _ObservedQueue(maxsize=1)
    dispatcher.out_q.put_nowait(object())

    async def request() -> None:
        request_started.set()
        await asyncio.Future()

    request_task = asyncio.create_task(request())
    await request_started.wait()
    group = GroupState(
        kind="eval",
        env_name="eval",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        eval_step=1,
        policy_version_at_start=0,
    )
    dispatcher.groups[group_id] = group
    dispatcher.inflight[request_task] = InflightRollout(
        kind="eval",
        env_name="eval",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
        eval_step=1,
    )
    dispatcher.inflight_permits = 1

    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await marker_put_started.wait()
    assert group.emitted == 0

    barrier.cancel()
    await asyncio.sleep(0)
    barrier.cancel()
    await asyncio.sleep(0)
    assert not barrier.done()
    assert group.emitted == 0

    dispatcher.out_q.get_nowait()
    with pytest.raises(asyncio.CancelledError):
        await barrier

    marker = dispatcher.out_q.get_nowait()
    assert marker.error.type == "Cancelled"
    assert group.emitted == 1
    assert not dispatcher.groups
    assert not dispatcher.inflight
    assert dispatcher.inflight_permits == 0

    assert not dispatcher.policy_update_pending


@pytest.mark.asyncio
async def test_policy_barrier_waits_for_completed_handler_before_computing_markers():
    dispatcher = _dispatcher(max_inflight=2)
    dispatcher.out_q = asyncio.Queue(maxsize=1)
    group_id = uuid.uuid4()

    async def completed_group() -> list[Rollout]:
        return [
            Rollout(
                task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt=None)),
                errors=[vf.Error(type="Existing", message="result")],
                stop_condition="error",
            )
            for _ in range(2)
        ]

    task = asyncio.create_task(completed_group())
    await task
    group = GroupState(
        kind="eval",
        env_name="eval",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=2,
        eval_step=1,
        policy_version_at_start=0,
        uses_mutable_policy=True,
    )
    dispatcher.groups[group_id] = group
    dispatcher.inflight[task] = InflightRollout(
        kind="eval",
        env_name="eval",
        group_id=group_id,
        policy_version=0,
        rollout_count=2,
        eval_step=1,
        uses_mutable_policy=True,
    )
    dispatcher.inflight_permits = 2

    handler = asyncio.create_task(dispatcher.handle_completed_rollout(task))
    while dispatcher.out_q.qsize() != 1:
        await asyncio.sleep(0)
    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await asyncio.sleep(0)

    assert not barrier.done()

    first = dispatcher.out_q.get_nowait()
    await handler
    await barrier
    second = dispatcher.out_q.get_nowait()

    assert [first.error.type, second.error.type] == ["Existing", "Existing"]
    assert group.emitted == 2
    await dispatcher.on_new_version(1)


@pytest.mark.asyncio
async def test_policy_barrier_marker_enqueue_aborts_promptly_on_dispatcher_stop():
    dispatcher = _dispatcher()
    marker_put_started = asyncio.Event()

    class _ObservedQueue(asyncio.Queue):
        async def put(self, item) -> None:
            marker_put_started.set()
            await super().put(item)

    dispatcher.out_q = _ObservedQueue(maxsize=1)
    dispatcher.out_q.put_nowait(object())
    group_id = uuid.uuid4()

    async def request() -> None:
        await asyncio.Future()

    task = asyncio.create_task(request())
    await asyncio.sleep(0)
    dispatcher.groups[group_id] = GroupState(
        kind="eval",
        env_name="eval",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        eval_step=1,
        policy_version_at_start=0,
        uses_mutable_policy=True,
    )
    dispatcher.inflight[task] = InflightRollout(
        kind="eval",
        env_name="eval",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
        eval_step=1,
        uses_mutable_policy=True,
    )
    dispatcher.inflight_permits = 1

    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await marker_put_started.wait()
    await dispatcher.stop()

    with pytest.raises(RuntimeError, match="stopped while emitting policy barrier"):
        await asyncio.wait_for(barrier, timeout=0.5)
    assert not dispatcher.policy_update_pending


@pytest.mark.asyncio
async def test_policy_barrier_drains_active_policy_calls_and_rejects_late_or_stale_calls():
    policy = Policy(version=0, model_name="policy")
    policy_gate = MutablePolicyGate(policy, enabled=True)
    dispatcher = _dispatcher(policy_gate=policy_gate)
    # The helper constructs its own Policy, so make both components share the
    # exact version object as production does.
    dispatcher.policy = policy

    call_started = asyncio.Event()
    release_call = asyncio.Event()

    async def active_policy_call() -> None:
        async with policy_gate.request(expected_version=0):
            call_started.set()
            await release_call.wait()

    score = asyncio.create_task(active_policy_call())
    await call_started.wait()
    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await asyncio.sleep(0)
    assert dispatcher.policy_update_pending
    assert not barrier.done()

    with pytest.raises(PolicyRequestRejected, match="update is pending"):
        async with policy_gate.request(expected_version=0):
            pass

    release_call.set()
    await score
    await barrier

    policy.version = 1
    await dispatcher.on_new_version(1)
    with pytest.raises(PolicyRequestRejected, match="expected policy version 0.*current version is 1"):
        async with policy_gate.request(expected_version=0):
            pass


@pytest.mark.asyncio
async def test_non_dynamo_dispatcher_retains_configured_off_policy_window():
    dispatcher = _dispatcher(enforce_policy_update_barrier=False, max_off_policy_steps=1)
    group_id = uuid.uuid4()
    request_started = asyncio.Event()

    async def request() -> None:
        request_started.set()
        await asyncio.Future()

    task = asyncio.create_task(request())
    await request_started.wait()
    dispatcher.groups[group_id] = GroupState(
        kind="train",
        env_name="train",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        policy_version_at_start=0,
    )
    dispatcher.inflight[task] = InflightRollout(
        kind="train",
        env_name="train",
        group_id=group_id,
        policy_version=0,
        rollout_count=1,
    )
    dispatcher.inflight_permits = 1

    await dispatcher.on_version_pending(1)
    assert dispatcher.inflight[task].off_policy_steps == 1
    assert not dispatcher.policy_update_pending

    await dispatcher.on_version_pending(2)
    assert task.cancelled()
    assert not dispatcher.inflight


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["collective update failed", "resume_generation failed"])
async def test_indeterminate_engine_failure_keeps_admission_fail_closed_without_advancing_version(
    tmp_path: Path,
    failure: str,
):
    dispatcher = _dispatcher()
    weight_path = get_step_path(get_broadcast_dir(tmp_path), 1)
    weight_path.mkdir(parents=True)
    (weight_path / "STABLE").touch()

    class _Inference:
        async def update_weights(self, *_args, **_kwargs) -> None:
            raise RuntimeError(failure)

    policy = dispatcher.policy
    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=policy,
        inference=_Inference(),
        observers=[dispatcher],
        lora_name=None,
    )

    with pytest.raises(RuntimeError, match=failure) as exc:
        await watcher.apply_policy_update(1)

    assert watcher.ckpt_step == 0
    assert policy.version == 0
    assert dispatcher.policy_update_pending
    assert exc.value.__notes__ == [
        "Policy update 1 may have mutated inference workers; mutable-policy admission remains fail-closed"
    ]

    second_weight_path = get_step_path(get_broadcast_dir(tmp_path), 2)
    second_weight_path.mkdir(parents=True)
    (second_weight_path / "STABLE").touch()
    with pytest.raises(RuntimeError, match="already pending"):
        await watcher.apply_policy_update(2)
    assert dispatcher.policy_update_pending


@pytest.mark.asyncio
async def test_success_updates_lead_gate_before_reopening_transition_fence(tmp_path: Path):
    events: list[str] = []
    weight_path = get_step_path(get_broadcast_dir(tmp_path), 1)
    weight_path.mkdir(parents=True)
    (weight_path / "STABLE").touch()

    class _Observer:
        def __init__(self, name: str) -> None:
            self.name = name

        async def on_version_pending(self, _step: int) -> None:
            events.append(f"{self.name}:pending")

        async def on_new_version(self, _step: int) -> None:
            events.append(f"{self.name}:new")

        async def on_version_update_failed(self, _step: int, _error: BaseException) -> None:
            events.append(f"{self.name}:failed")

    class _Inference:
        async def update_weights(self, *_args, **_kwargs) -> None:
            events.append("engine:update")

    policy = Policy(version=0, model_name="policy")
    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=policy,
        inference=_Inference(),
        observers=[_Observer("fence"), _Observer("lead-gate")],
        lora_name=None,
    )

    await watcher.apply_policy_update(1)

    assert events == [
        "fence:pending",
        "lead-gate:pending",
        "engine:update",
        "lead-gate:new",
        "fence:new",
    ]
    assert watcher.ckpt_step == 1
    assert policy.version == 1


@pytest.mark.asyncio
async def test_success_callback_cancellation_propagates_and_keeps_transition_fence_closed(tmp_path: Path):
    dispatcher = _dispatcher()
    callback_started = asyncio.Event()
    weight_path = get_step_path(get_broadcast_dir(tmp_path), 1)
    weight_path.mkdir(parents=True)
    (weight_path / "STABLE").touch()

    class _BlockingLeadGate:
        async def on_version_pending(self, _step: int) -> None:
            return

        async def on_new_version(self, _step: int) -> None:
            callback_started.set()
            await asyncio.Future()

        async def on_version_update_failed(self, _step: int, _error: BaseException) -> None:
            return

    class _Inference:
        async def update_weights(self, *_args, **_kwargs) -> None:
            return

    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=dispatcher.policy,
        inference=_Inference(),
        observers=[dispatcher, _BlockingLeadGate()],
        lora_name=None,
    )

    update = asyncio.create_task(watcher.apply_policy_update(1))
    await callback_started.wait()
    update.cancel()
    with pytest.raises(asyncio.CancelledError):
        await update

    assert watcher.ckpt_step == 1
    assert dispatcher.policy.version == 1
    assert dispatcher.policy_update_pending


@pytest.mark.asyncio
async def test_success_callback_error_propagates_and_keeps_transition_fence_closed(tmp_path: Path):
    dispatcher = _dispatcher()
    weight_path = get_step_path(get_broadcast_dir(tmp_path), 1)
    weight_path.mkdir(parents=True)
    (weight_path / "STABLE").touch()

    class _BrokenLeadGate:
        async def on_version_pending(self, _step: int) -> None:
            return

        async def on_new_version(self, _step: int) -> None:
            raise RuntimeError("lead gate callback failed")

        async def on_version_update_failed(self, _step: int, _error: BaseException) -> None:
            return

    class _Inference:
        async def update_weights(self, *_args, **_kwargs) -> None:
            return

    watcher = WeightWatcher(
        SimpleNamespace(output_dir=tmp_path),
        policy=dispatcher.policy,
        inference=_Inference(),
        observers=[dispatcher, _BrokenLeadGate()],
        lora_name=None,
    )

    with pytest.raises(RuntimeError, match="lead gate callback failed"):
        await watcher.apply_policy_update(1)

    assert watcher.ckpt_step == 1
    assert dispatcher.policy.version == 1
    assert dispatcher.policy_update_pending
