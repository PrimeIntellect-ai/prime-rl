import asyncio
import uuid
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import GroupState, InflightRollout, Policy


def _pool(model: str, request: str, admin: str):
    return SimpleNamespace(
        model_name=model,
        train_clients=[SimpleNamespace(base_url=request, headers={})],
        admin_clients=[SimpleNamespace(base_url=admin)],
    )


class _EnvCollection:
    def __init__(self, pool, *, run_rollout=None) -> None:
        self.env = SimpleNamespace(
            sampler=SimpleNamespace(samples_from_live_policy=False, pool=pool),
            requires_group_scoring=False,
            config=SimpleNamespace(group_size=1),
            run_rollout=run_rollout,
        )

    def get(self, _name: str):
        return self.env


def _dispatcher(*, enforce_barrier: bool, max_off_policy_steps: int = 0, run_rollout=None):
    policy_pool = _pool("policy", "http://policy/v1", "http://policy-worker:8081")
    distinct_pool = _pool("frozen", "http://frozen/v1", "http://frozen-worker:8081")
    train_envs = _EnvCollection(distinct_pool, run_rollout=run_rollout)
    dispatcher = RolloutDispatcher(
        train_envs=train_envs,
        eval_envs=None,
        train_source=object(),
        eval_source=None,
        policy_pool=policy_pool,
        policy=Policy(version=4, model_name="policy"),
        max_inflight_rollouts=1,
        tasks_per_minute=None,
        max_off_policy_steps=max_off_policy_steps,
        enforce_policy_update_barrier=enforce_barrier,
    )
    return dispatcher


@pytest.mark.asyncio
async def test_group_mutability_is_monotonic_for_cache_salt_after_pool_identity_churn():
    called: dict[str, object] = {}

    async def run_rollout(**kwargs):
        called.update(kwargs)
        await asyncio.Future()

    dispatcher = _dispatcher(enforce_barrier=True, run_rollout=run_rollout)
    group_id = uuid.uuid4()
    formerly_aliased_client = SimpleNamespace(base_url="http://policy/v1", headers={})
    group = GroupState(
        kind="train",
        env_name="train",
        task_idx=0,
        rollouts_to_schedule=1,
        target_rollouts=1,
        pinned_client=formerly_aliased_client,
        policy_version_at_start=4,
        uses_mutable_policy=False,
    )
    dispatcher.groups[group_id] = group
    epoch = await dispatcher.policy_gate.scheduling_epoch()
    assert epoch is not None

    assert await dispatcher.schedule_group_rollout(group_id, group, epoch=epoch)
    await asyncio.sleep(0)

    assert called["cache_salt"] == "4"
    assert next(iter(dispatcher.inflight.values())).uses_mutable_policy
    await dispatcher.cancel_inflight_rollouts()


@pytest.mark.asyncio
async def test_group_mutability_is_monotonic_for_non_dynamo_off_policy_aging():
    dispatcher = _dispatcher(enforce_barrier=False, max_off_policy_steps=0)
    group_id = uuid.uuid4()

    async def request() -> None:
        await asyncio.Future()

    task = asyncio.create_task(request())
    await asyncio.sleep(0)
    dispatcher.groups[group_id] = GroupState(
        kind="train",
        env_name="train",
        task_idx=0,
        rollouts_to_schedule=0,
        target_rollouts=1,
        policy_version_at_start=4,
        uses_mutable_policy=True,
    )
    dispatcher.inflight[task] = InflightRollout(
        kind="train",
        env_name="train",
        group_id=group_id,
        policy_version=4,
        rollout_count=1,
        uses_mutable_policy=True,
    )
    dispatcher.inflight_permits = 1

    await dispatcher.on_version_pending(5)

    assert task.cancelled()
    assert not dispatcher.inflight


@pytest.mark.asyncio
async def test_policy_update_does_not_wait_for_slow_client_selection():
    selection_started = asyncio.Event()
    release_selection = asyncio.Event()

    class _BlockingPool:
        model_name = "policy"
        train_clients = [SimpleNamespace(base_url="http://policy/v1", headers={})]
        admin_clients = [SimpleNamespace(base_url="http://policy-worker:8081")]

        async def select_train_client(self, _load):
            selection_started.set()
            await release_selection.wait()
            return self.train_clients[0]

    dispatcher = _dispatcher(enforce_barrier=True, run_rollout=None)
    dispatcher.train_envs.env.sampler.pool = _BlockingPool()
    group_id = uuid.uuid4()
    group = GroupState(
        kind="train",
        env_name="train",
        task_idx=0,
        rollouts_to_schedule=1,
        target_rollouts=1,
        policy_version_at_start=4,
        uses_mutable_policy=True,
    )
    dispatcher.groups[group_id] = group
    epoch = await dispatcher.policy_gate.scheduling_epoch()
    assert epoch is not None

    scheduling = asyncio.create_task(dispatcher.schedule_group_rollout(group_id, group, epoch=epoch))
    await selection_started.wait()
    await asyncio.wait_for(dispatcher.on_version_pending(5), timeout=0.5)

    assert dispatcher.policy_update_pending
    assert group_id not in dispatcher.groups

    release_selection.set()
    assert not await scheduling
    assert not dispatcher.inflight
    await dispatcher.on_new_version(5)


@pytest.mark.asyncio
async def test_selected_mutable_endpoint_survives_pool_churn_during_rate_limit_wait():
    rate_limit_started = asyncio.Event()
    release_rate_limit = asyncio.Event()
    selected_client = SimpleNamespace(base_url="http://policy/v1", headers={})

    class _ChurningPool:
        model_name = "frozen"
        train_clients = [selected_client]
        admin_clients = [SimpleNamespace(base_url="http://frozen-worker:8081")]

        async def select_train_client(self, _load):
            # The elastic snapshot loses the selected endpoint before the
            # dispatcher reaches its next slow wait.
            self.train_clients = [SimpleNamespace(base_url="http://frozen/v1", headers={})]
            return selected_client

    dispatcher = _dispatcher(enforce_barrier=True, run_rollout=None)
    dispatcher.train_envs.env.sampler.pool = _ChurningPool()

    async def wait_for_rate_limit(_permits: int) -> None:
        rate_limit_started.set()
        await release_rate_limit.wait()

    dispatcher._wait_for_rate_limit = wait_for_rate_limit  # type: ignore[method-assign]
    group_id = uuid.uuid4()
    group = GroupState(
        kind="train",
        env_name="train",
        task_idx=0,
        rollouts_to_schedule=1,
        target_rollouts=1,
        policy_version_at_start=4,
        uses_mutable_policy=False,
    )
    dispatcher.groups[group_id] = group
    epoch = await dispatcher.policy_gate.scheduling_epoch()
    assert epoch is not None

    scheduling = asyncio.create_task(dispatcher.schedule_group_rollout(group_id, group, epoch=epoch))
    await rate_limit_started.wait()

    assert group.uses_mutable_policy
    await asyncio.wait_for(dispatcher.on_version_pending(5), timeout=0.5)
    assert group_id not in dispatcher.groups

    release_rate_limit.set()
    assert not await scheduling
    await dispatcher.on_new_version(5)
