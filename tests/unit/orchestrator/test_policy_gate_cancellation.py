import asyncio

import pytest

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.types import Policy


def _dispatcher() -> RolloutDispatcher:
    pool = type(
        "Pool",
        (),
        {
            "model_name": "policy",
            "train_clients": [],
            "admin_clients": [],
        },
    )()
    return RolloutDispatcher(
        train_envs=object(),
        eval_envs=None,
        train_source=object(),
        eval_source=None,
        policy_pool=pool,
        policy=Policy(version=0, model_name="policy"),
        max_inflight_rollouts=1,
        tasks_per_minute=None,
        max_off_policy_steps=0,
        enforce_policy_update_barrier=True,
    )


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_partial_entry_rollback():
    dispatcher = _dispatcher()
    settle_started = asyncio.Event()
    release_settle = asyncio.Event()
    rollback_started = asyncio.Event()

    async def settle_policy_requests(*_args) -> None:
        settle_started.set()
        await release_settle.wait()

    dispatcher._settle_policy_requests = settle_policy_requests  # type: ignore[method-assign]
    finish_update = dispatcher.policy_gate.finish_update

    async def observed_finish_update(token) -> None:
        rollback_started.set()
        await finish_update(token)

    dispatcher.policy_gate.finish_update = observed_finish_update  # type: ignore[method-assign]
    barrier = asyncio.create_task(dispatcher.on_version_pending(1))
    await settle_started.wait()

    await dispatcher.policy_gate._admission_lock.acquire()
    barrier.cancel()
    release_settle.set()
    await rollback_started.wait()
    barrier.cancel()
    await asyncio.sleep(0)
    rollback_was_interrupted = barrier.done()
    dispatcher.policy_gate._admission_lock.release()

    assert not rollback_was_interrupted
    with pytest.raises(asyncio.CancelledError) as exc:
        await barrier
    assert not dispatcher.policy_update_pending
    assert exc.value.__notes__ == ["Policy transition rollback was cancelled again but settled before propagation"]
