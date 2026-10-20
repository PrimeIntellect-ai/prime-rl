import asyncio
import uuid

from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.types import GroupCancellation, GroupState, InflightEpisode, Policy


def dispatcher() -> Dispatcher:
    return Dispatcher(
        train_envs=None,
        eval_envs=None,
        train_source=None,
        eval_source=None,
        policy_clients=None,
        policy=Policy(),
        progress=None,
        initial_max_inflight=4,
        max_inflight_ceiling=None,
        dispatch_per_minute=None,
        max_off_policy_steps=1,
        run_id="run",
        run_name=None,
    )


def test_drop_group_cancels_inflight_episodes_on_their_servers():
    asyncio.run(drop_group())


async def drop_group():
    d = dispatcher()
    group_id = uuid.uuid4()
    d.groups[group_id] = GroupState(
        kind="eval", env_name="env", task=None, step=3, episodes_to_schedule=2, target_episodes=4
    )
    metas = [
        InflightEpisode(kind="eval", env_name="env", group_id=group_id, task=None, policy_version=0, step=3)
        for _ in range(2)
    ]
    runs = [asyncio.create_task(asyncio.sleep(60)) for _ in metas]
    for run, meta in zip(runs, metas):
        d.inflight[run] = meta
    d.current_inflight = 2

    assert await d.drop_group(group_id, reason="superseded") == 4
    assert all(meta.cancel.is_set() for meta in metas)
    assert d.current_inflight == 0  # permits free now, the runs still return their episodes
    assert not any(run.done() for run in runs) and len(d.inflight) == 2
    marker = d.out_q.get_nowait()
    assert isinstance(marker, GroupCancellation)
    assert (marker.count, marker.reason) == (2, "superseded")  # only the never-dispatched episodes

    assert await d.drop_group(group_id, reason="superseded") == 0  # already cancelling
    assert d.current_inflight == 0 and d.out_q.empty()
    for run in runs:
        run.cancel()
