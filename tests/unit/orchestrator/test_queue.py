import pytest

from prime_rl.orchestrator.metrics import Episodes
from prime_rl.orchestrator.queue import Queue
from prime_rl.orchestrator.types import Batch, cancel_reason
from tests.unit.orchestrator.fakes import RecordingHooks, make_group


def make_queue(step: int = 1, *, batch_size: int = 4, max_off_policy_steps: int = 8):
    hooks = RecordingHooks()
    state = {"step": step}
    queue = Queue(batch_size=batch_size, max_off_policy_steps=max_off_policy_steps)
    queue.bind(step=lambda: state["step"], on_batch=hooks.record_async("on_batch"))
    return queue, hooks, state


@pytest.mark.asyncio
async def test_batch_cuts_at_the_trace_target():
    queue, hooks, _ = make_queue(batch_size=4)
    await queue.put(make_group(2))
    assert hooks["on_batch"] == []
    assert queue.gauges()["queue/size"] == 2
    assert queue.status() == "Train batch 2/4 (50.0%)"
    await queue.put(make_group(2))
    ((batch,),) = hooks["on_batch"]
    assert isinstance(batch, Batch) and batch.step == 1
    assert len(batch.samples) == 4 and len(batch.groups) == 2
    assert queue.size == 0


@pytest.mark.asyncio
async def test_a_group_straddling_the_cut_is_split_and_its_tail_stays_queued():
    queue, hooks, _ = make_queue(batch_size=3)
    await queue.put(make_group(2))
    await queue.put(make_group(2))
    ((batch,),) = hooks["on_batch"]
    assert len(batch.samples) == 3
    assert [len(group.episodes) for group in batch.groups] == [2, 1]
    assert queue.size == 1
    (tail,) = queue.ready
    assert tail.id == batch.groups[1].id and len(tail.episodes) == 1


@pytest.mark.asyncio
async def test_stale_groups_are_voided_when_the_step_advances():
    queue, hooks, state = make_queue(batch_size=4, max_off_policy_steps=1)
    await queue.put(make_group(2, policy=(0, 0)))
    # step 4 trains v3; a v0 group would ship at staleness 3 > 1
    state["step"] = 4
    await queue.put(make_group(2, policy=(3, 3)))
    assert hooks["on_batch"] == []
    assert queue.gauges()["queue/dropped_stale"] == 2
    assert queue.gauges()["queue/size"] == 2
    await queue.put(make_group(2, policy=(3, 3)))
    ((batch,),) = hooks["on_batch"]
    episodes = Episodes(batch.groups)
    assert len(episodes) == 6 and len(episodes.sampled) == 4
    assert sum(episodes.cancelled.values) == 2
    assert all(cancel_reason(episode) == "stale" for episode in episodes if episode not in episodes.sampled)


@pytest.mark.asyncio
async def test_groups_without_samples_stay_in_the_window():
    queue, hooks, _ = make_queue(batch_size=2)
    await queue.put(make_group(2, samples=False, admitted=False))
    await queue.put(make_group(2))
    ((batch,),) = hooks["on_batch"]
    episodes = Episodes(batch.groups)
    assert len(episodes) == 4 and len(episodes.sampled) == 2 and len(episodes.admitted) == 2


@pytest.mark.asyncio
async def test_staleness_gauges_read_the_queued_traces():
    queue, _, state = make_queue(batch_size=10, max_off_policy_steps=8)
    await queue.put(make_group(2, policy=(0, 0)))
    state["step"] = 3
    await queue.put(make_group(2, policy=(2, 2)))
    gauges = queue.gauges()
    assert gauges["queue/staleness/max"] == 2
    assert gauges["queue/staleness/mean"] == 1.0
    assert gauges["queue/fill"] == 0.4


@pytest.mark.asyncio
async def test_status_breaks_down_by_env_when_several_are_queued():
    queue, _, _ = make_queue(batch_size=10)
    await queue.put(make_group(1, env_name="a"))
    await queue.put(make_group(2, env_name="b"))
    assert queue.status() == "Train batch 3/10 (30.0%) (a=1, b=2)"
