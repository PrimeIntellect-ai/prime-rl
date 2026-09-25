import pytest

from prime_rl.orchestrator.queue import Queue, QueueConfig
from prime_rl.orchestrator.types import FinalizedGroup, TrainBatch
from tests.unit.orchestrator.fakes import RecordingHooks, make_cancellation, make_episode, make_failure, make_sample


def group(n: int, *, policy=(0, 0), env_name="env", tokens=3, samples=True) -> FinalizedGroup:
    gid = f"g{policy}{n}"
    episodes = [make_episode(env_name=env_name, group_id=gid, policy=policy, sampled_tokens=tokens) for _ in range(n)]
    payload = {ep.traces[0].id: [make_sample(tokens)] for ep in episodes} if samples else {}
    return FinalizedGroup(
        env_name=env_name,
        episodes=episodes,
        samples=payload,
        survivors=[ep.traces[0] for ep in episodes],
        failures=[],
        cancellation=None,
        admitted=True,
    )


def make_queue(step: int = 1, **config):
    hooks = RecordingHooks()
    state = {"step": step}
    queue = Queue(QueueConfig(**config))
    queue.bind(step=lambda: state["step"], on_batch=hooks.record_async("on_batch"))
    return queue, hooks, state


@pytest.mark.asyncio
async def test_batch_cuts_at_the_trace_target_and_resets_the_window():
    queue, hooks, _ = make_queue(batch_size=4)
    await queue.put(group(2))
    assert hooks["on_batch"] == []
    assert queue.gauges()["queue/size"] == 2
    assert queue.status() == "Train batch 2/4 (50.0%)"
    await queue.put(group(2))
    ((batch,),) = hooks["on_batch"]
    assert isinstance(batch, TrainBatch)
    assert len(batch.samples) == 4
    assert len(batch.cohort) == 4
    assert len(batch.episodes) == 4
    assert queue.size == 0


@pytest.mark.asyncio
async def test_token_batching_cuts_once_the_token_target_is_met():
    queue, hooks, _ = make_queue(token_batch_size=10)
    await queue.put(group(2, tokens=3))
    assert hooks["on_batch"] == []
    await queue.put(group(2, tokens=3))
    ((batch,),) = hooks["on_batch"]
    assert sum(len(sample.token_ids) for sample in batch.samples) == 12
    assert queue.pending_tokens == 0


@pytest.mark.asyncio
async def test_stale_traces_are_swept_when_the_step_advances():
    queue, hooks, state = make_queue(batch_size=4, max_off_policy_steps=1)
    await queue.put(group(2, policy=(0, 0)))
    # step 4 trains v3; a v0 trace would ship at staleness 3 > 1
    state["step"] = 4
    await queue.put(group(2, policy=(3, 3)))
    assert hooks["on_batch"] == []
    assert queue.gauges()["queue/dropped_stale"] == 2
    assert queue.gauges()["queue/size"] == 2
    await queue.put(group(2, policy=(3, 3)))
    ((batch,),) = hooks["on_batch"]
    assert batch.stale_drops == 2
    assert len(batch.episodes.cancelled) == 2


@pytest.mark.asyncio
async def test_stale_cancellation_counts_the_whole_group_as_cancelled():
    queue, hooks, _ = make_queue(batch_size=2)
    g = group(1)
    g.cancellation = make_cancellation(group_id="x", count=1)
    g.samples = {}
    g.admitted = False
    await queue.put(g)
    await queue.put(group(2))
    ((batch,),) = hooks["on_batch"]
    assert batch.cancelled_attempts == 1
    assert batch.stale_attempts == 1
    assert g.episodes[0].id in batch.episodes.cancelled


@pytest.mark.asyncio
async def test_failures_and_rejected_groups_stay_in_the_window():
    queue, hooks, _ = make_queue(batch_size=2)
    rejected = group(2, samples=False)
    rejected.admitted = False
    rejected.failures = [make_failure(group_id="f")]
    await queue.put(rejected)
    await queue.put(group(2))
    ((batch,),) = hooks["on_batch"]
    assert len(batch.failures) == 1
    assert len(batch.episodes) == 4
    assert len(batch.cohort) == 2


@pytest.mark.asyncio
async def test_staleness_gauges_read_the_queued_traces():
    queue, _, state = make_queue(batch_size=10, max_off_policy_steps=8)
    await queue.put(group(2, policy=(0, 0)))
    state["step"] = 3
    await queue.put(group(2, policy=(2, 2)))
    gauges = queue.gauges()
    assert gauges["queue/staleness/max"] == 2
    assert gauges["queue/staleness/mean"] == 1.0
    assert gauges["queue/fill"] == 0.4


@pytest.mark.asyncio
async def test_status_breaks_down_by_env_when_several_are_queued():
    queue, _, _ = make_queue(batch_size=10)
    await queue.put(group(1, env_name="a"))
    await queue.put(group(2, env_name="b"))
    assert queue.status() == "Train batch 3/10 (30.0%) (a=1, b=2)"
