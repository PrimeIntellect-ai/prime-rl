import pytest

from prime_rl.orchestrator.train_sink import TrainSink
from tests.unit.orchestrator.fakes import (
    FakeEnv,
    FakeEnvs,
    RecordingHooks,
    make_cancellation,
    make_episode,
    make_failure,
)


def make_sink(*, group_size=2, admit=None):
    env = FakeEnv("env", group_size=group_size)
    sink = TrainSink(FakeEnvs(env))
    hooks = RecordingHooks()
    sink.bind(on_group=hooks.record_async("on_group"), admit=admit)
    return sink, hooks, env


@pytest.mark.asyncio
async def test_group_finalizes_once_every_episode_arrived():
    sink, hooks, env = make_sink()
    await sink.ingest(make_episode(group_id="g"))
    assert hooks["on_group"] == []
    assert sink.buffered_count() == 1
    assert sink.status() == "+1 buffered"
    await sink.ingest(make_episode(group_id="g"))
    ((group,),) = hooks["on_group"]
    assert group.admitted
    assert len(group.episodes) == 2
    assert set(group.samples) == {ep.traces[0].id for ep in group.episodes}
    assert env.algorithm.finalized_episodes == 2
    assert env.algorithm.finalized_groups == 1
    assert sink.buffered_count() == 0


@pytest.mark.asyncio
async def test_failures_and_cancellations_complete_the_group_budget():
    sink, hooks, _ = make_sink(group_size=3)
    await sink.ingest(make_episode(group_id="g"))
    await sink.ingest(make_failure(group_id="g"))
    assert hooks["on_group"] == []
    await sink.ingest(make_cancellation(group_id="g", count=1, reason="overload"))
    ((group,),) = hooks["on_group"]
    assert len(group.episodes) == 1 and len(group.failures) == 1
    assert group.cancellation is not None and not group.stale
    assert group.owed == 3


@pytest.mark.asyncio
async def test_stale_cancellation_skips_scoring_and_curriculum():
    admit = RecordingHooks()
    sink, hooks, env = make_sink(admit=admit.record("admit", result=True))
    await sink.ingest(make_episode(group_id="g"))
    await sink.ingest(make_cancellation(group_id="g", count=1, reason="stale"))
    ((group,),) = hooks["on_group"]
    assert group.stale and not group.admitted and group.samples == {}
    assert env.algorithm.finalized_groups == 0
    assert admit["admit"] == []


@pytest.mark.asyncio
async def test_rejected_group_carries_no_payload():
    admit = RecordingHooks()
    sink, hooks, env = make_sink(admit=admit.record("admit", result=False))
    await sink.ingest(make_episode(group_id="g"))
    await sink.ingest(make_episode(group_id="g"))
    ((group,),) = hooks["on_group"]
    assert not group.admitted and group.samples == {} and len(group.survivors) == 2
    assert env.algorithm.finalized_groups == 1


@pytest.mark.asyncio
async def test_errored_traces_are_not_survivors():
    sink, hooks, _ = make_sink()
    await sink.ingest(make_episode(group_id="g", ok=False))
    await sink.ingest(make_episode(group_id="g", ok=False))
    ((group,),) = hooks["on_group"]
    assert group.survivors == [] and group.samples == {}
