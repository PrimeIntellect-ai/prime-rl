import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.types import is_cancelled
from tests.unit.orchestrator.fakes import FakeEnv, FakeEnvs, RecordingHooks, make_blank, make_episode


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
    assert sink.status() == "+1 buffered"
    await sink.ingest(make_episode(group_id="g"))
    ((group,),) = hooks["on_group"]
    assert group.admitted and group.id == "g" and group.step == 1
    assert len(group.episodes) == 2
    assert set(group.samples) == {ep.traces[0].id for ep in group.episodes}
    assert env.algorithm.finalized_episodes == 2
    assert env.algorithm.finalized_groups == 1
    assert sink.status() is None


@pytest.mark.asyncio
async def test_blank_episodes_complete_the_group_budget():
    sink, hooks, env = make_sink(group_size=3)
    await sink.ingest(make_episode(group_id="g"))
    await sink.ingest(make_blank(group_id="g", error=vf.Error(type="Boom", message="boom")))
    assert hooks["on_group"] == []
    await sink.ingest(make_blank(group_id="g", error=vf.Error(type="Cancelled", message="overload")))
    ((group,),) = hooks["on_group"]
    assert len(group.episodes) == 3 and len(group.samples) == 1
    assert env.algorithm.finalized_episodes == 1  # blank episodes are never scored


@pytest.mark.asyncio
async def test_stale_cancellation_voids_the_group_and_skips_scoring_and_curriculum():
    admit = RecordingHooks()
    sink, hooks, env = make_sink(admit=admit.record("admit", result=True))
    await sink.ingest(make_episode(group_id="g"))
    await sink.ingest(make_blank(group_id="g"))
    ((group,),) = hooks["on_group"]
    assert not group.admitted and group.samples == {}
    assert all(is_cancelled(episode) for episode in group.episodes)
    assert env.algorithm.finalized_groups == 0
    assert admit["admit"] == []


@pytest.mark.asyncio
async def test_rejected_group_carries_no_payload():
    admit = RecordingHooks()
    sink, hooks, env = make_sink(admit=admit.record("admit", result=False))
    await sink.ingest(make_episode(group_id="g"))
    await sink.ingest(make_episode(group_id="g"))
    ((group,),) = hooks["on_group"]
    assert not group.admitted and group.samples == {}
    assert env.algorithm.finalized_groups == 1


@pytest.mark.asyncio
async def test_errored_traces_do_not_compile():
    sink, hooks, _ = make_sink()
    await sink.ingest(make_episode(group_id="g", ok=False))
    await sink.ingest(make_episode(group_id="g", ok=False))
    ((group,),) = hooks["on_group"]
    assert group.samples == {} and not group.admitted
