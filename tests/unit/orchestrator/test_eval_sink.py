from prime_rl.orchestrator.eval_sink import EvalSink
from tests.unit.orchestrator.fakes import FakeEnv, FakeEnvs, make_cancellation, make_episode, make_failure, make_task


def make_sink(examples=2, group_size=2):
    env = FakeEnv("env", group_size=group_size, examples=[make_task(i) for i in range(examples)])
    return EvalSink(eval_envs=FakeEnvs(env))


def test_epoch_completes_when_every_rollout_is_accounted_for():
    sink = make_sink(examples=2, group_size=2)
    assert sink.ingest(make_episode(kind="eval", step=3, group_id="a")) is None
    assert sink.ingest(make_failure(kind="eval", step=3, group_id="a")) is None
    assert sink.batch_progress() == [("env", 3, 2, 4)]
    assert sink.ingest(make_cancellation(kind="eval", step=3, group_id="b", count=2, reason="superseded")) is not None


def test_batches_carry_their_cohort_and_accounting():
    sink = make_sink(examples=1, group_size=2)
    sink.ingest(make_episode(kind="eval", step=1, group_id="a"))
    batch = sink.ingest(make_failure(kind="eval", step=1, group_id="a"))
    assert batch is not None
    assert batch.env_name == "env" and batch.step == 1
    assert len(batch.episodes) == 1 and len(batch.failures) == 1 and batch.cancelled == 0
    assert sink.batch_progress() == []


def test_epochs_of_different_steps_do_not_mix():
    sink = make_sink(examples=1, group_size=1)
    assert sink.ingest(make_episode(kind="eval", step=1)) is not None
    assert sink.ingest(make_episode(kind="eval", step=2)) is not None
