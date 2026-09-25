import pytest

from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.evaluator import Evaluator, EvaluatorConfig
from tests.unit.orchestrator.fakes import FakeEnv, FakeEnvs, RecordingHooks, RecordingMonitors, make_episode, make_task


def make_evaluator(*, intervals=None, examples=2, group_size=1, **config):
    env = FakeEnv("env", group_size=group_size, examples=[make_task(i) for i in range(examples)])
    envs = FakeEnvs(env)
    source = EvalSource(envs, intervals=intervals)
    evaluator = Evaluator(EvaluatorConfig(**config), eval_source=source, eval_envs=envs)
    hooks, monitors = RecordingHooks(), RecordingMonitors()
    evaluator.bind(prefer_eval=hooks.record("prefer_eval"), monitors=monitors)
    return evaluator, source, hooks, monitors


@pytest.mark.asyncio
async def test_trigger_queues_the_epoch_and_prefers_eval():
    evaluator, source, hooks, monitors = make_evaluator()
    assert await evaluator.trigger(0) == ["env"]
    assert len(source) == 2
    assert monitors.plans == [("env", 0, 2)]
    assert hooks["prefer_eval"] == [("eval was triggered at step 0",)]
    assert evaluator.is_pending(0, ["env"])
    # a step fires once
    assert await evaluator.trigger(0) == []


@pytest.mark.asyncio
async def test_epoch_finalizes_with_metrics_once_every_rollout_lands():
    evaluator, _, _, monitors = make_evaluator(examples=2)
    await evaluator.trigger(0)
    await evaluator.ingest(make_episode(kind="eval", step=0, reward=1.0))
    assert evaluator.is_pending(0, ["env"])
    await evaluator.ingest(make_episode(kind="eval", step=0, reward=0.0))
    assert not evaluator.is_pending(0, ["env"])
    assert evaluator.changed.is_set()
    ((metrics, step),) = monitors.metrics
    assert step == 0 and metrics["eval/env/policy_version"] == 0.0
    assert metrics["eval/env/effective/agent/reward/mean"] == 0.5
    assert monitors.epochs == []  # not uploaded unless asked


@pytest.mark.asyncio
async def test_upload_epochs_hands_the_cohort_to_the_monitors():
    evaluator, _, _, monitors = make_evaluator(examples=1, upload_epochs=True)
    await evaluator.trigger(0)
    await evaluator.ingest(make_episode(kind="eval", step=0))
    assert [(env, step, len(eps)) for env, step, eps in monitors.epochs] == [("env", 0, 1)]


@pytest.mark.asyncio
async def test_intervals_and_final_step_govern_which_steps_fire():
    evaluator, _, _, _ = make_evaluator(intervals={"env": 5}, max_steps=7)
    assert await evaluator.trigger(0) == ["env"]  # the base policy always evaluates
    assert await evaluator.trigger(1) == []
    assert await evaluator.trigger(5) == ["env"]
    assert await evaluator.trigger(7) == ["env"]  # final step fires regardless of interval


@pytest.mark.asyncio
async def test_resume_step_is_skipped_unless_retriggered():
    evaluator, _, _, _ = make_evaluator(resume_step=5, retrigger_on_resume=False)
    assert await evaluator.trigger(5) == []
    evaluator, _, _, _ = make_evaluator(resume_step=5, retrigger_on_resume=True)
    assert await evaluator.trigger(5) == ["env"]
