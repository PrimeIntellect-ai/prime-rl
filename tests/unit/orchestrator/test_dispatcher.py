import asyncio

import pytest
import verifiers.v1 as vf

from prime_rl.orchestrator.dispatcher import Dispatcher, DispatcherMode
from prime_rl.orchestrator.eval_source import EvalSource
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation
from tests.unit.orchestrator.fakes import FakeClients, FakeEnv, FakeEnvs, FakeSource, RecordingMonitors, make_task


class Harness:
    def __init__(self, *, envs: FakeEnvs, max_inflight=4, limit=None, eval_envs=None, step=1, version=0):
        self.clients = FakeClients()
        for env in envs:
            env.generation_source.clients = self.clients
        self.train = []
        self.eval = []
        self.completed = []
        self.monitors = RecordingMonitors()
        self.state = {"step": step, "version": version}
        self.eval_source = EvalSource(eval_envs) if eval_envs is not None else None
        self.dispatcher = Dispatcher(
            dispatch_per_minute=None,
            train_envs=envs,
            eval_envs=eval_envs,
            train_source=FakeSource(envs, limit=limit),
            eval_source=self.eval_source,
            policy_clients=self.clients,
            initial_max_inflight=max_inflight,
            max_inflight_ceiling=max_inflight,
            max_off_policy_steps=1,
            run_id="run",
            run_name="test",
        )

        async def on_train(item):
            self.train.append(item)

        async def on_eval(item):
            self.eval.append(item)

        self.dispatcher.bind(
            step=lambda: self.state["step"],
            version=lambda: self.state["version"],
            on_train=on_train,
            on_eval=on_eval,
            on_episode_complete=lambda *args: self.completed.append(args),
            monitors=self.monitors,
        )

    async def run_until(self, predicate, timeout=5.0):
        task = asyncio.create_task(self.dispatcher.start())
        try:
            async with asyncio.timeout(timeout):
                while not predicate():
                    if task.done():
                        task.result()
                    await asyncio.sleep(0.01)
        finally:
            await self.dispatcher.stop()
        return task


@pytest.mark.asyncio
async def test_episodes_are_delivered_once_with_provenance():
    env = FakeEnv("env", group_size=2)
    h = Harness(envs=FakeEnvs(env), limit=2, step=7, version=3)
    await h.run_until(lambda: len(h.train) == 4)
    assert all(isinstance(item, vf.Episode) for item in h.train)
    groups = {item.group.id for item in h.train}
    assert len(groups) == 2
    run = h.train[0].run
    assert isinstance(run, vf.TrainRunInfo) and run.id == "run"
    assert run.work.type == "train" and run.work.step == 7 and run.work.policy.start == 3
    # every arrival landed in the ``all`` stream at the collecting step
    assert [(step, kind, subset) for _, step, kind, subset in h.monitors.episodes] == [(7, "train", "all")] * 4
    assert len(h.completed) == 4
    assert h.dispatcher.is_idle


@pytest.mark.asyncio
async def test_inflight_never_exceeds_the_cap():
    env = FakeEnv("env", group_size=4, run_delay=0.05)
    h = Harness(envs=FakeEnvs(env), max_inflight=2, limit=2)
    peak = 0

    async def watch():
        nonlocal peak
        while True:
            peak = max(peak, h.dispatcher.current_inflight)
            await asyncio.sleep(0.005)

    watcher = asyncio.create_task(watch())
    await h.run_until(lambda: len(h.train) == 8)
    watcher.cancel()
    assert peak == 2


@pytest.mark.asyncio
async def test_set_limit_raises_the_cap_and_gate_stops_train_scheduling():
    env = FakeEnv("env", group_size=2, run_delay=0.02)
    h = Harness(envs=FakeEnvs(env), max_inflight=1, limit=None)
    h.dispatcher.gate(False)
    task = asyncio.create_task(h.dispatcher.start())
    await asyncio.sleep(0.1)
    assert h.train == [] and h.dispatcher.current_inflight == 0
    h.dispatcher.gate(True)
    h.dispatcher.set_limit(3)
    await asyncio.sleep(0.15)
    assert len(h.train) > 0
    await h.dispatcher.stop()
    assert task.done()


@pytest.mark.asyncio
async def test_failed_requests_become_dispatch_failures():
    env = FakeEnv("env", group_size=2, fail_every=2)
    h = Harness(envs=FakeEnvs(env), limit=1)
    await h.run_until(lambda: len(h.train) == 2)
    kinds = sorted(type(item).__name__ for item in h.train)
    assert kinds == ["DispatchFailure", "Episode"]
    failure = next(item for item in h.train if isinstance(item, DispatchFailure))
    assert failure.error.type == "RuntimeError"
    assert h.dispatcher.gauges()["dispatcher/errored/train"] == 1


@pytest.mark.asyncio
async def test_pending_version_drops_stale_train_groups():
    env = FakeEnv("env", group_size=2, run_delay=10)
    h = Harness(envs=FakeEnvs(env), max_inflight=2, limit=1, step=5, version=0)
    task = asyncio.create_task(h.dispatcher.start())
    await asyncio.sleep(0.05)
    assert h.dispatcher.current_inflight == 2
    # step 5 trains v4; max_off_policy_steps=1 means v0 groups are past the bound
    await h.dispatcher.on_version_pending(5)
    await asyncio.sleep(0.05)
    assert len(h.train) == 1 and isinstance(h.train[0], GroupCancellation)
    assert h.train[0].reason == "stale" and h.train[0].count == 2
    assert h.dispatcher.policy_update_pending
    await h.dispatcher.on_new_version(5)
    assert not h.dispatcher.policy_update_pending
    await h.dispatcher.stop()
    assert task.done()


@pytest.mark.asyncio
async def test_eval_mode_schedules_the_epoch_then_returns_to_train():
    train_env = FakeEnv("train", group_size=1)
    eval_env = FakeEnv("eval", group_size=1, examples=[make_task(1), make_task(2)])
    h = Harness(envs=FakeEnvs(train_env), eval_envs=FakeEnvs(eval_env), limit=0)
    h.eval_source.trigger(0)
    h.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason="test")
    await h.run_until(lambda: len(h.eval) == 2)
    assert all(item.run.work.type == "eval" and item.run.work.step == 0 for item in h.eval)
    assert h.dispatcher.mode == DispatcherMode.PREFER_TRAIN
    assert h.train == []


@pytest.mark.asyncio
async def test_cancel_eval_step_covers_queued_and_active_groups():
    train_env = FakeEnv("train", group_size=1)
    eval_env = FakeEnv("eval", group_size=2, examples=[make_task(1), make_task(2), make_task(3)], run_delay=10)
    h = Harness(envs=FakeEnvs(train_env), eval_envs=FakeEnvs(eval_env), max_inflight=2, limit=0)
    h.eval_source.trigger(4)
    h.dispatcher.switch_mode(DispatcherMode.PREFER_EVAL, reason="test")
    task = asyncio.create_task(h.dispatcher.start())
    await asyncio.sleep(0.05)
    cancelled = await h.dispatcher.cancel_eval_step(4)
    await asyncio.sleep(0.05)
    assert cancelled == 6
    assert sum(item.count for item in h.eval if isinstance(item, GroupCancellation)) == 6
    assert all(item.reason == "superseded" for item in h.eval)
    await h.dispatcher.stop()
    assert task.done()


@pytest.mark.asyncio
async def test_a_failing_consumer_ends_the_dispatcher_even_with_a_full_buffer():
    env = FakeEnv("env", group_size=1)
    h = Harness(envs=FakeEnvs(env), max_inflight=8, limit=None)

    async def on_train(item):
        raise RuntimeError("sink exploded")

    h.dispatcher.bind(on_train=on_train)
    task = asyncio.create_task(h.dispatcher.start())
    with pytest.raises(RuntimeError, match="sink exploded"):
        await asyncio.wait_for(task, timeout=5.0)
    await h.dispatcher.stop()


@pytest.mark.asyncio
async def test_live_events_reach_the_monitors():
    env = FakeEnv("env", group_size=1)
    h = Harness(envs=FakeEnvs(env), limit=1)
    await h.run_until(lambda: len(h.train) == 1)
    events = [event for batch in h.monitors.live for event in batch]
    assert any("pending" in event for event in events)
    assert any("dispatched" in event for event in events)
