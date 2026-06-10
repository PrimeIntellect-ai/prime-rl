"""Dispatcher starvation regression tests (crash1 failure class).

Dropped or errored groups must never silently shrink dispatch capacity:
either the fill loop backfills fresh groups from the infinite train source
so the batch target stays reachable, or ``raise_if_starved`` crashes loudly.
There is no silent-stall path.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

import pytest
import verifiers as vf

from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Policy, TrainRollout


@dataclass
class FakeClient:
    api_base_url: str = "http://fake:8000/v1"
    extra_headers: dict = field(default_factory=dict)


class FakePool:
    def __init__(self) -> None:
        self.client = FakeClient()
        self.model_name = "fake-model"

    async def select_train_client(self, load) -> FakeClient:
        return self.client

    async def get_eval_client(self) -> FakeClient:
        return self.client


@dataclass
class FakeEnvConfig:
    group_size: int
    ratio: float | None = 1.0


class FakeEnv:
    """Scripted env: ``behaviors`` is consumed one entry per ``run_rollout``
    call — "ok" returns a healthy rollout, "error" raises, "hang" blocks
    until cancelled (so the rollout is in flight when a drop hits it).
    An exhausted script defaults to "ok"."""

    def __init__(
        self,
        *,
        name: str = "fake-env",
        group_size: int = 2,
        requires_group_scoring: bool = False,
        behaviors: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.config = FakeEnvConfig(group_size=group_size)
        self.requires_group_scoring = requires_group_scoring
        self.is_multi_agent = False
        self.behaviors = deque(behaviors)

    def get_dataset(self, seed: int | None = None) -> list[dict]:
        return [{"example_id": i} for i in range(4)]

    async def run_rollout(self, *, client, example, model_name, cache_salt, generation, dispatch_id):
        behavior = self.behaviors.popleft() if self.behaviors else "ok"
        if behavior == "error":
            raise RuntimeError("scripted rollout failure")
        if behavior == "hang":
            await asyncio.Event().wait()
        out = vf.RolloutOutput()
        out["example_id"] = example["example_id"]
        out["error"] = None
        out["trajectory"] = [{"tokens": [1]}]
        out["reward"] = 1.0
        return out


class FakeEnvs:
    def __init__(self, *envs: FakeEnv) -> None:
        self._envs = {env.name: env for env in envs}

    def get(self, name: str) -> FakeEnv:
        return self._envs[name]

    def __iter__(self):
        return iter(self._envs.values())


def make_dispatcher(env: FakeEnv, *, max_inflight: int = 4, max_off_policy_steps: int = 8) -> RolloutDispatcher:
    envs = FakeEnvs(env)
    return RolloutDispatcher(
        train_envs=envs,
        eval_envs=None,
        train_source=TrainSource(envs, seed=0),
        eval_source=None,
        inference=FakePool(),
        eval_inference=FakePool(),
        policy=Policy(version=0, model_name="fake-model"),
        max_inflight_rollouts=max_inflight,
        tasks_per_minute=None,
        max_off_policy_steps=max_off_policy_steps,
        training_mode="rl",
    )


def test_errored_groups_are_backfilled_until_batch_target():
    """Sustained early errors must not stall the pipeline: the fill loop
    keeps opening fresh groups, so survivors still reach the batch target."""

    async def scenario() -> None:
        env = FakeEnv(group_size=2, behaviors=("error",) * 4)
        dispatcher = make_dispatcher(env)
        runner = asyncio.create_task(dispatcher.start())

        survivors = 0
        errored = 0
        async with asyncio.timeout(10):
            while survivors < 4:
                rollout = await dispatcher.out_q.get()
                assert isinstance(rollout, TrainRollout)
                if rollout.raw["error"] is None:
                    survivors += 1
                else:
                    errored += 1
        assert errored == 4
        await dispatcher.stop()
        assert runner.done()

    asyncio.run(scenario())


def test_dropped_groups_emit_markers_and_are_backfilled():
    """Off-policy drops emit one Cancelled marker per owed rollout, restore
    every permit, and the freed capacity is refilled with fresh groups."""

    async def scenario() -> None:
        env = FakeEnv(group_size=2, behaviors=("hang",) * 4)
        dispatcher = make_dispatcher(env, max_off_policy_steps=0)
        runner = asyncio.create_task(dispatcher.start())

        async with asyncio.timeout(10):
            while dispatcher.available_permits > 0:
                await asyncio.sleep(0.01)
            await dispatcher.on_new_version(1)  # every inflight group is now past max_off_policy_steps

            cancelled = 0
            survivors = 0
            while survivors < 2:
                rollout = await dispatcher.out_q.get()
                if rollout.raw["error"] is None:
                    survivors += 1
                else:
                    assert rollout.raw["error"]["error"] == "Cancelled"
                    cancelled += 1

        assert cancelled == 4
        assert dispatcher.dropped_group_count == 2
        assert dispatcher.cancelled_rollout_count == 4
        await dispatcher.stop()
        assert runner.done()
        assert dispatcher.inflight_permits == 0

    asyncio.run(scenario())


def test_starvation_invariant_fires_when_nothing_is_schedulable():
    """A live fill loop that can never schedule (permit cost > max inflight)
    must crash via ``raise_if_starved`` instead of idling forever."""

    async def scenario() -> None:
        env = FakeEnv(group_size=16, requires_group_scoring=True)
        dispatcher = make_dispatcher(env, max_inflight=4)
        dispatcher.starvation_timeout = 0.05
        runner = asyncio.create_task(dispatcher.start())

        with pytest.raises(RuntimeError, match="Dispatcher starved") as excinfo:
            async with asyncio.timeout(10):
                while True:
                    dispatcher.raise_if_starved(batch_progress=(296, 512, "rollouts"))
                    await asyncio.sleep(0.01)
        message = str(excinfo.value)
        assert "296/512 rollouts" in message
        assert "dropped_groups=0" in message
        assert "permits=0/4" in message
        await dispatcher.stop()
        assert runner.done()

    asyncio.run(scenario())


def test_starvation_invariant_respects_intentional_pauses():
    """Gate cleared (trainer behind) and drain mode (train scheduling
    disabled, no eval queued) are intentional pauses, not starvation."""

    env = FakeEnv(group_size=2)
    dispatcher = make_dispatcher(env)
    dispatcher.starvation_timeout = 0.0

    dispatcher.dispatch_allowed.clear()
    dispatcher.raise_if_starved()
    dispatcher.raise_if_starved()
    assert dispatcher.starved_since is None

    dispatcher.dispatch_allowed.set()
    dispatcher.disable_train_scheduling()
    dispatcher.raise_if_starved()
    dispatcher.raise_if_starved()
    assert dispatcher.starved_since is None

    # Re-enabled and idle: first call arms the timer, second call fires.
    dispatcher.train_scheduling_disabled = False
    dispatcher.raise_if_starved()
    with pytest.raises(RuntimeError, match="Dispatcher starved"):
        dispatcher.raise_if_starved()
