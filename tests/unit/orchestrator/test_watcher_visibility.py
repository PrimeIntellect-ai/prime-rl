import asyncio
from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.watcher import WeightWatcher


def test_orchestrator_raises_when_component_task_fails():
    async def run() -> None:
        async def fail() -> None:
            raise ValueError("watcher boom")

        task = asyncio.create_task(fail(), name="watcher")
        await asyncio.sleep(0)

        orchestrator = object.__new__(Orchestrator)
        orchestrator.component_tasks = [task]

        with pytest.raises(RuntimeError, match="Pipeline component 'watcher' died") as exc_info:
            orchestrator.check_pipeline_health()

        assert isinstance(exc_info.value.__cause__, ValueError)

    asyncio.run(run())


def test_orchestrator_raises_when_component_task_exits_unexpectedly():
    async def run() -> None:
        async def exit_cleanly() -> None:
            return None

        task = asyncio.create_task(exit_cleanly(), name="watcher")
        await asyncio.sleep(0)

        orchestrator = object.__new__(Orchestrator)
        orchestrator.component_tasks = [task]

        with pytest.raises(RuntimeError, match="Pipeline component 'watcher' exited unexpectedly"):
            orchestrator.check_pipeline_health()

    asyncio.run(run())


def test_watcher_gauges_expose_progress_and_health():
    watcher = object.__new__(WeightWatcher)
    watcher.policy = SimpleNamespace(version=3)
    watcher.update_count = 2
    watcher.last_update_weights_time = 1.5
    watcher.last_wait_for_ckpt_time = 0.25
    watcher.ckpt_step = 2
    watcher.last_seen_broadcast_step = 4
    watcher.next_ckpt_step = 4
    watcher.current_update_step = 4
    watcher.last_error = None

    assert watcher.gauges() == {
        "watcher/policy_version": 3.0,
        "watcher/update_count": 2.0,
        "watcher/last_update_weights_time": 1.5,
        "watcher/last_wait_for_ckpt_time": 0.25,
        "watcher/last_seen_broadcast_step": 4.0,
        "watcher/next_ckpt_step": 4.0,
        "watcher/is_updating": 1.0,
        "watcher/current_update_step": 4.0,
        "watcher/healthy": 1.0,
    }


def test_watcher_gauges_mark_unhealthy_after_error():
    watcher = object.__new__(WeightWatcher)
    watcher.policy = SimpleNamespace(version=3)
    watcher.update_count = 2
    watcher.last_update_weights_time = 1.5
    watcher.last_wait_for_ckpt_time = 0.25
    watcher.ckpt_step = 2
    watcher.last_seen_broadcast_step = 4
    watcher.next_ckpt_step = 4
    watcher.current_update_step = None
    watcher.last_error = "RuntimeError('boom')"

    gauges = watcher.gauges()

    assert gauges["watcher/is_updating"] == 0.0
    assert gauges["watcher/current_update_step"] == 0.0
    assert gauges["watcher/healthy"] == 0.0
