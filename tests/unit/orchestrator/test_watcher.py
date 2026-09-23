import asyncio

import pytest

from prime_rl.configs.orchestrator import WatcherConfig
from prime_rl.orchestrator.watcher import WeightWatcher
from tests.unit.orchestrator.fakes import FakeReceiver, RecordingHooks


def make_watcher(poll: float = 0.01):
    receiver = FakeReceiver()
    hooks = RecordingHooks()
    watcher = WeightWatcher(WatcherConfig(poll_interval=poll), receiver)
    watcher.bind(
        on_version_pending=[hooks.record_async("pending")],
        on_new_version=[hooks.record_async("new")],
    )
    return watcher, receiver, hooks


@pytest.mark.asyncio
async def test_sync_startup_adopts_the_version_and_notifies():
    watcher, receiver, hooks = make_watcher()
    await watcher.sync_startup(3, timeout=1.0)
    assert watcher.version == 3
    assert receiver.synced == [3]
    assert hooks["new"] == [(3,)] and hooks["pending"] == []


@pytest.mark.asyncio
async def test_apply_drains_before_receiving_and_advances_after():
    watcher, receiver, hooks = make_watcher()
    order = []
    watcher.bind(
        on_version_pending=[lambda step: order.append(("pending", step, watcher.version)) or asyncio.sleep(0)],
        on_new_version=[lambda step: order.append(("new", step, watcher.version)) or asyncio.sleep(0)],
    )
    receiver.publish(1)
    await watcher.apply(1)
    assert receiver.received == [1]
    assert order == [("pending", 1, 0), ("new", 1, 1)]
    # an already-applied version is a no-op
    await watcher.apply(1)
    assert receiver.received == [1]


@pytest.mark.asyncio
async def test_poll_loop_applies_published_versions_in_order():
    watcher, receiver, hooks = make_watcher()
    task = asyncio.create_task(watcher.start())
    receiver.publish(1)
    receiver.publish(2)
    await asyncio.wait_for(watcher.wait_for(2, timeout=2.0), timeout=3.0)
    assert receiver.received == [1, 2]
    assert [call[0] for call in hooks["new"]] == [1, 2]
    await watcher.stop()
    assert task.done()


@pytest.mark.asyncio
async def test_wait_for_times_out_without_a_version():
    watcher, _, _ = make_watcher()
    assert await watcher.wait_for(5, timeout=0.05) is False
    assert await watcher.wait_for(0) is True


@pytest.mark.asyncio
async def test_hook_errors_do_not_stop_the_update():
    watcher, receiver, hooks = make_watcher()

    async def boom(step):
        raise RuntimeError("hook failed")

    watcher.bind(on_version_pending=[boom], on_new_version=[boom, hooks.record_async("new")])
    receiver.publish(1)
    await watcher.apply(1)
    assert watcher.version == 1 and hooks["new"] == [(1,)]
