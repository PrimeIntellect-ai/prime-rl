import asyncio

import pytest

from prime_rl.orchestrator import watcher as module
from prime_rl.orchestrator.watcher import WeightWatcher
from tests.unit.orchestrator.fakes import FakeReceiver, RecordingHooks


def make_watcher(monkeypatch=None):
    receiver = FakeReceiver()
    hooks = RecordingHooks()
    if monkeypatch is not None:
        monkeypatch.setattr(module, "POLL_INTERVAL", 0.01)
    watcher = WeightWatcher(receiver)
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
async def test_poll_loop_applies_published_versions_in_order(monkeypatch):
    watcher, receiver, hooks = make_watcher(monkeypatch)
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
async def test_pending_hook_errors_never_block_the_swap_but_new_version_errors_propagate():
    watcher, receiver, hooks = make_watcher()

    async def boom(step):
        raise RuntimeError("hook failed")

    watcher.bind(on_version_pending=[boom], on_new_version=[hooks.record_async("new")])
    receiver.publish(1)
    await watcher.apply(1)
    assert watcher.version == 1 and hooks["new"] == [(1,)]

    watcher.bind(on_new_version=[boom])
    receiver.publish(2)
    with pytest.raises(RuntimeError, match="hook failed"):
        await watcher.apply(2)
    assert watcher.version == 2  # the swap happened; the failed hook ends the run


@pytest.mark.asyncio
async def test_a_dead_watcher_unblocks_waiters(monkeypatch):
    watcher, receiver, _ = make_watcher(monkeypatch)
    receiver.next_version = lambda current: (_ for _ in ()).throw(RuntimeError("receiver broke"))
    task = asyncio.create_task(watcher.start())
    with pytest.raises(RuntimeError, match="stopped before"):
        await asyncio.wait_for(watcher.wait_for(3), timeout=2.0)
    with pytest.raises(RuntimeError, match="receiver broke"):
        await task
