import asyncio
from types import SimpleNamespace

import pytest

import prime_rl.orchestrator.watcher as watcher_module
from prime_rl.orchestrator.types import Policy
from prime_rl.orchestrator.watcher import WeightWatcher


class Receiver:
    """A receiver whose replacement trainer offers v4; ``reconnect`` fails ``failures`` times or hangs."""

    def __init__(self, *, retryable: bool, failures: int = 0, hang: bool = False):
        self.config = SimpleNamespace(timeout=60)
        self.admin_plane = SimpleNamespace(update_times={})
        self.aborted = asyncio.Event()
        self.reconnect_retryable = retryable
        self.failures = failures
        self.hang = hang
        self.reconnects = 0
        self.reconnect_started = asyncio.Event()
        self.received: list[int] = []

    def abort(self) -> None:
        self.aborted.set()

    def clear_abort(self) -> None:
        self.aborted.clear()

    def clear_offers(self) -> None:
        pass

    def next_version(self, current: int) -> int:
        return max(current, 4)

    async def wait_published(self, step, cancelled=None) -> None:
        pass

    async def reconnect(self) -> None:
        self.reconnects += 1
        self.reconnect_started.set()
        if self.hang:
            await asyncio.Event().wait()
        if self.reconnects <= self.failures:
            raise RuntimeError("NCCL initialization failed")

    async def receive(self, step: int) -> None:
        self.received.append(step)


class Observer:
    def __init__(self):
        self.versions: list[int] = []

    async def on_version_pending(self, step: int) -> None:
        pass

    async def on_new_version(self, step: int) -> None:
        self.versions.append(step)


async def reconciled(receiver: Receiver) -> tuple[WeightWatcher, Observer]:
    """A watcher serving complete v3 that reconciled a replacement restored at step 3."""
    observer = Observer()
    watcher = WeightWatcher(
        receiver, policy=Policy(version=3, model_name="model"), observers=[observer], trainer_recovery=True
    )
    watcher.ckpt_step, watcher.weights_valid = 3, True
    await watcher.reconcile_replacement(3)
    return watcher, observer


def test_a_failed_one_shot_rejoin_is_not_retried_and_keeps_the_policy():
    """NCCL workers that joined the replacement's group are not torn down by another rejoin; inference
    skips the offer it cannot receive and keeps serving its complete policy."""

    async def run():
        receiver = Receiver(retryable=False, failures=1)
        watcher, observer = await reconciled(receiver)
        await watcher.apply_policy_update(4)
        assert receiver.reconnects == 1 and receiver.received == []
        assert watcher.weights_valid and watcher.invalid_since is None
        assert watcher.policy.version == 3 and watcher.ckpt_step == 4
        assert observer.versions == [3] and watcher.reconnecting is None

    asyncio.run(run())


def test_a_failed_rejoin_is_retried_when_the_transport_can_join_again(monkeypatch):
    monkeypatch.setattr(watcher_module, "REJOIN_RETRY_INTERVAL_S", 0.01)

    async def run():
        receiver = Receiver(retryable=True, failures=1)
        watcher, observer = await reconciled(receiver)
        await watcher.apply_policy_update(4)
        assert receiver.reconnects == 2 and receiver.received == [4]
        assert watcher.policy.version == 4 and observer.versions == [4]

    asyncio.run(run())


@pytest.mark.parametrize("retryable", [False, True])
def test_stop_ends_a_rejoin_with_a_dead_trainer(retryable):
    async def run():
        receiver = Receiver(retryable=retryable, hang=True)
        watcher, _ = await reconciled(receiver)
        update = asyncio.create_task(watcher.apply_policy_update(4))
        await receiver.reconnect_started.wait()
        await asyncio.wait_for(watcher.stop(), 5)
        assert update.done() and receiver.received == []
        assert watcher.reconnecting is None

    asyncio.run(run())
