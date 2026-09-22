import asyncio
from unittest.mock import AsyncMock, MagicMock

from prime_rl.orchestrator.types import Policy
from prime_rl.orchestrator.watcher import WeightWatcher


def test_policy_update_activates_returned_model_before_notifying_observers():
    policy = Policy(version=0, model_name="Qwen/Qwen3-0.6B")
    receiver = MagicMock()
    receiver.wait_published = AsyncMock()
    receiver.receive = AsyncMock(return_value="prime-rl-policy-v1-test")
    observer = MagicMock()
    observer.on_version_pending = AsyncMock()

    async def assert_active_policy(step: int) -> None:
        assert step == 1
        assert policy.model_name == "prime-rl-policy-v1-test"
        assert policy.version == 1

    observer.on_new_version = AsyncMock(side_effect=assert_active_policy)
    watcher = WeightWatcher(receiver, policy=policy, observers=[observer])

    asyncio.run(watcher.apply_policy_update(1))

    receiver.receive.assert_awaited_once_with(1)
    observer.on_new_version.assert_awaited_once_with(1)
