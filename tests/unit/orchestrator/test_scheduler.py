import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import MagicMock

from prime_rl.orchestrator.scheduler import InflightRolloutInfo, Scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()
        scheduler.inflight_request_counts = Counter()

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {}
        scheduler._track_inflight_request(
            stale_task,
            InflightRolloutInfo(off_policy_steps=1, client_config=client, group_id=1),
        )
        scheduler._track_inflight_request(
            survivor_task,
            InflightRolloutInfo(off_policy_steps=0, client_config=client, group_id=2),
        )

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler._pop_inflight_request(task)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler._track_inflight_request(
                    interleaved_task,
                    InflightRolloutInfo(
                        off_policy_steps=0,
                        client_config=client,
                        group_id=3,
                    ),
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_client_load_tracking_matches_inflight_requests():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.inflight_requests = {}
        scheduler.inflight_request_counts = Counter()

        client_a = SimpleNamespace(api_base_url="http://a", extra_headers={})
        client_b = SimpleNamespace(api_base_url="http://b", extra_headers={})

        task_a = asyncio.create_task(asyncio.sleep(60))
        task_b = asyncio.create_task(asyncio.sleep(60))
        task_c = asyncio.create_task(asyncio.sleep(60))

        def expected_counts() -> Counter:
            return Counter(
                scheduler._client_identity(info.client_config) for info in scheduler.inflight_requests.values()
            )

        scheduler._track_inflight_request(
            task_a,
            InflightRolloutInfo(off_policy_steps=0, client_config=client_a, group_id=1),
        )
        scheduler._track_inflight_request(
            task_b,
            InflightRolloutInfo(off_policy_steps=0, client_config=client_a, group_id=2),
        )
        scheduler._track_inflight_request(
            task_c,
            InflightRolloutInfo(off_policy_steps=0, client_config=client_b, group_id=3),
        )
        assert scheduler.inflight_request_counts == expected_counts()

        scheduler._track_inflight_request(
            task_a,
            InflightRolloutInfo(off_policy_steps=1, client_config=client_a, group_id=1),
        )
        assert scheduler.inflight_request_counts == expected_counts()

        assert scheduler._pop_inflight_request(task_b) is not None
        assert scheduler.inflight_request_counts == expected_counts()

        for task in (task_a, task_b, task_c):
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_drop_group_and_cancel_inflight_keep_client_loads_consistent():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.inflight_requests = {}
        scheduler.inflight_request_counts = Counter()
        scheduler.groups = {1: object(), 2: object()}
        scheduler.cancelled_rollouts_count = 0

        client_a = SimpleNamespace(api_base_url="http://a", extra_headers={})
        client_b = SimpleNamespace(api_base_url="http://b", extra_headers={})

        group_1_task_a = asyncio.create_task(asyncio.sleep(60))
        group_1_task_b = asyncio.create_task(asyncio.sleep(60))
        group_2_task = asyncio.create_task(asyncio.sleep(60))

        def expected_counts() -> Counter:
            return Counter(
                scheduler._client_identity(info.client_config) for info in scheduler.inflight_requests.values()
            )

        scheduler._track_inflight_request(
            group_1_task_a,
            InflightRolloutInfo(off_policy_steps=0, client_config=client_a, group_id=1),
        )
        scheduler._track_inflight_request(
            group_1_task_b,
            InflightRolloutInfo(off_policy_steps=0, client_config=client_b, group_id=1),
        )
        scheduler._track_inflight_request(
            group_2_task,
            InflightRolloutInfo(off_policy_steps=0, client_config=client_b, group_id=2),
        )

        removed = await scheduler.drop_group(1)
        assert removed == 2
        assert scheduler.inflight_request_counts == expected_counts()
        assert group_2_task in scheduler.inflight_requests
        assert group_1_task_a not in scheduler.inflight_requests
        assert group_1_task_b not in scheduler.inflight_requests

        await scheduler.cancel_inflight_rollouts()
        assert scheduler.inflight_requests == {}
        assert scheduler.inflight_request_counts == Counter()

        for task in (group_1_task_a, group_1_task_b, group_2_task):
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_select_least_loaded_client_uses_identity_counts_after_refresh():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.inflight_requests = {}
        scheduler.inflight_request_counts = Counter()

        old_a = SimpleNamespace(api_base_url="http://a", extra_headers={"X-data-parallel-rank": "0"})
        old_b = SimpleNamespace(api_base_url="http://b", extra_headers={"X-data-parallel-rank": "0"})

        task_a = asyncio.create_task(asyncio.sleep(60))
        task_b1 = asyncio.create_task(asyncio.sleep(60))
        task_b2 = asyncio.create_task(asyncio.sleep(60))

        scheduler._track_inflight_request(
            task_a,
            InflightRolloutInfo(off_policy_steps=0, client_config=old_a, group_id=1),
        )
        scheduler._track_inflight_request(
            task_b1,
            InflightRolloutInfo(off_policy_steps=0, client_config=old_b, group_id=2),
        )
        scheduler._track_inflight_request(
            task_b2,
            InflightRolloutInfo(off_policy_steps=0, client_config=old_b, group_id=3),
        )

        refreshed_a = SimpleNamespace(api_base_url="http://a", extra_headers={"X-data-parallel-rank": "0"})
        refreshed_b = SimpleNamespace(api_base_url="http://b", extra_headers={"X-data-parallel-rank": "0"})
        scheduler.inference_pool = SimpleNamespace(clients=[refreshed_a, refreshed_b])

        selected = await scheduler._select_least_loaded_client()
        assert selected is refreshed_a

        for task in (task_a, task_b1, task_b2):
            if not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())
