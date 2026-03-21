import asyncio
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.scheduler import GroupState, InflightRolloutInfo, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_async_level = 1
    scheduler.strict_async_level = False
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(
        output_dir=Path("/tmp/prime-rl-test"),
        verification=SimpleNamespace(enabled=True),
    )
    scheduler.logger = MagicMock()
    scheduler.checkpoint_ready = asyncio.Event()
    scheduler.checkpoint_ready.set()
    scheduler.lora_name = None
    scheduler.model_name = "test-model"
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.max_error_reschedule_attempts = 3
    scheduler.rollouts_per_example = 2
    scheduler.empty_rollouts_by_task = defaultdict(int)
    scheduler.errored_rollouts_by_task = defaultdict(int)
    scheduler.total_rollouts_by_task = defaultdict(int)
    scheduler.completed_groups_by_task = defaultdict(int)
    scheduler.partial_groups_by_task = defaultdict(int)
    scheduler.error_family_counts = Counter()
    scheduler.deferred_group_scoring_tasks = set()
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
    return scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRolloutInfo(off_policy_steps=1, client_config=client, task="test", group_id=1),
            survivor_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRolloutInfo(
                    off_policy_steps=0,
                    client_config=client,
                    task="test",
                    group_id=3,
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


def test_maybe_update_policy_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(scheduler.maybe_update_policy())
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert scheduler.ckpt_step == 8

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler.update_policy_task = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())


def _make_success_rollout(task: str = "env_a", example_id: int = 1) -> dict:
    return {
        "task": task,
        "example_id": example_id,
        "trajectory": [{"tokens": {}}],
        "reward": 1.0,
        "metrics": {},
        "timing": {"generation_ms": 0.0, "scoring_ms": 0.0},
        "error": None,
    }


def test_errored_rollout_reschedules_before_retry_budget():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.max_error_reschedule_attempts = 2
        group = GroupState(
            example={"task": "env_a", "example_id": 1},
            rollouts_to_schedule=0,
            target_rollouts=2,
        )

        await scheduler._handle_errored_rollout(
            group_id=1,
            group=group,
            task="env_a",
            error_info={"error": "SandboxError", "error_chain_repr": "vf.SandboxError('boom')"},
        )

        assert group.rollouts_to_schedule == 1
        assert group.error_reschedule_attempts == 1
        assert group.target_rollouts == 2
        assert scheduler.error_family_counts["vf.Error"] == 1
        assert scheduler.error_family_counts["vf.InfraError"] == 1
        assert scheduler.error_family_counts["vf.SandboxError"] == 1

    asyncio.run(run())


def test_errored_rollout_shrinks_group_after_retry_budget():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.max_error_reschedule_attempts = 1
        group = GroupState(
            example={"task": "env_a", "example_id": 1},
            rollouts_to_schedule=0,
            target_rollouts=2,
            completed_rollouts=[_make_success_rollout()],
            error_reschedule_attempts=1,
        )
        scheduler.groups = {1: group}
        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        leftover_task = asyncio.create_task(asyncio.sleep(60))
        scheduler.inflight_requests = {
            leftover_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="env_a", group_id=1)
        }

        completed_group = await scheduler._handle_errored_rollout(
            group_id=1,
            group=group,
            task="env_a",
            error_info={"error": "SandboxError", "error_chain_repr": "vf.SandboxError('boom')"},
        )

        assert completed_group is not None
        assert completed_group.was_shrunk is True
        assert completed_group.target_rollouts == 1
        assert 1 not in scheduler.groups
        assert scheduler.partial_groups_by_task["env_a"] == 1
        assert scheduler.completed_groups_by_task["env_a"] == 1
        assert leftover_task.cancelled()

    asyncio.run(run())


def test_errored_rollout_drops_deferred_scoring_group_after_retry_budget():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.max_error_reschedule_attempts = 0
        scheduler.deferred_group_scoring_tasks = {"env_a"}
        group = GroupState(
            example={"task": "env_a", "example_id": 1},
            rollouts_to_schedule=0,
            target_rollouts=2,
        )
        scheduler.groups = {1: group}
        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        leftover_task = asyncio.create_task(asyncio.sleep(60))
        scheduler.inflight_requests = {
            leftover_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="env_a", group_id=1)
        }

        completed_group = await scheduler._handle_errored_rollout(
            group_id=1,
            group=group,
            task="env_a",
            error_info={"error": "SandboxError", "error_chain_repr": "vf.SandboxError('boom')"},
        )

        assert completed_group is None
        assert 1 not in scheduler.groups
        assert scheduler.partial_groups_by_task["env_a"] == 0
        assert scheduler.completed_groups_by_task["env_a"] == 0
        assert leftover_task.cancelled()

    asyncio.run(run())


def test_get_metrics_reports_partial_groups_and_error_families():
    scheduler = make_scheduler()
    scheduler.total_rollouts_by_task["env_a"] = 4
    scheduler.errored_rollouts_by_task["env_a"] = 2
    scheduler.completed_groups_by_task["env_a"] = 2
    scheduler.partial_groups_by_task["env_a"] = 1
    scheduler.error_family_counts.update(
        {
            "vf.Error": 2,
            "vf.InfraError": 2,
            "vf.SandboxError": 1,
        }
    )

    metrics = scheduler.get_metrics()

    assert metrics["partial_groups/all"] == 0.5
    assert metrics["partial_groups/env_a"] == 0.5
    assert metrics["error/vf.Error"] == 0.5
    assert metrics["error/vf.InfraError"] == 0.5
    assert metrics["error/vf.SandboxError"] == 0.25
    assert scheduler.completed_groups_by_task == {}
    assert scheduler.partial_groups_by_task == {}
    assert scheduler.error_family_counts == Counter()
