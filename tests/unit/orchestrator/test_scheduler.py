import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.scheduler import GroupState, InflightRequest, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_async_level = 1
    scheduler.strict_async_level = False
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(output_dir=Path("/tmp/prime-rl-test"))
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
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.enable_policy_updates = True
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
            stale_task: InflightRequest(off_policy_steps=1, client_config=client, env_name="test", group_id=1),
            survivor_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="test", group_id=2),
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
                scheduler.inflight_requests[interleaved_task] = InflightRequest(
                    off_policy_steps=0,
                    client_config=client,
                    env_name="test",
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


def test_generate_batch_uses_schedule_time_policy_step():
    class DummyProgressTracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, _value):
            pass

        def close(self):
            pass

    class FakeBuffer:
        def __init__(self):
            self.rollouts = []

        def update(self, rollouts):
            self.rollouts.extend(rollouts)

        def sample_rollouts(self, n: int):
            return self.rollouts[-n:]

    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_policy_updates = False
        scheduler.checkpoint_ready = asyncio.Event()
        scheduler.checkpoint_ready.set()
        scheduler.logger = MagicMock()
        scheduler.json_logging = False
        scheduler.batch_size = 1
        scheduler.token_batch_size = None
        scheduler.rollouts_per_example = 1
        scheduler.max_async_level = 0
        scheduler.step = 0
        scheduler.ckpt_step = 11
        scheduler.last_batch_generation_time = 0.0
        scheduler.max_inflight_rollouts = 1
        scheduler.buffer = FakeBuffer()
        scheduler.inflight_requests = {}
        scheduler.groups = {}
        scheduler.total_rollouts_by_env = defaultdict(int)
        scheduler.empty_rollouts_by_env = defaultdict(int)
        scheduler.errored_rollouts_by_env = defaultdict(int)
        scheduler.cancelled_rollouts_count = 0
        scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
        scheduler.train_envs = SimpleNamespace(get=lambda _env_name: SimpleNamespace(requires_group_scoring=False))
        scheduler._fill_inflight_requests = AsyncMock()

        finished = asyncio.Future()
        finished.set_result(
            {
                "example_id": 1,
                "reward": 1.0,
                "error": None,
                "trajectory": [
                    {
                        "tokens": {
                            "prompt_ids": [1],
                            "prompt_mask": [False],
                            "completion_ids": [2],
                            "completion_mask": [True],
                            "completion_logprobs": [-0.1],
                        }
                    }
                ],
                "timing": {"generation_ms": 1.0, "scoring_ms": 1.0},
                "metrics": {},
                "is_truncated": False,
            }
        )

        group_id = 7
        scheduler.inflight_requests = {
            finished: InflightRequest(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                env_name="math",
                policy_step=5,
                group_id=group_id,
            )
        }
        scheduler.groups = {group_id: GroupState(example={"env_name": "math"}, rollouts_to_schedule=0)}

        with patch("prime_rl.orchestrator.scheduler.ProgressTracker", DummyProgressTracker):
            batch = await scheduler.generate_batch(step=12, target=1)

        assert batch[0]["env_name"] == "math"
        assert batch[0]["policy_step"] == 5

    asyncio.run(run())
