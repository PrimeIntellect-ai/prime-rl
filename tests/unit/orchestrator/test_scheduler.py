import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import verifiers as vf

from prime_rl.configs.multi_agent import FixedMemberTargetConfig, MultiAgentConfig
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


def test_client_identity_distinguishes_base_url_and_dp_rank():
    client_a = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers={"X-data-parallel-rank": "0"},
    )
    client_b = vf.ClientConfig(
        api_base_url="http://worker-a:8000/v1",
        extra_headers={"X-data-parallel-rank": "1"},
    )

    assert Scheduler._client_identity(client_a) != Scheduler._client_identity(client_b)


def test_scheduler_excludes_examples_already_reserved_for_current_batch():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_inflight_rollouts = 64
        scheduler.rollouts_per_example = 8
        scheduler.inflight_requests = {}
        scheduler.groups = {}
        scheduler.next_group_id = 0

        examples = [{"env_name": "env", "example_id": i} for i in range(4)]
        seen_exclusions = []

        def sample_examples(n: int, exclude_keys=None):
            assert n == 1
            excluded = set(exclude_keys or ())
            seen_exclusions.append(set(excluded))
            for example in examples:
                key = (example["env_name"], example["example_id"])
                if key not in excluded:
                    return [example]
            raise AssertionError("test ran out of examples")

        async def schedule_rollout(group_id: int) -> None:
            scheduler.groups[group_id].rollouts_to_schedule = 0

        scheduler.buffer = SimpleNamespace(sample_examples=sample_examples)
        scheduler.schedule_rollout = schedule_rollout

        batch_keys = set()
        assert await scheduler._schedule_next_request(batch_keys)
        assert await scheduler._schedule_next_request(batch_keys)
        assert await scheduler._schedule_next_request(batch_keys)

        assert [group.example["example_id"] for group in scheduler.groups.values()] == [0, 1, 2]
        assert seen_exclusions == [
            set(),
            {("env", 0)},
            {("env", 0), ("env", 1)},
        ]

    asyncio.run(run())


def test_scheduler_releases_completed_batch_keys_after_unique_pool_exhausted():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_inflight_rollouts = 16
        scheduler.rollouts_per_example = 8
        scheduler.inflight_requests = {}
        scheduler.groups = {
            0: GroupState(example={"env_name": "env", "example_id": 0}, rollouts_to_schedule=0),
        }
        scheduler.next_group_id = 1
        scheduler.logger = MagicMock()

        examples = [{"env_name": "env", "example_id": i} for i in range(2)]
        seen_exclusions = []

        def sample_examples(n: int, exclude_keys=None):
            assert n == 1
            excluded = set(exclude_keys or ())
            seen_exclusions.append(set(excluded))
            for example in examples:
                key = (example["env_name"], example["example_id"])
                if key not in excluded:
                    return [example]
            raise ValueError("No environments left with examples.")

        async def schedule_rollout(group_id: int) -> None:
            scheduler.groups[group_id].rollouts_to_schedule = 0

        scheduler.buffer = SimpleNamespace(sample_examples=sample_examples)
        scheduler.schedule_rollout = schedule_rollout

        batch_keys = {("env", 0), ("env", 1)}
        assert await scheduler._schedule_next_request(batch_keys)

        assert scheduler.groups[1].example["example_id"] == 1
        assert seen_exclusions == [
            {("env", 0), ("env", 1)},
            {("env", 0)},
        ]
        assert batch_keys == {("env", 0), ("env", 1)}
        scheduler.logger.warning.assert_called_once()

    asyncio.run(run())


def test_compile_generation_ignores_single_agent_env_with_global_config():
    scheduler = make_scheduler()
    scheduler.config.multi_agent = MultiAgentConfig(
        fixed={
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge/v1"],
            )
        }
    )
    env = SimpleNamespace(
        is_multi_agent=False,
        compile_generation=MagicMock(side_effect=AssertionError("should not compile")),
    )

    generation = scheduler._compile_generation(
        env=env,
        client_config=vf.ClientConfig(api_base_url="http://learner/v1"),
        cache_salt="1",
        dispatch_id="dispatch-1",
    )

    assert generation is None
    env.compile_generation.assert_not_called()


def test_compile_generation_routes_multi_agent_env_with_global_config():
    scheduler = make_scheduler()
    scheduler.config.multi_agent = MultiAgentConfig(
        fixed={
            "judge": FixedMemberTargetConfig(
                members=["judge"],
                model="judge-model",
                base_url=["http://judge/v1"],
            )
        }
    )
    expected = vf.MemberGenerationPlan()
    env = SimpleNamespace(
        is_multi_agent=True,
        compile_generation=MagicMock(return_value=expected),
    )
    client = vf.ClientConfig(api_base_url="http://learner/v1")

    generation = scheduler._compile_generation(
        env=env,
        client_config=client,
        cache_salt="1",
        dispatch_id="dispatch-1",
    )

    assert generation is expected
    env.compile_generation.assert_called_once_with(
        scheduler.config.multi_agent,
        client=client,
        model_name="test-model",
        cache_salt="1",
        dispatch_id="dispatch-1",
    )


def test_pause_policy_updates_cancels_poller_and_waits_for_inflight_update():
    async def run() -> None:
        scheduler = make_scheduler()
        poller_cancelled = asyncio.Event()
        update_finished = asyncio.Event()

        async def policy_poller() -> None:
            try:
                await asyncio.Future()
            finally:
                poller_cancelled.set()

        async def inflight_update() -> None:
            await asyncio.sleep(0)
            update_finished.set()

        scheduler.update_policy_task = asyncio.create_task(policy_poller())
        scheduler.inflight_policy_update_task = asyncio.create_task(inflight_update())

        await asyncio.sleep(0)
        await scheduler.pause_policy_updates()

        assert poller_cancelled.is_set()
        assert update_finished.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())


def test_sync_policy_for_step_waits_for_checkpoint_before_batch_generation():
    async def run() -> None:
        scheduler = make_scheduler()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=0),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            await scheduler.sync_policy_for_step(11)

        assert applied_steps == [10]
        assert scheduler.step == 11
        assert scheduler.ckpt_step == 10
        assert scheduler.update_policy_task is not None

        await scheduler.stop()

    asyncio.run(run())


def test_generate_batch_drops_timeout_error_and_continues():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.enable_policy_updates = False
        scheduler.batch_size = 1
        scheduler.token_batch_size = None
        scheduler.rollouts_per_example = 1
        scheduler.json_logging = True
        scheduler.empty_rollouts_by_env = defaultdict(int)
        scheduler.errored_rollouts_by_env = defaultdict(int)
        scheduler.total_rollouts_by_env = defaultdict(int)
        scheduler.deferred_group_scoring_tasks = set()
        scheduler._fill_inflight_requests = AsyncMock()
        scheduler.buffer = MagicMock()
        scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
        scheduler.train_envs = SimpleNamespace(get=lambda _name: SimpleNamespace(requires_group_scoring=False))

        client = SimpleNamespace(api_base_url="http://test", extra_headers={})
        success_rollout = {
            "task": "debate",
            "trajectory": [{"tokens": None}],
            "error": None,
        }

        async def raise_timeout():
            raise TimeoutError("Environment timeout for run_group request after 30s")

        async def return_rollout():
            await asyncio.sleep(0.01)
            return success_rollout

        timeout_task = asyncio.create_task(raise_timeout())
        success_task = asyncio.create_task(return_rollout())

        scheduler.groups = {
            1: GroupState(example={"env_name": "debate", "example_id": 1}, rollouts_to_schedule=0),
            2: GroupState(example={"env_name": "debate", "example_id": 2}, rollouts_to_schedule=0),
        }
        scheduler.inflight_requests = {
            timeout_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="debate", group_id=1),
            success_task: InflightRequest(off_policy_steps=0, client_config=client, env_name="debate", group_id=2),
        }
        scheduler.buffer.sample_rollouts.return_value = [success_rollout]
        scheduler.sync_policy_for_step = AsyncMock(side_effect=AssertionError("generate_batch must not resync policy"))

        with patch("prime_rl.orchestrator.scheduler.ProgressTracker") as progress_tracker:
            progress_tracker.return_value = SimpleNamespace(update=MagicMock(), close=MagicMock())
            batch = await scheduler.generate_batch(step=3)

        scheduler.sync_policy_for_step.assert_not_called()
        assert batch == [success_rollout]
        assert 1 not in scheduler.groups
        assert 2 not in scheduler.groups
        scheduler.buffer.update.assert_called_once_with([success_rollout])
        assert any(
            "Retryable rollout error in group 1" in str(call) and "TimeoutError" in str(call)
            for call in scheduler.logger.warning.call_args_list
        )

    asyncio.run(run())
