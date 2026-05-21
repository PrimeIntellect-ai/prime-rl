import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import verifiers as vf

from prime_rl.orchestrator.scheduler import (
    GroupState,
    InflightRequest,
    Scheduler,
    is_reschedulable_rollout_error,
)
from prime_rl.utils.async_utils import safe_cancel


def make_error(
    error: str,
    error_chain_str: str | None = None,
    error_chain_repr: str | None = None,
    is_retryable: bool | None = None,
) -> dict:
    error_info = {
        "error": error,
        "error_chain_str": error_chain_str or error,
        "error_chain_repr": error_chain_repr or f"{error}()",
    }
    if is_retryable is not None:
        error_info["is_retryable"] = is_retryable
    return error_info


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
    scheduler.batch_size = 1
    scheduler.token_batch_size = None
    scheduler.rollouts_per_example = 1
    scheduler.max_inflight_rollouts = 1
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.empty_rollouts_by_env = defaultdict(int)
    scheduler.errored_rollouts_by_env = defaultdict(int)
    scheduler.total_rollouts_by_env = defaultdict(int)
    scheduler.dropped_groups_by_env = defaultdict(int)
    scheduler.json_logging = False
    scheduler.last_batch_generation_time = 0.0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.rate_limiter = None
    return scheduler


def test_reschedulable_serialized_errors_match_verifiers_retry_semantics():
    retryable_errors = [
        make_error("InfraError", is_retryable=True),
        make_error("SandboxError", "SandboxError -> TimeoutError", is_retryable=True),
        make_error("SandboxCreationError", "SandboxCreationError", is_retryable=True),
        make_error("TunnelError", is_retryable=True),
        make_error("BrowserSandboxError", is_retryable=True),
        make_error("InvalidModelResponseError", is_retryable=True),
        make_error("EmptyModelResponseError", is_retryable=True),
        make_error("SubLLMEmptyModelResponseError", is_retryable=True),
    ]

    for error in retryable_errors:
        assert is_reschedulable_rollout_error(error), error


def test_serialized_error_names_without_retryable_flag_are_terminal():
    assert not is_reschedulable_rollout_error(make_error("SandboxError"))
    assert not is_reschedulable_rollout_error(make_error("EmptyModelResponseError"))


def test_plain_model_error_and_unknown_serialized_errors_are_terminal():
    no_workers = make_error(
        "ModelError",
        "ModelError -> InternalServerError",
        "ModelError() -> InternalServerError('No available workers (all circuits open or unhealthy)')",
        is_retryable=False,
    )
    unknown = make_error("MysteryRolloutError")

    assert not is_reschedulable_rollout_error(no_workers)
    assert not is_reschedulable_rollout_error(unknown)


@pytest.mark.parametrize(
    "error",
    [
        make_error(
            "ModelError",
            "ModelError -> InternalServerError",
            "ModelError() -> InternalServerError('No available workers (all circuits open or unhealthy)')",
            is_retryable=False,
        ),
        make_error("MysteryRolloutError"),
    ],
)
def test_generate_batch_aborts_non_reschedulable_serialized_errors(error: dict):
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.maybe_update_policy = AsyncMock()
        scheduler.update_policy_loop = AsyncMock()
        scheduler._fill_inflight_requests = AsyncMock()
        scheduler.buffer = MagicMock()
        scheduler.train_envs = SimpleNamespace(get=MagicMock(return_value=SimpleNamespace(requires_group_scoring=True)))
        scheduler.groups = {
            0: GroupState(
                example={"env_name": "swe", "example_id": "ex-1"},
                rollouts_to_schedule=0,
            )
        }

        task = asyncio.create_task(asyncio.sleep(0, result={"error": error, "trajectory": []}))
        await asyncio.sleep(0)
        scheduler.inflight_requests = {
            task: InflightRequest(
                off_policy_steps=0,
                client_config=SimpleNamespace(api_base_url="http://test", extra_headers={}),
                env_name="swe",
                group_id=0,
            )
        }

        with (
            patch(
                "prime_rl.orchestrator.scheduler.ProgressTracker",
                return_value=SimpleNamespace(update=MagicMock(), close=MagicMock()),
            ),
            pytest.raises(RuntimeError, match="not rescheduling replacement rollouts"),
        ):
            await scheduler.generate_batch(step=1)

        assert scheduler.groups == {}
        assert scheduler.errored_rollouts_by_env["swe"] == 1
        scheduler._fill_inflight_requests.assert_awaited_once()
        scheduler.buffer.update.assert_not_called()

    asyncio.run(run())


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

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
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

        scheduler.student_inference = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler.rollout_inference = scheduler.student_inference
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


def test_lora_policy_update_in_sft_keeps_teacher_model_name():
    """In sft mode, train_pool is the teacher. LoRA updates the student inference
    pool but must not change scheduler.model_name (which is what gets sent to the
    teacher endpoint on each rollout request)."""

    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.model_name = "teacher-model"
        scheduler.lora_name = "student-lora"

        student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )
        teacher_inference = SimpleNamespace()
        scheduler.student_inference = student_inference
        scheduler.rollout_inference = teacher_inference  # sft: train_pool != student_inference
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            await scheduler.maybe_update_policy()

        student_inference.update_weights.assert_awaited_once()
        student_inference.update_model_name.assert_called_once_with("student-lora")
        assert scheduler.model_name == "teacher-model"

    asyncio.run(run())


def test_lora_policy_update_in_rl_updates_model_name():
    """In rl/opd mode, train_pool is the student. LoRA updates redirect rollout
    requests to the new LoRA name."""

    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.model_name = "student-model"
        scheduler.lora_name = "student-lora"

        student_inference = SimpleNamespace(
            update_weights=AsyncMock(),
            update_model_name=MagicMock(),
        )
        scheduler.student_inference = student_inference
        scheduler.rollout_inference = student_inference  # rl/opd: same pool
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            await scheduler.maybe_update_policy()

        student_inference.update_weights.assert_awaited_once()
        student_inference.update_model_name.assert_called_once_with("student-lora")
        assert scheduler.model_name == "student-lora"

    asyncio.run(run())


def test_schedule_rollout_uses_train_pool():
    """schedule_rollout dispatches to train_pool's clients with train_pool's model name."""

    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.model_name = "teacher-model"
        teacher_client = vf.ClientConfig(api_base_url="http://teacher.example/v1")
        env = SimpleNamespace(
            requires_group_scoring=False,
            run_rollout=AsyncMock(return_value=[]),
        )
        scheduler.rollout_inference = SimpleNamespace(train_clients=[teacher_client])
        scheduler.train_envs = SimpleNamespace(get=MagicMock(return_value=env))
        scheduler.groups = {
            0: GroupState(
                example={"env_name": "math", "example_id": "ex-1"},
                rollouts_to_schedule=1,
            )
        }

        await scheduler.schedule_rollout(group_id=0)
        await asyncio.gather(*scheduler.inflight_requests)

        env.run_rollout.assert_awaited_once_with(
            client=teacher_client,
            example={"env_name": "math", "example_id": "ex-1"},
            model_name="teacher-model",
            cache_salt="7",
        )
        assert scheduler.groups[0].pinned_client is teacher_client

    asyncio.run(run())
