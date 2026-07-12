import asyncio
from contextlib import suppress
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from renderers import Qwen3VLRendererConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator import component_supervision
from prime_rl.orchestrator.component_supervision import raise_if_component_failed, run_with_component_supervision
from prime_rl.orchestrator.utils import setup_policy_inference_pool


def test_component_failure_is_raised_into_main_loop():
    async def run() -> None:
        async def failed_watcher() -> None:
            raise RuntimeError("indeterminate policy update")

        watcher = asyncio.create_task(failed_watcher(), name="watcher")
        await asyncio.wait({watcher})
        with pytest.raises(RuntimeError, match="indeterminate policy update"):
            raise_if_component_failed([watcher])

    asyncio.run(run())


def test_component_failure_wins_over_queued_output_and_prevents_trainer_send():
    async def run() -> None:
        async def failed_watcher() -> None:
            raise RuntimeError("indeterminate policy update")

        watcher = asyncio.create_task(failed_watcher(), name="watcher")
        await asyncio.wait({watcher})
        operation = AsyncMock(return_value=object())

        with pytest.raises(RuntimeError, match="indeterminate policy update"):
            await run_with_component_supervision(operation, [watcher])

        operation.assert_not_called()

    asyncio.run(run())


def test_component_failure_cancels_in_progress_external_operation():
    async def run() -> None:
        operation_started = asyncio.Event()
        operation_cancelled = asyncio.Event()
        fail_component = asyncio.Event()

        async def operation() -> None:
            operation_started.set()
            try:
                await asyncio.Future()
            finally:
                operation_cancelled.set()

        async def watcher() -> None:
            await fail_component.wait()
            raise RuntimeError("watcher failed during send")

        watcher_task = asyncio.create_task(watcher(), name="watcher")
        supervised = asyncio.create_task(
            run_with_component_supervision(
                operation,
                [watcher_task],
                timeout=1.0,
                timeout_description="training batch send",
            )
        )
        await operation_started.wait()
        fail_component.set()

        with pytest.raises(RuntimeError, match="watcher failed during send"):
            await supervised
        assert operation_cancelled.is_set()

    asyncio.run(run())


def test_supervised_operation_timeout_cancels_stuck_send_and_raises():
    async def run() -> None:
        operation_cancelled = asyncio.Event()

        async def stuck_send() -> None:
            try:
                await asyncio.Future()
            finally:
                operation_cancelled.set()

        with pytest.raises(TimeoutError, match=r"training batch send timed out after 0\.01 seconds"):
            await run_with_component_supervision(
                stuck_send,
                [],
                timeout=0.01,
                timeout_description="training batch send",
            )

        assert operation_cancelled.is_set()

    asyncio.run(run())


def test_timeout_cleanup_is_bounded_when_operation_suppresses_cancellation(monkeypatch: pytest.MonkeyPatch):
    async def run() -> None:
        operation_started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release_orphan = asyncio.Event()

        async def cancellation_suppressing_send() -> None:
            operation_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_orphan.wait()
                raise RuntimeError("late orphan failure")

        monkeypatch.setattr(
            component_supervision,
            "SUPERVISED_OPERATION_CANCEL_GRACE_SECONDS",
            0.01,
            raising=False,
        )
        supervised = asyncio.create_task(
            run_with_component_supervision(
                cancellation_suppressing_send,
                [],
                timeout=0.01,
                timeout_description="training batch send",
            )
        )
        try:
            with pytest.raises(TimeoutError, match=r"training batch send timed out after 0\.01 seconds"):
                await asyncio.wait_for(asyncio.shield(supervised), timeout=0.1)
            assert cancellation_seen.is_set()
            assert len(component_supervision._ORPHANED_OPERATIONS) == 1
        finally:
            release_orphan.set()
            if not supervised.done():
                with suppress(BaseException):
                    await supervised

        for _ in range(10):
            if not component_supervision._ORPHANED_OPERATIONS:
                break
            await asyncio.sleep(0)
        assert not component_supervision._ORPHANED_OPERATIONS

    asyncio.run(run())


def test_caller_cancellation_during_timeout_cleanup_is_not_swallowed(monkeypatch: pytest.MonkeyPatch):
    async def run() -> None:
        cancellation_seen = asyncio.Event()
        release_orphan = asyncio.Event()

        async def cancellation_suppressing_poll() -> None:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_orphan.wait()

        monkeypatch.setattr(component_supervision, "SUPERVISED_OPERATION_CANCEL_GRACE_SECONDS", 0.01)
        supervised = asyncio.create_task(
            run_with_component_supervision(cancellation_suppressing_poll, [], timeout=0.01)
        )
        await cancellation_seen.wait()
        supervised.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(supervised), timeout=0.1)
        assert len(component_supervision._ORPHANED_OPERATIONS) == 1

        release_orphan.set()
        for _ in range(10):
            if not component_supervision._ORPHANED_OPERATIONS:
                break
            await asyncio.sleep(0)
        assert not component_supervision._ORPHANED_OPERATIONS

    asyncio.run(run())


def test_component_failure_precedes_caller_cancellation_during_cleanup(monkeypatch: pytest.MonkeyPatch):
    async def run() -> None:
        operation_started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release_orphan = asyncio.Event()
        fail_watcher = asyncio.Event()

        async def cancellation_suppressing_send() -> None:
            operation_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_seen.set()
                await release_orphan.wait()

        async def failed_watcher() -> None:
            await fail_watcher.wait()
            raise RuntimeError("component failed before cleanup cancellation")

        monkeypatch.setattr(component_supervision, "SUPERVISED_OPERATION_CANCEL_GRACE_SECONDS", 0.01)
        watcher = asyncio.create_task(failed_watcher(), name="watcher")
        supervised = asyncio.create_task(run_with_component_supervision(cancellation_suppressing_send, [watcher]))
        await operation_started.wait()
        fail_watcher.set()
        await cancellation_seen.wait()
        supervised.cancel()

        with pytest.raises(RuntimeError, match="component failed before cleanup cancellation"):
            await asyncio.wait_for(asyncio.shield(supervised), timeout=0.1)
        assert len(component_supervision._ORPHANED_OPERATIONS) == 1

        release_orphan.set()
        for _ in range(10):
            if not component_supervision._ORPHANED_OPERATIONS:
                break
            await asyncio.sleep(0)
        assert not component_supervision._ORPHANED_OPERATIONS

    asyncio.run(run())


def test_component_failure_wins_when_operation_completes_in_same_loop_turn():
    async def run() -> None:
        release = asyncio.Event()

        async def operation() -> str:
            await release.wait()
            return "sent"

        async def watcher() -> None:
            await release.wait()
            raise RuntimeError("simultaneous watcher failure")

        watcher_task = asyncio.create_task(watcher(), name="watcher")
        supervised = asyncio.create_task(run_with_component_supervision(operation, [watcher_task]))
        await asyncio.sleep(0)
        release.set()

        with pytest.raises(RuntimeError, match="simultaneous watcher failure"):
            await supervised

    asyncio.run(run())


def test_setup_policy_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=True,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


def test_setup_policy_inference_pool_keeps_renderer_without_policy_sampling():
    """Frozen-sourced runs (e.g. sft) have no train env sampling from the live
    policy, but training is renderer-only: the renderer is still built and the
    pool is wired with the renderer train client. ``any_policy_sourced`` only
    flips the log line, not the pool setup."""

    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                name="policy-model",
            ),
            renderer=renderer_settings,
            pool_size=None,
            any_policy_sourced=False,
        )
        renderer = object()
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_policy_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.model.client,
            model_name="policy-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


def test_setup_policy_inference_pool_uses_environment_resolved_client(monkeypatch: pytest.MonkeyPatch):
    async def run() -> None:
        monkeypatch.setenv(
            "DYN_RL_TOPOLOGY",
            """{"schema_version":1,"admin_api":"dynamo","base_url":["http://frontend:8000/v1"],"rl_base_url":["http://frontend-rl:8001"],"dynamo_worker_roles":["agg"],"dynamo_gpus_per_worker":1}""",
        )
        config = SimpleNamespace(
            model=SimpleNamespace(client=ClientConfig(), name="policy-model"),
            renderer=Qwen3VLRendererConfig(),
            pool_size=None,
            any_policy_sourced=True,
        )

        with (
            patch("renderers.base.create_renderer", return_value=object()),
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=object()),
            ) as setup_pool,
        ):
            await setup_policy_inference_pool(config=config, tokenizer=object())

        resolved = setup_pool.await_args.args[0]
        assert resolved.admin_api == "dynamo"
        assert resolved.base_url == ["http://frontend:8000/v1"]
        assert config.model.client.admin_api == "vllm"

    asyncio.run(run())
