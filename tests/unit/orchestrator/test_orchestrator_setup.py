import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import setup_rollout_inference_pool


def test_setup_rollout_inference_pool_uses_plain_client_for_external_teacher_rollout():
    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            teacher_rollout_model=SimpleNamespace(),
            model=SimpleNamespace(renderer="auto", name="student-model"),
        )
        rollout_client_config = SimpleNamespace(base_url=["https://api.pinference.ai/api/v1"])
        logger = MagicMock()
        inference_pool = object()

        with (
            patch(
                "prime_rl.orchestrator.orchestrator.setup_inference_pool", new=AsyncMock(return_value=inference_pool)
            ),
            patch("prime_rl.orchestrator.orchestrator.create_renderer") as create_renderer_mock,
        ):
            renderer, returned_pool = await setup_rollout_inference_pool(
                config=config,
                rollout_client_config=rollout_client_config,
                rollout_model_name="teacher-model",
                tokenizer=tokenizer,
                logger=logger,
            )

        assert renderer is None
        assert returned_pool is inference_pool
        create_renderer_mock.assert_not_called()

    asyncio.run(run())


def test_setup_rollout_inference_pool_uses_direct_renderer_client_for_local_vllm():
    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            teacher_rollout_model=None,
            use_renderer=True,
            use_token_client=False,
            model=SimpleNamespace(name="student-model"),
            renderer=SimpleNamespace(
                name="qwen3_vl",
                tool_parser=None,
                reasoning_parser=None,
                pool_size=None,
                preserve_all_thinking=False,
                preserve_thinking_between_tool_calls=False,
            ),
        )
        rollout_client_config = SimpleNamespace(base_url=["http://localhost:8000/v1"])
        logger = MagicMock()
        renderer = object()
        inference_pool = object()

        with (
            patch("prime_rl.orchestrator.orchestrator.create_renderer", return_value=renderer) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            returned_renderer, returned_pool = await setup_rollout_inference_pool(
                config=config,
                rollout_client_config=rollout_client_config,
                rollout_model_name="student-model",
                tokenizer=tokenizer,
                logger=logger,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(
            tokenizer,
            renderer="qwen3_vl",
            tool_parser=None,
            reasoning_parser=None,
            preserve_all_thinking=False,
            preserve_thinking_between_tool_calls=False,
        )
        setup_pool_mock.assert_awaited_once_with(
            rollout_client_config,
            model_name="student-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_name="qwen3_vl",
            tool_parser=None,
            reasoning_parser=None,
            renderer_pool_size=None,
            preserve_all_thinking=False,
            preserve_thinking_between_tool_calls=False,
        )

    asyncio.run(run())


def test_setup_rollout_inference_pool_forwards_preserve_thinking_flags():
    """``RendererConfig.preserve_*_thinking`` must reach both
    ``create_renderer`` (training tokenization) and ``setup_inference_pool``
    (inference render path) — train/infer mismatch otherwise."""

    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            teacher_rollout_model=None,
            use_renderer=True,
            use_token_client=False,
            model=SimpleNamespace(name="student-model"),
            renderer=SimpleNamespace(
                name="glm5",
                tool_parser=None,
                reasoning_parser=None,
                pool_size=4,
                preserve_all_thinking=True,
                preserve_thinking_between_tool_calls=True,
            ),
        )
        rollout_client_config = SimpleNamespace(base_url=["http://localhost:8000/v1"])
        logger = MagicMock()

        with (
            patch(
                "prime_rl.orchestrator.orchestrator.create_renderer",
                return_value=object(),
            ) as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.setup_inference_pool",
                new=AsyncMock(return_value=object()),
            ) as setup_pool_mock,
        ):
            await setup_rollout_inference_pool(
                config=config,
                rollout_client_config=rollout_client_config,
                rollout_model_name="student-model",
                tokenizer=tokenizer,
                logger=logger,
            )

        _, kwargs = create_renderer_mock.call_args
        assert kwargs["preserve_all_thinking"] is True
        assert kwargs["preserve_thinking_between_tool_calls"] is True

        _, pool_kwargs = setup_pool_mock.call_args
        assert pool_kwargs["preserve_all_thinking"] is True
        assert pool_kwargs["preserve_thinking_between_tool_calls"] is True
        assert pool_kwargs["renderer_pool_size"] == 4

    asyncio.run(run())
