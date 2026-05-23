import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import setup_student_inference_pool


def test_setup_student_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            training_mode="rl",
            use_renderer=True,
            student=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                model=SimpleNamespace(name="student-model"),
            ),
            renderer=SimpleNamespace(
                name="qwen3",
                tool_parser=None,
                reasoning_parser=None,
                pool_size=None,
                preserve_all_thinking=False,
                preserve_thinking_between_tool_calls=False,
            ),
            train=SimpleNamespace(
                sampling=SimpleNamespace(extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            ),
        )
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
            returned_renderer, returned_pool = await setup_student_inference_pool(
                config=config,
                tokenizer=tokenizer,
                logger=logger,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(
            tokenizer,
            renderer="qwen3",
            tool_parser=None,
            reasoning_parser=None,
            chat_template_kwargs={"enable_thinking": False},
            preserve_all_thinking=False,
            preserve_thinking_between_tool_calls=False,
        )
        setup_pool_mock.assert_awaited_once_with(
            config.student.client,
            model_name="student-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_name="qwen3",
            tool_parser=None,
            reasoning_parser=None,
            renderer_pool_size=None,
            preserve_all_thinking=False,
            preserve_thinking_between_tool_calls=False,
        )

    asyncio.run(run())


def test_setup_student_inference_pool_defaults_to_mito():
    """No renderer -> plain MITO chat completions."""

    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            training_mode="rl",
            use_renderer=False,
            student=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                model=SimpleNamespace(name="student-model"),
            ),
        )
        logger = MagicMock()
        inference_pool = object()

        with (
            patch("prime_rl.orchestrator.orchestrator.create_renderer") as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.orchestrator.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            renderer, returned_pool = await setup_student_inference_pool(
                config=config,
                tokenizer=tokenizer,
                logger=logger,
            )

        assert renderer is None
        assert returned_pool is inference_pool
        create_renderer_mock.assert_not_called()
        setup_pool_mock.assert_awaited_once_with(
            config.student.client,
            model_name="student-model",
            train_client_type="openai_chat_completions",
            eval_client_type="openai_chat_completions",
        )

    asyncio.run(run())
