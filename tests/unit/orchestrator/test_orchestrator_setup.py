import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from renderers import Qwen3VLRendererConfig

from prime_rl.orchestrator.utils import setup_student_inference_pool
from prime_rl.utils.client import NoOpInferencePool


def test_setup_student_inference_pool_uses_renderer_when_enabled():
    async def run() -> None:
        tokenizer = object()
        renderer_settings = Qwen3VLRendererConfig()
        config = SimpleNamespace(
            training_mode="rl",
            debug=SimpleNamespace(no_inference=False),
            student=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                model=SimpleNamespace(name="student-model"),
            ),
            renderer=renderer_settings,
            pool_size=None,
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
            returned_renderer, returned_pool = await setup_student_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert returned_renderer is renderer
        assert returned_pool is inference_pool
        create_renderer_mock.assert_called_once_with(tokenizer, renderer_settings)
        setup_pool_mock.assert_awaited_once_with(
            config.student.client,
            model_name="student-model",
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=renderer_settings,
            pool_size=None,
        )

    asyncio.run(run())


def test_setup_student_inference_pool_defaults_to_mito():
    """No renderer -> plain MITO chat completions."""

    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            training_mode="rl",
            debug=SimpleNamespace(no_inference=False),
            renderer=None,
            pool_size=None,
            student=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                model=SimpleNamespace(name="student-model"),
            ),
        )
        inference_pool = object()

        with (
            patch("renderers.base.create_renderer") as create_renderer_mock,
            patch(
                "prime_rl.orchestrator.utils.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
        ):
            renderer, returned_pool = await setup_student_inference_pool(
                config=config,
                tokenizer=tokenizer,
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


def test_setup_student_inference_pool_uses_noop_pool_for_no_inference_debug():
    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            training_mode="rl",
            debug=SimpleNamespace(no_inference=True),
            renderer=Qwen3VLRendererConfig(),
            pool_size=None,
            student=SimpleNamespace(
                client=SimpleNamespace(base_url=["http://localhost:8000/v1"]),
                model=SimpleNamespace(name="student-model"),
            ),
        )

        with (
            patch("renderers.base.create_renderer") as create_renderer_mock,
            patch("prime_rl.orchestrator.utils.setup_inference_pool", new=AsyncMock()) as setup_pool_mock,
        ):
            renderer, inference_pool = await setup_student_inference_pool(
                config=config,
                tokenizer=tokenizer,
            )

        assert renderer is None
        assert isinstance(inference_pool, NoOpInferencePool)
        assert inference_pool.model_name == "student-model"
        create_renderer_mock.assert_not_called()
        setup_pool_mock.assert_not_called()

    asyncio.run(run())
