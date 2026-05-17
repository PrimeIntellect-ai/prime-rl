import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import setup_rollout_inference_pool
from prime_rl.orchestrator.utils import setup_external_rollout_model


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
                "prime_rl.orchestrator.orchestrator.setup_inference_pool",
                new=AsyncMock(return_value=inference_pool),
            ) as setup_pool_mock,
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
        setup_pool_mock.assert_awaited_once_with(
            rollout_client_config,
            model_name="teacher-model",
            train_client_type="openai_chat_completions",
            eval_client_type="openai_chat_completions",
        )

    asyncio.run(run())


def test_setup_external_rollout_model_uses_explicit_weight_update_opt_in():
    teacher_client = SimpleNamespace(base_url=["https://teacher.example/v1"])
    student_client = SimpleNamespace(base_url=["http://localhost:8000/v1"])
    config = SimpleNamespace(
        client=student_client,
        model=SimpleNamespace(name="student-model"),
        update_student_inference_weights=None,
        teacher_rollout_model=SimpleNamespace(
            client=teacher_client,
            model=SimpleNamespace(name="teacher-model"),
        ),
    )
    logger = MagicMock()

    rollout_client, rollout_model, enable_policy_updates = setup_external_rollout_model(config, logger)
    assert rollout_client is teacher_client
    assert rollout_model == "teacher-model"
    assert not enable_policy_updates

    config.update_student_inference_weights = True
    rollout_client, rollout_model, enable_policy_updates = setup_external_rollout_model(config, logger)
    assert rollout_client is teacher_client
    assert rollout_model == "teacher-model"
    assert enable_policy_updates


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
