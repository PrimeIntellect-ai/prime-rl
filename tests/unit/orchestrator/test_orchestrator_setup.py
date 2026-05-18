import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.orchestrator import setup_rollout_inference_pool
from prime_rl.orchestrator.utils import setup_external_rollout_model


def test_setup_rollout_inference_pool_uses_plain_client_for_sft_mode():
    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            training_mode="sft",
            student=SimpleNamespace(renderer="auto", model=SimpleNamespace(name="student-model")),
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


def test_setup_external_rollout_model_sft_uses_teacher_and_checks_student_client():
    from prime_rl.configs.orchestrator import ClientConfig, RolloutModelConfig

    teacher_client = SimpleNamespace(base_url=["https://teacher.example/v1"])
    logger = MagicMock()

    # SFT mode, student client has default base_url (not in model_fields_set) → policy updates disabled
    config = SimpleNamespace(
        training_mode="sft",
        student=RolloutModelConfig(),  # default client — base_url not explicitly set
        teacher=SimpleNamespace(client=teacher_client, model=SimpleNamespace(name="teacher-model")),
    )
    rollout_client, rollout_model, enable_policy_updates = setup_external_rollout_model(config, logger)
    assert rollout_client is teacher_client
    assert rollout_model == "teacher-model"
    assert not enable_policy_updates

    # SFT mode, student client base_url explicitly set → policy updates enabled
    student_model = RolloutModelConfig(client=ClientConfig(base_url=["http://localhost:8000/v1"]))
    config.student = student_model
    rollout_client, rollout_model, enable_policy_updates = setup_external_rollout_model(config, logger)
    assert rollout_client is teacher_client
    assert rollout_model == "teacher-model"
    assert enable_policy_updates


def test_setup_rollout_inference_pool_uses_direct_renderer_client_for_local_vllm():
    async def run() -> None:
        tokenizer = object()
        config = SimpleNamespace(
            training_mode="rl",
            use_renderer=True,
            use_token_client=False,
            student=SimpleNamespace(model=SimpleNamespace(name="student-model")),
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
