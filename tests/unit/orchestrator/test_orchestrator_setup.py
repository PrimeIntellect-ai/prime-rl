import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from renderers import Qwen3VLRendererConfig

from prime_rl.configs.orchestrator import FileSystemWeightBroadcastConfig, NCCLWeightBroadcastConfig
from prime_rl.orchestrator.utils import setup_policy_inference_pool


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
            weight_broadcast=SimpleNamespace(type="filesystem"),
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
            expected_inference_world_size=None,
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
            weight_broadcast=SimpleNamespace(inference_world_size=8),
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
            expected_inference_world_size=8,
        )

    asyncio.run(run())


def test_filesystem_broadcast_accepts_expected_dynamo_world_size():
    config = FileSystemWeightBroadcastConfig(inference_world_size=8)
    assert config.inference_world_size == 8


@pytest.mark.parametrize(
    "weight_broadcast",
    [FileSystemWeightBroadcastConfig(), NCCLWeightBroadcastConfig()],
    ids=["filesystem-missing", "nccl-default"],
)
def test_setup_policy_inference_pool_requires_explicit_world_size_for_dynamo(weight_broadcast):
    async def run() -> None:
        config = SimpleNamespace(
            model=SimpleNamespace(
                client=SimpleNamespace(
                    base_url=["http://frontend:8000/v1"],
                    dynamo_discovery_url="http://frontend:8001",
                    is_dynamo=True,
                ),
                name="policy-model",
            ),
            renderer=Qwen3VLRendererConfig(),
            pool_size=None,
            any_policy_sourced=True,
            weight_broadcast=weight_broadcast,
        )
        with (
            patch("renderers.base.create_renderer", return_value=object()),
            patch("prime_rl.orchestrator.utils.setup_inference_pool", new=AsyncMock()),
            pytest.raises(ValueError, match="inference_world_size"),
        ):
            await setup_policy_inference_pool(config=config, tokenizer=object())

    asyncio.run(run())
