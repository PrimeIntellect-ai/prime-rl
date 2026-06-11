import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import verifiers as vf

from prime_rl.configs.multi_agent import FixedMemberTargetConfig
from prime_rl.configs.shared import ClientConfig
from prime_rl.orchestrator.member_generation import _fixed_client
from prime_rl.utils.client import _is_retryable_lora_error, load_lora_adapter, setup_clients


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert [client.client_type for client in clients] == ["renderer", "renderer"]
    assert [client.renderer_config for client in clients] == [renderer_settings, renderer_settings]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.api_base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.extra_headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].extra_headers["X-Test"] == "test"
    assert clients[0].extra_headers_from_state == {"X-Session-ID": "session_id"}


def test_setup_clients_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert clients[0].renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_clients_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(client_config)

    assert clients == [
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            api_key_var="PRIME_API_KEY",
            api_base_url="http://worker-a:8000/v1",
            timeout=client_config.timeout,
            connect_timeout=client_config.connect_timeout,
            max_connections=8192,
            max_keepalive_connections=8192,
            max_retries=10,
            extra_headers={},
            extra_headers_from_state={},
        )
    ]


def test_build_client_contract_matches_both_call_sites():
    """Lock the build_client factory contract: the pool path (setup_clients)
    and the fixed-member path (_fixed_client) must each map their policy
    config onto the full vf.ClientConfig exactly as documented."""
    pool_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        timeout=900,
        connect_timeout=12.0,
        max_retries=3,
        max_connections=64,
        max_keepalive_connections=32,
        headers={"X-Test": "test"},
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    assert setup_clients(pool_config) == [
        vf.ClientConfig(
            client_idx=0,
            client_type="openai_chat_completions",
            api_base_url="http://worker-a:8000/v1",
            api_key_var="PRIME_API_KEY",
            timeout=900,
            connect_timeout=12.0,
            max_connections=64,
            max_keepalive_connections=32,
            max_retries=3,
            extra_headers={"X-Test": "test"},
            extra_headers_from_state={"X-Session-ID": "session_id"},
        )
    ]

    target = FixedMemberTargetConfig(
        members=["judge"],
        model="judge-model",
        base_url=["http://judge:8000/v1"],
        api_key_var="JUDGE_API_KEY",
        request_mode="token",
        headers={"X-Judge": "1"},
        timeout=600.0,
        connect_timeout=5.0,
        max_retries=2,
        max_connections=16,
        max_keepalive_connections=8,
    )

    assert _fixed_client("judge", target, member_id="judge", group_id="group-0") == vf.ClientConfig(
        client_idx=0,
        client_type="openai_chat_completions_token",
        api_base_url="http://judge:8000/v1",
        api_key_var="JUDGE_API_KEY",
        timeout=600.0,
        connect_timeout=5.0,
        max_connections=16,
        max_keepalive_connections=8,
        max_retries=2,
        extra_headers={"X-Judge": "1"},
        extra_headers_from_state={},
    )


def test_as_train_client_derives_renderer_twin_from_chat_eval_client():
    from renderers import Qwen3VLRendererConfig

    from prime_rl.utils.client import StaticInferencePool

    renderer_settings = Qwen3VLRendererConfig()
    pool = StaticInferencePool(
        ClientConfig(
            base_url=["http://worker-a:8000/v1"],
            api_key_var="PRIME_API_KEY",
            headers={"X-Test": "test"},
        ),
        model_name="Qwen/Qwen3-VL-4B-Instruct",
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=renderer_settings,
        pool_size=4,
    )

    eval_client = pool.eval_clients[0]
    assert eval_client.client_type == "openai_chat_completions"

    learner = pool.as_train_client(eval_client)
    assert learner.client_type == "renderer"
    assert learner.renderer_config == renderer_settings
    assert learner.renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"
    assert learner.renderer_pool_size == 4
    # Same server, headers, and timeouts — only the client type changed
    assert learner.api_base_url == eval_client.api_base_url
    assert learner.extra_headers == eval_client.extra_headers
    assert learner.timeout == eval_client.timeout
    # Train clients already speak the train type — identity
    train_client = pool.train_clients[0]
    assert pool.as_train_client(train_client) is train_client


def test_as_train_client_is_identity_for_chat_trained_pool():
    from prime_rl.utils.client import StaticInferencePool

    pool = StaticInferencePool(
        ClientConfig(base_url=["http://worker-a:8000/v1"], api_key_var="PRIME_API_KEY"),
        model_name="some-model",
    )

    eval_client = pool.eval_clients[0]
    assert pool.as_train_client(eval_client) is eval_client
