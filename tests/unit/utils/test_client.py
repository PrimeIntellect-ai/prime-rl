import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import verifiers as vf

from prime_rl.configs.shared import ClientConfig
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


class _GeneratePayload(dict):
    @property
    def content(self):
        return json.dumps(self).encode()


class _FakeOpenAI:
    def __init__(self):
        self.base_url = "http://fake-host:8000/v1"
        self.calls = []

    async def get(self, path, *, cast_to=dict):
        return {"data": []}

    async def post(self, path, *, cast_to=dict, body=None, options=None):
        self.calls.append({"path": path, "body": body, "options": options})
        return _GeneratePayload(
            {
                "request_id": "renderer-test",
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [42],
                        "logprobs": {"content": [{"token": "x", "logprob": -0.1}]},
                        "finish_reason": "stop",
                    }
                ],
            }
        )


class _PromptCapturingRenderer:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = self._resolve_config(model_name, config)
        self.prompts = []

    @staticmethod
    def _resolve_config(model_name, config):
        from renderers import AutoRendererConfig, config_from_name
        from renderers.base import MODEL_RENDERER_MAP

        if config is not None and not isinstance(config, AutoRendererConfig):
            return config
        renderer_name = MODEL_RENDERER_MAP.get(model_name, "default")
        resolved = config_from_name(renderer_name)
        if config is None or resolved is None:
            return resolved
        return resolved.model_copy(
            update={
                "preserve_all_thinking": config.preserve_all_thinking,
                "preserve_thinking_between_tool_calls": config.preserve_thinking_between_tool_calls,
            }
        )

    @property
    def last_prompt(self):
        return self.prompts[-1]

    def render(self, messages, *, tools=None, add_generation_prompt=False):
        prompt = self._render_prompt(messages, add_generation_prompt=add_generation_prompt)
        self.prompts.append(prompt)
        return SimpleNamespace(
            token_ids=[ord(ch) for ch in prompt],
            multi_modal_data=None,
        )

    def render_ids(self, messages, *, tools=None, add_generation_prompt=False):
        rendered = self.render(messages, tools=tools, add_generation_prompt=add_generation_prompt)
        return rendered.token_ids

    def get_stop_token_ids(self):
        return [0]

    def parse_response(self, token_ids, **kwargs):
        return SimpleNamespace(content="ok", reasoning_content=None, tool_calls=[])

    def _render_prompt(self, messages, *, add_generation_prompt):
        body = "\n".join(f"{message['role']}:{message.get('content', '')}" for message in messages)
        controls = []
        name = getattr(self.config, "name", "auto")
        if name == "gpt-oss":
            controls.append(f"reasoning_effort={self.config.reasoning_effort}")
        if name in {"qwen3.5", "nemotron-3", "laguna-xs.2"}:
            controls.append("<think>" if self._enable_thinking_default() else "reasoning=off")
        if add_generation_prompt:
            controls.append("assistant:")
        return "\n".join([name, *controls, body])

    def _enable_thinking_default(self):
        enable_thinking = getattr(self.config, "enable_thinking", None)
        if enable_thinking is not None:
            return enable_thinking
        if getattr(self.config, "name", None) == "qwen3.5":
            return self.model_name not in {"Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B"}
        return False


class TestRendererClientChatTemplateKwargs:
    def _render_with_client(
        self,
        *,
        model,
        chat_template_kwargs=None,
        renderer_config=None,
    ):
        from verifiers.clients.renderer_client import RendererClient

        RendererClient._shared_pools.clear()
        fake_openai = _FakeOpenAI()
        client = object.__new__(RendererClient)
        client._renderer = None
        client._pool_size = 1
        client._config = vf.ClientConfig(client_type="renderer", renderer_config=renderer_config)
        client._client = fake_openai
        client.logger = MagicMock()

        renderers = []

        def fake_create_renderer_pool(model_name, config, *, size):
            renderer = _PromptCapturingRenderer(model_name, config)
            renderers.append(renderer)
            return renderer

        extra_body = {"top_k": 20}
        if chat_template_kwargs is not None:
            extra_body["chat_template_kwargs"] = chat_template_kwargs

        with patch("verifiers.clients.renderer_client.create_renderer_pool", side_effect=fake_create_renderer_pool):
            response = asyncio.run(
                client.get_native_response(
                    prompt=[{"role": "user", "content": "solve it"}],
                    model=model,
                    sampling_args={"extra_body": extra_body, "max_tokens": 4},
                    tools=None,
                )
            )

        assert response["content"] == "ok"
        assert len(renderers) == 1
        return renderers[0], fake_openai.calls[0]["body"]

    def test_gpt_oss_reasoning_effort_materializes_in_prompt(self):
        renderer, body = self._render_with_client(
            model="openai/gpt-oss-20b",
            chat_template_kwargs={"reasoning_effort": "high"},
        )

        assert "reasoning_effort=high" in renderer.last_prompt
        assert "chat_template_kwargs" not in body["sampling_params"]
        assert body["sampling_params"]["top_k"] == 20

    @pytest.mark.parametrize(
        "model",
        [
            "Qwen/Qwen3.5-4B",
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            "poolside/Laguna-XS.2",
        ],
    )
    def test_enable_thinking_false_suppresses_reasoning_markup(self, model):
        renderer, _body = self._render_with_client(
            model=model,
            chat_template_kwargs={"enable_thinking": False},
        )

        assert "reasoning=off" in renderer.last_prompt
        assert "<think>" not in renderer.last_prompt

    @pytest.mark.parametrize(
        ("model", "expected_marker"),
        [
            ("Qwen/Qwen3.5-0.8B", "reasoning=off"),
            ("Qwen/Qwen3.5-4B", "<think>"),
            ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "<think>"),
            ("poolside/Laguna-XS.2", "reasoning=off"),
        ],
    )
    def test_omitted_enable_thinking_uses_model_default(self, model, expected_marker):
        renderer, _body = self._render_with_client(model=model)

        assert expected_marker in renderer.last_prompt

    def test_unsupported_renderer_kwarg_raises_clear_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="enable_thinking"):
            self._render_with_client(
                model="openai/gpt-oss-20b",
                chat_template_kwargs={"enable_thinking": False},
            )


def test_setup_clients_does_not_forward_renderer_settings_to_chat_client():
    from renderers import Qwen35RendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        renderer_config=Qwen35RendererConfig(enable_thinking=False),
        renderer_model_name="Qwen/Qwen3.5-4B",
    )

    assert clients[0].client_type == "openai_chat_completions"
    assert clients[0].renderer_config is None
    assert clients[0].renderer_model_name is None
