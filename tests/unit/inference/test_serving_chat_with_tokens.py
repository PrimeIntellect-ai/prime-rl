import asyncio
from types import MethodType, SimpleNamespace

import pytest
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.exceptions import VLLMValidationError

from prime_rl.inference.vllm.serving_chat_with_tokens import (
    ChatCompletionRequestWithTokens,
    OpenAIServingChatWithTokens,
)


def _handler(max_model_len: int) -> OpenAIServingChatWithTokens:
    handler = object.__new__(OpenAIServingChatWithTokens)
    handler.model_config = SimpleNamespace(max_model_len=max_model_len)

    def _extract_prompt_len(self, prompt):
        return len(prompt["prompt_token_ids"])

    handler._extract_prompt_len = MethodType(_extract_prompt_len, handler)
    return handler


def test_validate_prompt_has_generation_room_allows_one_token_margin():
    handler = _handler(max_model_len=4)

    assert handler._validate_prompt_has_generation_room({"prompt_token_ids": [1, 2, 3]}) == 3


def test_validate_prompt_has_generation_room_rejects_full_context():
    handler = _handler(max_model_len=4)

    with pytest.raises(VLLMValidationError) as exc_info:
        handler._validate_prompt_has_generation_room({"prompt_token_ids": [1, 2, 3, 4]})

    message = str(exc_info.value)
    assert "maximum context length is 4 tokens" in message
    assert "4 input tokens" in message


def test_render_chat_request_returns_context_error_before_max_tokens_underflows(monkeypatch):
    async def fake_render_chat_request(self, request):
        return "conversation", [{"prompt_token_ids": [1, 2, 3, 4]}]

    monkeypatch.setattr(OpenAIServingChat, "render_chat_request", fake_render_chat_request)

    handler = _handler(max_model_len=4)

    def create_error_response(self, error):
        return {"error": str(error)}

    handler.create_error_response = MethodType(create_error_response, handler)

    result = asyncio.run(handler.render_chat_request(object()))

    assert "maximum context length is 4 tokens" in result["error"]
    assert "4 input tokens" in result["error"]


def test_render_chat_request_defers_token_endpoint_validation_until_tokens_are_installed(monkeypatch):
    async def fake_render_chat_request(self, request):
        return "conversation", [{"prompt_token_ids": [1, 2, 3, 4]}]

    monkeypatch.setattr(OpenAIServingChat, "render_chat_request", fake_render_chat_request)

    handler = _handler(max_model_len=4)
    request = object.__new__(ChatCompletionRequestWithTokens)

    result = asyncio.run(handler.render_chat_request(request))

    assert result == ("conversation", [{"prompt_token_ids": [1, 2, 3, 4]}])
