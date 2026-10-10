"""Chat completions served through a renderer, against a stub ``/inference/v1/generate`` handler."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from renderers import Qwen3RendererConfig, create_renderer
from renderers.base import load_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse, GenerateResponseChoice
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.sampling_params import RequestOutputKind

from prime_rl.inference.vllm.serving_renderer import RendererChatCompletions

MODEL = "Qwen/Qwen3-0.6B"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    }
]
MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}]


class StubServingTokens:
    """Records the generate request and answers with fixed completion ids."""

    def __init__(self, completion: str, max_model_len: int = 4096):
        self.tokenizer = load_tokenizer(MODEL)
        self.completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)
        model_config = SimpleNamespace(max_model_len=max_model_len, get_diff_sampling_param=lambda: {})
        self.engine_client = SimpleNamespace(model_config=model_config)
        self.requests = []

    async def serve_tokens(self, request, raw_request):
        self.requests.append(request)
        return GenerateResponse(
            model=request.model,
            choices=[GenerateResponseChoice(index=0, token_ids=self.completion_ids, finish_reason="stop")],
            usage=UsageInfo(prompt_tokens=len(request.token_ids), completion_tokens=len(self.completion_ids)),
        )


def serve(completion: str, body: dict, **kwargs):
    stub = StubServingTokens(completion, **kwargs)
    handler = RendererChatCompletions(stub, Qwen3RendererConfig(), MODEL, {"enable_thinking": True})
    request = ChatCompletionRequest.model_validate({"model": "m", "messages": MESSAGES, **body})
    response = asyncio.run(handler.create_chat_completion(request, None))
    return stub, response


def payload(response):
    return json.loads(response.body)


def test_prompt_is_the_rendered_conversation():
    stub, response = serve("<think>\nok\n</think>\n\nSunny.<|im_end|>", {"tools": TOOLS})
    assert response.status_code == 200
    renderer = create_renderer(stub.tokenizer, Qwen3RendererConfig())
    (request,) = stub.requests
    assert request.token_ids == renderer.render_ids(MESSAGES, tools=TOOLS, add_generation_prompt=True)
    params = request.sampling_params
    assert set(renderer.get_stop_token_ids()) <= set(params.stop_token_ids)
    assert params.skip_special_tokens is False
    assert params.output_kind == RequestOutputKind.FINAL_ONLY
    assert params.max_tokens == 4096 - len(request.token_ids)


def test_tool_calls_come_back_as_openai_tool_calls():
    completion = '<think>\nCall the tool.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>'
    _, response = serve(completion, {"tools": TOOLS})
    (choice,) = payload(response)["choices"]
    message = choice["message"]
    assert choice["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["reasoning"] == message["reasoning_content"] == "Call the tool."
    (call,) = message["tool_calls"]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    assert call["id"].startswith("call_")


@pytest.mark.parametrize(
    ("completion", "content", "reasoning"),
    [
        ("<think>\n\n</think>\n\nSunny.<|im_end|>", "Sunny.", None),
        ("<think>\nok\n</think>\n\n<|im_end|>", "", "ok"),
        ("<think>\n\n</think>\n\n<|im_end|>", "", None),
    ],
)
def test_empty_content_and_reasoning(completion, content, reasoning):
    _, response = serve(completion, {})
    (choice,) = payload(response)["choices"]
    assert (choice["message"]["content"], choice["message"]["reasoning"]) == (content, reasoning)
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


def test_request_template_kwargs_override_defaults_and_unknown_ones_are_ignored():
    stub, _ = serve("Sunny.<|im_end|>", {"chat_template_kwargs": {"enable_thinking": False, "unknown": 1}})
    renderer = create_renderer(stub.tokenizer, Qwen3RendererConfig(enable_thinking=False))
    assert stub.requests[0].token_ids == renderer.render_ids(MESSAGES, add_generation_prompt=True)


def test_token_ids_are_returned_on_request():
    stub, response = serve("<think>\n\n</think>\n\nSunny.<|im_end|>", {"return_token_ids": True})
    body = payload(response)
    assert body["prompt_token_ids"] == stub.requests[0].token_ids
    assert body["choices"][0]["token_ids"] == stub.completion_ids


def test_stream_sends_the_completion_as_one_chunk():
    _, response = serve(
        "<think>\nok\n</think>\n\nSunny.<|im_end|>", {"stream": True, "stream_options": {"include_usage": True}}
    )

    async def collect():
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())
    assert chunks[-1] == "data: [DONE]\n\n"
    events = [json.loads(chunk.removeprefix("data: ")) for chunk in chunks[:-1]]
    assert events[0]["choices"][0]["delta"]["content"] == "Sunny."
    assert events[1]["choices"][0]["finish_reason"] == "stop"
    assert events[2]["usage"]["completion_tokens"] > 0


def test_overlong_prompt_is_a_bad_request():
    stub, response = serve("Sunny.<|im_end|>", {}, max_model_len=8)
    assert response.status_code == 400
    assert not stub.requests


def test_inference_config_passes_the_renderer_to_the_server():
    from prime_rl.configs.inference import InferenceConfig

    namespace = InferenceConfig.model_validate({"renderer": {"name": "qwen3", "enable_thinking": False}}).to_namespace()
    assert namespace.renderer == Qwen3RendererConfig(enable_thinking=False)
    assert not hasattr(InferenceConfig().to_namespace(), "renderer")
