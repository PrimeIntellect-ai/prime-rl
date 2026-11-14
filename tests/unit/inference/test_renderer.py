"""Chat prompts and completions through a renderer, without an engine."""

from __future__ import annotations

import json

import pytest
from renderers import PrimeQwen3RendererConfig, Qwen3RendererConfig, create_renderer
from renderers.base import load_tokenizer
from vllm.config import ModelConfig
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.renderers.params import ChatParams

from prime_rl.inference.vllm.renderer import RendererBackedRenderer, RendererParser, Renderers

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


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL)


@pytest.fixture(scope="module")
def model_config():
    return ModelConfig(model=MODEL)


def classes(model_config, renderer_config=Qwen3RendererConfig()):
    renderers = Renderers(renderer_config, MODEL)
    renderer = object.__new__(type("Renderer", (RendererBackedRenderer,), {"renderers": renderers}))
    renderer.model_config = model_config
    return renderer, type("Parser", (RendererParser,), {"renderers": renderers})


def render(renderer, messages, **kwargs):
    params = ChatParams(chat_template_kwargs={"add_generation_prompt": True, **kwargs})
    return renderer.render_messages(messages, params)[1]["prompt_token_ids"]


def test_prompt_is_the_rendered_conversation(tokenizer, model_config):
    renderer, _ = classes(model_config)
    expected = create_renderer(tokenizer, Qwen3RendererConfig()).render_ids(
        MESSAGES, tools=TOOLS, add_generation_prompt=True
    )
    assert render(renderer, MESSAGES, tools=TOOLS) == expected


def test_template_kwargs_select_the_renderer_and_unknown_ones_are_ignored(tokenizer, model_config):
    renderer, _ = classes(model_config)
    expected = create_renderer(tokenizer, Qwen3RendererConfig(enable_thinking=False)).render_ids(
        MESSAGES, add_generation_prompt=True
    )
    assert render(renderer, MESSAGES, enable_thinking=False, unknown=1) == expected


def test_messages_are_normalized_like_the_chat_template_route(tokenizer, model_config):
    renderer, _ = classes(model_config, PrimeQwen3RendererConfig())
    parts = [{"type": "text", "text": "Weather"}, {"type": "text", "text": "in Paris?"}]
    history = [
        {"role": "assistant", "content": "Sunny.", "reasoning_content": "Look it up."},
        {"role": "user", "content": "And tomorrow?"},
    ]
    expected = create_renderer(tokenizer, PrimeQwen3RendererConfig()).render_ids(
        [{"role": "user", "content": "Weather\nin Paris?"}, *history], add_generation_prompt=True
    )
    # Request validation renames reasoning_content to reasoning, in place.
    messages = [{"role": "user", "content": parts}, *(dict(message) for message in history)]
    request = ChatCompletionRequest.model_validate({"model": "m", "messages": messages})
    assert render(renderer, request.messages) == expected


def test_parse_returns_reasoning_content_and_tool_calls(tokenizer, model_config):
    _, parser_cls = classes(model_config)
    completion = '<think>\nCall the tool.\n</think>\n\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>'
    request = ChatCompletionRequest.model_validate({"model": "m", "messages": MESSAGES, "tools": TOOLS})
    parser = parser_cls(tokenizer, request.tools)
    reasoning, content, calls = parser.parse("", request, model_output_token_ids=tokenizer.encode(completion))
    assert (reasoning, content) == ("Call the tool.", None)
    (call,) = calls
    assert (call.name, json.loads(call.arguments)) == ("get_weather", {"city": "Paris"})


def test_parse_delta_sends_the_completion_when_it_finishes(tokenizer, model_config):
    _, parser_cls = classes(model_config)
    request = ChatCompletionRequest.model_validate({"model": "m", "messages": MESSAGES})
    parser = parser_cls(tokenizer)
    ids = tokenizer.encode("<think>\nok\n</think>\n\nSunny.<|im_end|>")
    assert parser.parse_delta("", ids[:3], request, finished=False) is None
    delta = parser.parse_delta("", ids[3:], request, finished=True)
    assert (delta.reasoning, delta.content, delta.tool_calls) == ("ok", "Sunny.", [])


def test_inference_config_passes_the_renderer_to_the_server():
    from prime_rl.configs.inference import InferenceConfig

    namespace = InferenceConfig.model_validate({"renderer": {"name": "qwen3", "enable_thinking": False}}).to_namespace()
    assert namespace.renderer == Qwen3RendererConfig(enable_thinking=False)
    assert not hasattr(InferenceConfig().to_namespace(), "renderer")
