"""Chat prompts and completions through a renderer instead of a chat template and parsers.

With ``[inference] renderer`` set, ``RendererBackedRenderer`` renders the messages of every
chat-style route (chat completions, responses, anthropic messages, tokenize) to prompt token
ids, and ``RendererParser`` parses completion token ids into content, reasoning and tool calls.
A served prompt has exactly the tokens the renderer trains on. A streamed completion arrives
as one delta when it finishes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from renderers import create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer, template_field_names
from vllm.entrypoints.chat_utils import make_tool_call_id, parse_chat_messages
from vllm.entrypoints.generate.base.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall, FunctionCall
from vllm.exceptions import VLLMValidationError
from vllm.parser.abstract_parser import Parser
from vllm.parser.parser_manager import ParserManager
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.tool_parsers import ToolParser
from vllm.utils.async_utils import make_async

# The non-streaming parse gets no prompt ids, and renderers need them to see a prefilled
# reasoning block. A request renders and parses in the same task.
_prompt_ids: ContextVar[list[int] | None] = ContextVar("renderer_prompt_ids", default=None)


def _chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """A chat completions tool from a chat completions or responses tool."""
    if "function" in tool:
        return tool
    return {
        "type": "function",
        "function": {key: tool[key] for key in ("name", "description", "parameters") if key in tool},
    }


class Renderers(threading.local):
    """Renderers by template kwargs. A renderer wraps a tokenizer that one thread must own,
    so each thread builds its own."""

    def __init__(self, config: Any, tokenizer_name: str):
        self.config = config
        self.tokenizer_name = tokenizer_name
        self.fields = template_field_names(config)
        self.tokenizer = None
        self.cache: dict[str, Any] = {}

    def get(self, chat_template_kwargs: dict[str, Any]):
        kwargs = {key: value for key, value in chat_template_kwargs.items() if key in self.fields}
        key = json.dumps(kwargs, sort_keys=True)
        if key not in self.cache:
            if self.tokenizer is None:
                self.tokenizer = load_tokenizer(self.tokenizer_name)
            self.cache[key] = create_renderer(self.tokenizer, self.config, chat_template_kwargs=kwargs)
        return self.cache[key]


class RendererBackedRenderer(HfRenderer):
    renderers: Renderers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._render_messages_async = make_async(self.render_messages, executor=self._executor)

    def render_messages(self, messages, params):
        conversation, mm_data, _ = parse_chat_messages(messages, self.model_config, content_format="string")
        if mm_data is not None:
            raise VLLMValidationError("A renderer serves text-only messages.")
        kwargs = params.chat_template_kwargs
        renderer = self.renderers.get(kwargs)
        try:
            prompt_ids = renderer.render_ids(
                conversation,
                tools=kwargs.get("tools"),
                add_generation_prompt=kwargs.get("add_generation_prompt", True),
            )
        except (ValueError, TypeError, KeyError) as error:
            raise VLLMValidationError(str(error)) from error
        return conversation, parse_dec_only_prompt(prompt_ids)

    async def render_messages_async(self, messages, params):
        conversation, prompt = await self._render_messages_async(messages, params)
        _prompt_ids.set(prompt["prompt_token_ids"])
        return conversation, prompt


class RendererParser(Parser):
    renderers: Renderers
    # The chat routes return tool calls only when the parser names a tool parser class.
    tool_parser_cls = ToolParser

    def __init__(self, tokenizer, tools=None, *args, chat_template_kwargs: dict[str, Any] | None = None, **kwargs):
        self.model_tokenizer = tokenizer
        self._reasoning_parser = None
        self._tool_parser = None
        self.tools = [_chat_tool(tool.model_dump(exclude_none=True)) for tool in tools] if tools else None
        self.renderer = self.renderers.get(chat_template_kwargs or {})
        self.token_ids: list[int] = []

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return False

    def parse(self, model_output, request, enable_auto_tools=False, model_output_token_ids: Sequence[int] = ()):
        return self._parse(list(model_output_token_ids), _prompt_ids.get())

    def parse_delta(self, delta_text, delta_token_ids, request, prompt_token_ids=None, *, finished):
        self.token_ids.extend(delta_token_ids)
        if not finished:
            return None
        reasoning, content, calls = self._parse(self.token_ids, prompt_token_ids)
        tool_calls = [
            DeltaToolCall(
                index=index,
                id=make_tool_call_id(),
                type="function",
                function=DeltaFunctionCall(name=call.name, arguments=call.arguments),
            )
            for index, call in enumerate(calls)
        ]
        return DeltaMessage(reasoning=reasoning, content=content, tool_calls=tool_calls)

    def _parse(self, token_ids: list[int], prompt_ids: list[int] | None):
        parsed = self.renderer.parse_response(token_ids, tools=self.tools, prompt_ids=prompt_ids)
        calls = [
            FunctionCall(name=call.name, arguments=json.dumps(call.arguments, ensure_ascii=False))
            for call in parsed.tool_calls
            if call.status == ToolCallParseStatus.OK
        ]
        content = parsed.content or (None if calls else "")
        return parsed.reasoning_content or None, content, calls


def install_renderer(engine_client: Any, renderer_config: Any, tokenizer_name: str) -> None:
    """Render and parse every chat-style route with the renderer. Call before the serving objects exist."""
    renderers = Renderers(renderer_config, tokenizer_name)
    renderer_cls = type("RendererBackedRenderer", (RendererBackedRenderer,), {"renderers": renderers})
    parser_cls = type("RendererParser", (RendererParser,), {"renderers": renderers})
    engine_client.renderer = renderer_cls(engine_client.vllm_config, engine_client.renderer.tokenizer)
    ParserManager.get_parser = classmethod(lambda cls, *args, **kwargs: parser_cls)

    # Every request stops on the generation config's eos ids, so the renderer's stop ids join them.
    generation_config = engine_client.input_processor.generation_config_fields
    eos_token_ids = generation_config.get("eos_token_id")
    eos_token_ids = [eos_token_ids] if isinstance(eos_token_ids, int) else list(eos_token_ids or [])
    stop_token_ids = renderers.get({}).get_stop_token_ids()
    generation_config["eos_token_id"] = list(dict.fromkeys([*eos_token_ids, *stop_token_ids]))
