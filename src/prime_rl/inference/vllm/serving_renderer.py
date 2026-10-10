"""OpenAI chat completions through a renderer instead of a chat template.

With ``[inference] renderer`` set, the server answers ``/v1/chat/completions`` with this
handler instead of vLLM's. The renderer turns the request's messages and tools into prompt
token ids, the ``/inference/v1/generate`` handler samples them, and the renderer parses the
completion ids back into content, reasoning and tool calls. No chat template, tool parser or
reasoning parser runs, so a served prompt has exactly the tokens the renderer trains on.

A streamed request gets the finished completion as one chunk.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from renderers import create_renderer
from renderers.base import ToolCallParseStatus, load_tokenizer, template_field_names
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest, GenerateResponseChoice
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.sampling_params import RequestOutputKind
from vllm.utils import random_uuid


def _error(message: str, status_code: int = 400) -> JSONResponse:
    error = {"message": message, "type": "BadRequestError", "code": status_code}
    return JSONResponse({"error": error, "object": "error"}, status_code=status_code)


class RendererChatCompletions:
    """Serves ``/v1/chat/completions`` with a renderer on top of a ``ServingTokens`` handler."""

    def __init__(
        self,
        serving_tokens: ServingTokens,
        renderer_config: Any,
        tokenizer_name: str,
        default_chat_template_kwargs: Mapping[str, Any] | None = None,
        max_workers: int = 8,
    ):
        self.serving_tokens = serving_tokens
        self.renderer_config = renderer_config
        self.tokenizer_name = tokenizer_name
        self.default_chat_template_kwargs = dict(default_chat_template_kwargs or {})
        self.template_fields = template_field_names(renderer_config)
        model_config = serving_tokens.engine_client.model_config
        self.max_model_len = model_config.max_model_len
        self.default_sampling_params = model_config.get_diff_sampling_param()
        # Renderers wrap a tokenizer that one thread must own, so every worker builds its own.
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="renderer")
        self._local = threading.local()

    def _renderer(self, kwargs: tuple[tuple[str, str], ...]):
        renderers = self._local.__dict__.setdefault("renderers", {})
        if kwargs not in renderers:
            if not hasattr(self._local, "tokenizer"):
                self._local.tokenizer = load_tokenizer(self.tokenizer_name)
            chat_template_kwargs = {key: json.loads(value) for key, value in kwargs}
            renderers[kwargs] = create_renderer(
                self._local.tokenizer, self.renderer_config, chat_template_kwargs=chat_template_kwargs
            )
        return renderers[kwargs]

    def _render(self, kwargs, messages, tools) -> list[int]:
        return self._renderer(kwargs).render_ids(messages, tools=tools, add_generation_prompt=True)

    def _parse(self, kwargs, token_ids, tools, prompt_ids):
        return self._renderer(kwargs).parse_response(token_ids, tools=tools, prompt_ids=prompt_ids)

    def _stop_token_ids(self, kwargs) -> list[int]:
        return self._renderer(kwargs).get_stop_token_ids()

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    def _template_kwargs(self, request: ChatCompletionRequest) -> tuple[tuple[str, str], ...]:
        """The request's renderer kwargs over the server defaults; kwargs the renderer lacks are ignored."""
        kwargs = {**self.default_chat_template_kwargs, **(request.chat_template_kwargs or {})}
        return tuple(sorted((key, json.dumps(value)) for key, value in kwargs.items() if key in self.template_fields))

    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        if request.use_beam_search:
            return _error("Beam search is not supported with a renderer.")
        kwargs = self._template_kwargs(request)
        messages = [dict(message) for message in request.messages]
        tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if request.tools else None
        try:
            prompt_ids = await self._run(self._render, kwargs, messages, tools)
            stop_token_ids = await self._run(self._stop_token_ids, kwargs)
            max_tokens = get_max_tokens(
                self.max_model_len,
                request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens,
                len(prompt_ids),
                self.default_sampling_params,
            )
        except (ValueError, TypeError, KeyError) as error:
            return _error(str(error))

        sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)
        sampling_params.stop_token_ids = list(dict.fromkeys([*(sampling_params.stop_token_ids or []), *stop_token_ids]))
        sampling_params.skip_special_tokens = False
        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        generate_request = GenerateRequest(
            token_ids=prompt_ids,
            sampling_params=sampling_params,
            model=request.model,
            cache_salt=request.cache_salt,
            priority=request.priority,
        )
        result = await self.serving_tokens.serve_tokens(generate_request, raw_request)
        if isinstance(result, ErrorResponse):
            return JSONResponse(result.model_dump(), status_code=result.error.code)

        choices = [await self._choice(kwargs, choice, tools, prompt_ids, request) for choice in result.choices]
        completion: dict[str, Any] = {
            "id": f"chatcmpl-{random_uuid()}",
            "object": "chat.completion",
            "created": result.created or int(time.time()),
            "model": request.model or result.model,
            "choices": choices,
            "usage": result.usage.model_dump(exclude_none=True) if result.usage else None,
        }
        if request.return_token_ids:
            completion["prompt_token_ids"] = prompt_ids
        if request.stream:
            return StreamingResponse(self._stream(completion, request), media_type="text/event-stream")
        return JSONResponse(completion)

    async def _choice(self, kwargs, choice: GenerateResponseChoice, tools, prompt_ids, request) -> dict[str, Any]:
        token_ids = choice.token_ids or []
        parsed = await self._run(self._parse, kwargs, token_ids, tools, prompt_ids)
        calls = [call for call in parsed.tool_calls if call.status == ToolCallParseStatus.OK]
        # Strict clients reject content=None without tool calls; empty reasoning is no reasoning.
        message: dict[str, Any] = {
            "role": "assistant",
            "content": parsed.content or (None if calls else ""),
            "reasoning": parsed.reasoning_content or None,
            "reasoning_content": parsed.reasoning_content or None,
        }
        if calls:
            message["tool_calls"] = [
                {
                    "id": f"call_{random_uuid()}",
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
                }
                for call in calls
            ]
        finish_reason = "tool_calls" if calls else choice.finish_reason or "stop"
        result: dict[str, Any] = {
            "index": choice.index,
            "message": message,
            "logprobs": choice.logprobs.model_dump() if choice.logprobs is not None else None,
            "finish_reason": finish_reason,
        }
        if request.return_token_ids:
            result["token_ids"] = token_ids
        return result

    @staticmethod
    async def _stream(completion: dict[str, Any], request: ChatCompletionRequest) -> AsyncIterator[str]:
        chunk = {key: completion[key] for key in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
        for choice in completion["choices"]:
            delta = dict(choice["message"])
            if "tool_calls" in delta:
                delta["tool_calls"] = [{"index": i, **call} for i, call in enumerate(delta["tool_calls"])]
            body = {"index": choice["index"], "delta": delta, "logprobs": choice["logprobs"], "finish_reason": None}
            yield f"data: {json.dumps(chunk | {'choices': [body]})}\n\n"
            end = {"index": choice["index"], "delta": {}, "finish_reason": choice["finish_reason"]}
            yield f"data: {json.dumps(chunk | {'choices': [end]})}\n\n"
        if request.stream_options and request.stream_options.include_usage:
            yield f"data: {json.dumps(chunk | {'choices': [], 'usage': completion['usage']})}\n\n"
        yield "data: [DONE]\n\n"
