"""Native Chat Completions with exact tokens and optional training attribution."""

from __future__ import annotations

import asyncio
import base64
import json
from threading import Lock

import msgpack
from renderers import Renderer, RendererConfig, create_renderer
from renderers.base import ToolCallParseStatus, _resolve_renderer_config, is_multimodal, load_tokenizer
from renderers.client import _build_mm_features
from verifiers.v1.serve.encoding import msgpack_encoder
from vllm.entrypoints.generate.base.protocol import FunctionCall, ToolCall
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.scale_out.token_in_token_out.mm_serde import decode_mm_kwargs_item
from vllm.entrypoints.serve.utils.tool_calls_utils import maybe_filter_parallel_tool_calls
from vllm.inputs import mm_input, tokens_input
from vllm.multimodal.inputs import MultiModalKwargsItems, PlaceholderRange

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts


class TrainingChoice(ChatCompletionResponseChoice):
    routed_experts: dict | None = None  # type: ignore[assignment]
    sampling_mask: list[list[int]] | None = None


class TrainingCompletion(ChatCompletionResponse):
    choices: list[TrainingChoice]
    training_metadata: dict | None = None


class PrimeRlServingChat(OpenAIServingChat):
    training_renderer_config: RendererConfig
    training_renderer_lock: Lock
    training_renderer: Renderer | None = None
    training_tokenizer = None
    training_template_kwargs: dict | None = None

    async def render_chat_request(self, request: ChatCompletionRequest):
        if not getattr(request, "return_training_metadata", False):
            return await super().render_chat_request(request)
        if request.stream or not request.return_token_ids or not request.include_reasoning:
            raise ValueError("training metadata requires stream=false, return_token_ids=true, include_reasoning=true")
        if request.continue_final_message or request.chat_template is not None:
            raise ValueError("training records require complete messages and the server's configured renderer")
        error = await self._check_model(request)
        if error is not None:
            return error
        if self.engine_client.errored:
            raise self.engine_client.dead_error
        # Tokenizer/processor calls are synchronous. One worker-owned renderer is
        # sufficient; no rollout state or previously sampled tokens are kept here.
        return await asyncio.to_thread(self._render_training_prompt, request)

    def _render_training_prompt(self, request: ChatCompletionRequest):
        # Hold the lock in the worker thread even if the HTTP request is cancelled.
        with self.training_renderer_lock:
            if self.training_tokenizer is None:
                self.training_tokenizer = load_tokenizer(self.model_config.tokenizer)
            self.training_renderer_config = _resolve_renderer_config(
                self.training_tokenizer, self.training_renderer_config
            )
            kwargs = {**self.default_chat_template_kwargs, **(request.chat_template_kwargs or {})}
            template_fields = self.training_renderer_config.template_field_names()
            if request.reasoning_effort is not None:
                if "reasoning_effort" in template_fields:
                    kwargs.setdefault("reasoning_effort", request.reasoning_effort)
                elif "enable_thinking" in template_fields:
                    kwargs.setdefault("enable_thinking", request.reasoning_effort != "none")
            if self.training_renderer is None or kwargs != self.training_template_kwargs:
                self.training_renderer = create_renderer(
                    self.training_tokenizer, self.training_renderer_config, chat_template_kwargs=kwargs
                )
                self.training_template_kwargs = kwargs
            renderer = self.training_renderer
            messages = [dict(message) for message in request.messages]
            for message in messages:
                if "reasoning" in message:
                    message["reasoning_content"] = message.pop("reasoning")
            tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if request.tools else None
            rendered = renderer.render(messages, tools=tools, add_generation_prompt=request.add_generation_prompt)
            if len(rendered.token_ids) >= self.model_config.max_model_len:
                raise ValueError(
                    f"Prompt length ({len(rendered.token_ids)}) exceeds maximum context length "
                    f"({self.model_config.max_model_len})"
                )
            request.stop_token_ids = list(
                dict.fromkeys([*(request.stop_token_ids or []), *renderer.get_stop_token_ids()])
            )
            mmd = rendered.multi_modal_data
            # Decode with this request's renderer even if another request replaces the cache.
            request._prime_response_renderer = (renderer, tools)
            request._prime_training_metadata = {
                "message_spans": rendered.message_token_spans(),
                "is_content": rendered.is_content,
                "mm_token_type_id_map": renderer.mm_token_type_id_map if is_multimodal(renderer) else None,
                "multi_modal_data": base64.b64encode(
                    msgpack.packb(mmd, default=msgpack_encoder, use_bin_type=True)
                ).decode()
                if mmd is not None
                else None,
            }
            engine_input = tokens_input(rendered.token_ids, cache_salt=request.cache_salt)
            if mmd is not None:
                features = _build_mm_features(renderer, mmd)
                engine_input = mm_input(
                    prompt_token_ids=rendered.token_ids,
                    mm_kwargs=MultiModalKwargsItems(
                        {
                            modality: [decode_mm_kwargs_item(item) for item in items]
                            for modality, items in features["kwargs_data"].items()
                        }
                    ),
                    mm_hashes=features["mm_hashes"],
                    mm_placeholders={
                        modality: [PlaceholderRange(**span) for span in spans]
                        for modality, spans in features["mm_placeholders"].items()
                    },
                    cache_salt=request.cache_salt,
                )
            return messages, [engine_input]

    async def chat_completion_full_generator(self, request, result_generator, *args, **kwargs):
        experts = {}
        masks = {}

        async def capture():
            async for result in result_generator:
                for output in result.outputs:
                    routing = serialize_routed_experts(output.routed_experts, request.routed_experts_prompt_start)
                    if routing is not None:
                        experts[output.index] = routing
                    # Native vLLM emits .npy here; the PD router consumes compact arrays.
                    output.routed_experts = None
                    if output.sampling_mask is not None:
                        masks[output.index] = output.sampling_mask.token_ids
                yield result

        response_renderer = getattr(request, "_prime_response_renderer", None)
        if response_renderer is not None:
            kwargs["parser"] = None
        response = await super().chat_completion_full_generator(request, capture(), *args, **kwargs)
        if not isinstance(response, ChatCompletionResponse):
            return response
        if response_renderer is not None:
            renderer, tools = response_renderer
            for choice in response.choices:
                parsed = await asyncio.to_thread(renderer.parse_response, choice.token_ids, tools=tools)
                choice.message.content = parsed.content or None
                choice.message.reasoning = parsed.reasoning_content
                choice.message.tool_calls = [
                    ToolCall(
                        **({"id": call.id} if call.id else {}),
                        function=FunctionCall(
                            name=call.name,
                            arguments=call.arguments
                            if isinstance(call.arguments, str)
                            else json.dumps(call.arguments or {}),
                        ),
                    )
                    for call in parsed.tool_calls
                    if call.status == ToolCallParseStatus.OK
                ]
                if (
                    choice.message.tool_calls
                    and choice.finish_reason == "stop"
                    and request.tool_choice in (None, "auto", "required")
                ):
                    choice.finish_reason = "tool_calls"
        return TrainingCompletion(
            **response.model_dump(exclude={"choices"}),
            choices=[
                maybe_filter_parallel_tool_calls(
                    TrainingChoice(
                        **choice.model_dump(exclude={"routed_experts"}),
                        routed_experts=experts.get(choice.index),
                        sampling_mask=masks.get(choice.index),
                    ),
                    request,
                )
                for choice in response.choices
            ],
            training_metadata=getattr(request, "_prime_training_metadata", None),
        )
