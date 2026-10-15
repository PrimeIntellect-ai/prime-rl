"""Native Chat Completions with exact tokens and optional training attribution."""

from __future__ import annotations

import asyncio
from threading import Lock

from renderers import Renderer, RendererConfig, create_renderer
from renderers.base import _resolve_renderer_config, is_multimodal, load_tokenizer
from renderers.client import build_mm_features
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
)
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.inputs import mm_input, tokens_input

from prime_rl.inference.vllm.serving_tokens import _PackedOutputs


class PrimeRlChatRequest(ChatCompletionRequest):
    return_training_metadata: bool = False

    def to_sampling_params(self, *args, **kwargs):
        params = super().to_sampling_params(*args, **kwargs)
        # Top-k and prompt-logprob responses retain vLLM's native representation.
        params.flat_logprobs = (
            self.return_training_metadata
            and params.prompt_logprobs is None
            and not self.top_logprobs
            and not self.logprob_token_ids
        )
        return params


class TrainingChoice(ChatCompletionResponseChoice):
    routed_experts: dict | None = None  # type: ignore[assignment]
    completion_logprobs: dict | None = None
    sampling_mask: dict | None = None


class TrainingCompletion(ChatCompletionResponse):
    choices: list[TrainingChoice]
    training_metadata: dict | None = None


class PrimeRlServingChat(OpenAIServingChat):
    prompt_renderer_config: RendererConfig
    prompt_renderer_lock: Lock
    prompt_renderer: Renderer | None = None
    prompt_tokenizer = None
    prompt_template_kwargs: dict | None = None

    async def create_chat_completion(self, request, raw_request=None):
        request = PrimeRlChatRequest.model_validate(request.model_dump())
        if request.return_training_metadata:
            if request.stream:
                return self.create_error_response("Training records require stream=false")
            request.return_token_ids = True
            request.return_tokens_as_token_ids = True
            request.include_reasoning = True
        return await super().create_chat_completion(request, raw_request)

    async def render_chat_request(self, request: ChatCompletionRequest):
        if request.continue_final_message or request.chat_template is not None:
            raise ValueError("Chat requests require complete messages and the server's configured renderer")
        error = await self._check_model(request)
        if error is not None:
            return error
        self._preflight(request.n or 1)
        if request.tool_choice not in (None, "none") and not self.online_renderer.use_harmony:
            tool_parser = self.parser_cls.tool_parser_cls if self.parser_cls is not None else None
            if tool_parser is None or (request.tool_choice == "auto" and not self.enable_auto_tools):
                return self.create_error_response("Tool choice requires an enabled tool-call parser")
        if self.parser_cls is not None:
            self._make_parser(
                request, self.renderer.tokenizer, self._effective_chat_template_kwargs(request)
            ).adjust_request(request)
        # Tokenizer/processor calls are synchronous. One worker-owned renderer is
        # sufficient; no rollout state or previously sampled tokens are kept here.
        return await asyncio.to_thread(self._render_prompt, request)

    def _render_prompt(self, request: ChatCompletionRequest):
        # Hold the lock in the worker thread even if the HTTP request is cancelled.
        with self.prompt_renderer_lock:
            if self.prompt_tokenizer is None:
                self.prompt_tokenizer = load_tokenizer(self.model_config.tokenizer)
            self.prompt_renderer_config = _resolve_renderer_config(self.prompt_tokenizer, self.prompt_renderer_config)
            template_fields = self.prompt_renderer_config.template_field_names()
            # vLLM defaults may include template options for unrelated models.
            kwargs = {
                key: value
                for key, value in self.default_chat_template_kwargs.items()
                if self.prompt_renderer_config._allow_opaque_template_kwargs or key in template_fields
            }
            kwargs.update(request.chat_template_kwargs or {})
            if request.reasoning_effort is not None:
                if "reasoning_effort" in template_fields:
                    kwargs.setdefault("reasoning_effort", request.reasoning_effort)
                elif "enable_thinking" in template_fields:
                    kwargs.setdefault("enable_thinking", request.reasoning_effort != "none")
            if self.prompt_renderer is None or kwargs != self.prompt_template_kwargs:
                self.prompt_renderer = create_renderer(
                    self.prompt_tokenizer, self.prompt_renderer_config, chat_template_kwargs=kwargs
                )
                self.prompt_template_kwargs = kwargs
            renderer = self.prompt_renderer
            messages = [dict(message) for message in request.messages]
            for message in messages:
                if (reasoning := message.pop("reasoning", None)) is not None:
                    message["reasoning_content"] = reasoning
                if message.get("reasoning_content") is None:
                    message.pop("reasoning_content", None)
            tools = [tool.model_dump(exclude_none=True) for tool in request.tools] if request.tools else None
            if request.tool_choice == "none" and self.online_renderer.exclude_tools_when_tool_choice_none:
                tools = None
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
            if request.return_training_metadata:
                request._prime_training_metadata = {
                    "message_spans": rendered.message_token_spans(),
                    "is_content": rendered.is_content or None,
                    "mm_token_type_id_map": renderer.mm_token_type_id_map if is_multimodal(renderer) else None,
                }
            engine_input = tokens_input(rendered.token_ids, cache_salt=request.cache_salt)
            if mmd is not None:
                engine_input = mm_input(
                    prompt_token_ids=rendered.token_ids,
                    **build_mm_features(renderer, mmd),
                    cache_salt=request.cache_salt,
                )
            return messages, [engine_input]

    async def chat_completion_full_generator(self, request, result_generator, *args, **kwargs):
        packed = _PackedOutputs(result_generator, request, collect=request.return_training_metadata)
        response = await super().chat_completion_full_generator(request, packed, *args, **kwargs)
        if not request.return_training_metadata or not isinstance(response, ChatCompletionResponse):
            return response
        return TrainingCompletion(
            **response.model_dump(exclude={"choices"}),
            choices=[
                TrainingChoice(
                    **choice.model_dump(exclude={"routed_experts"}),
                    **packed.fields.get(choice.index, {}),
                )
                for choice in response.choices
            ],
            training_metadata=request._prime_training_metadata,
        )
