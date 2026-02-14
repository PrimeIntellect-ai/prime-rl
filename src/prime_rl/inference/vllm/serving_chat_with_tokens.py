from collections.abc import AsyncGenerator
from typing import ClassVar, Optional, Union

from fastapi import Request
from pydantic import Field
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.openai.engine.serving import GenerationError
from vllm.entrypoints.utils import get_max_tokens
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.v1.sample.logits_processor import validate_logits_processors_parameters

logger = init_logger(__name__)


def _collapse_image_placeholders(
    original_tokens: list[int],
    override_tokens: list[int],
) -> list[int]:
    """
    Collapse pre-expanded image placeholder tokens in override_tokens.

    In multi-turn VLM conversations, tokens from previous turns already have
    expanded image placeholders (e.g., 64 consecutive <|image_pad|> tokens).
    If left as-is, _process_inputs re-expands each token individually, causing
    compounding token inflation across turns.

    Detects placeholder token IDs by comparing with the original engine tokens
    (which have single placeholders from _preprocess_chat), then collapses
    consecutive runs back to single tokens so _process_inputs can expand
    them correctly.
    """
    if not override_tokens or original_tokens == override_tokens:
        return override_tokens

    def get_block_tokens(tokens: list[int], min_run: int) -> set[int]:
        """Return token IDs that appear in consecutive runs >= min_run."""
        result = set()
        run_start = 0
        for i in range(1, len(tokens) + 1):
            if i == len(tokens) or tokens[i] != tokens[run_start]:
                if i - run_start >= min_run:
                    result.add(tokens[run_start])
                run_start = i
        return result

    # Placeholder tokens appear in blocks of 2+ in override but only as singles in original
    placeholder_ids = get_block_tokens(override_tokens, 2) - get_block_tokens(original_tokens, 2)
    if not placeholder_ids:
        return override_tokens

    result = []
    for token in override_tokens:
        if token in placeholder_ids and result and result[-1] == token:
            continue
        result.append(token)
    return result


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    field_names: ClassVar[Optional[set[str]]] = None
    tokens: list[int] = Field(description=("Prompt tokens to use for the request."))


class OpenAIServingChatWithTokens(OpenAIServingChat):
    """OpenAI-compatible generate API that allows token-in."""

    async def create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
        """
        Copy of OpenAIServingChat.create_chat_completion, adapted to use prompt
        ids directly via ChatCompletionRequestWithTokens.
        """
        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        reasoning_parser: ReasoningParser | None = None
        try:
            if self.reasoning_parser_cls:
                # Pass the same chat template kwargs as used in tokenization
                chat_template_kwargs = self._prepare_extra_chat_template_kwargs(
                    request.chat_template_kwargs,
                    self.default_chat_template_kwargs,
                )
                reasoning_parser = self.reasoning_parser_cls(
                    tokenizer,
                    chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
                )
        except RuntimeError as e:
            logger.exception("Error in reasoning parser creation.")
            return self.create_error_response(str(e))
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_prompts = result

        # For VLM models: collapse pre-expanded image placeholders before _process_inputs.
        # In multi-turn token prompts, previous turns' image placeholders are already expanded
        # (e.g., 64 consecutive <|image_pad|>). Without collapsing, _process_inputs re-expands
        # each token, inflating counts (64 → 127 → 253 → ...) across turns.
        if engine_prompts[0].get("multi_modal_data"):
            override_tokens = _collapse_image_placeholders(engine_prompts[0]["prompt_token_ids"], request.tokens)  # type: ignore
        else:
            override_tokens = request.tokens

        engine_prompts[0]["prompt_token_ids"] = override_tokens  # type: ignore

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)
        except (ValueError, TypeError, RuntimeError) as e:
            logger.exception("Error preparing request components")
            return self.create_error_response(e)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_text = self._extract_prompt_text(engine_prompt)

                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"

                max_tokens = get_max_tokens(
                    self.max_model_len,
                    request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens,
                    self._extract_prompt_len(engine_prompt),
                    self.default_sampling_params,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.model_config.logits_processor_pattern,
                        self.default_sampling_params,
                    )
                    validate_logits_processors_parameters(
                        self.logits_processors,
                        sampling_params,
                    )

                self._log_inputs(
                    sub_request_id,
                    engine_prompt,
                    params=sampling_params,
                    lora_request=lora_request,
                )

                trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.beam_search(
                        prompt=engine_prompt,
                        request_id=sub_request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                    )
                else:
                    tok_params = request.build_tok_params(self.model_config)
                    tokenization_kwargs = tok_params.get_encode_kwargs()

                    engine_request = self.input_processor.process_inputs(
                        sub_request_id,
                        engine_prompt,
                        sampling_params,
                        lora_request=lora_request,
                        tokenization_kwargs=tokenization_kwargs,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        data_parallel_rank=data_parallel_rank,
                    )
                    reasoning_ended = None
                    if reasoning_parser:
                        reasoning_ended = reasoning_parser.is_reasoning_end(
                            engine_request.prompt_token_ids or []  # type: ignore[attr-defined]
                        )
                        engine_request.reasoning_ended = reasoning_ended
                    generator = self.engine_client.generate(
                        engine_request,
                        sampling_params,
                        sub_request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        prompt_text=prompt_text,
                        tokenization_kwargs=tokenization_kwargs,
                        data_parallel_rank=data_parallel_rank,
                    )

                generators.append(generator)
        except ValueError as e:
            return self.create_error_response(e)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )

        try:
            return await self.chat_completion_full_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
            )
        except GenerationError as e:
            return self._convert_generation_error_to_response(e)
        except ValueError as e:
            return self.create_error_response(e)
