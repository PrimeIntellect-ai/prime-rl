from collections.abc import AsyncGenerator
from typing import ClassVar, Optional, Union

import jinja2
from fastapi import Request
from pydantic import Field
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.utils import get_max_tokens
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers.mistral import (
    MistralTokenizer,
    maybe_serialize_tool_calls,
    truncate_tool_call_ids,
    validate_request_params,
)
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
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        try:
            lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

            model_name = self.models.model_name(lora_request)

            tokenizer = await self.engine_client.get_tokenizer()

            tool_parser = self.tool_parser

            if isinstance(tokenizer, MistralTokenizer):
                # because of issues with pydantic we need to potentially
                # re-serialize the tool_calls field of the request
                # for more info: see comment in `maybe_serialize_tool_calls`
                maybe_serialize_tool_calls(request)
                truncate_tool_call_ids(request)
                validate_request_params(request)

            if (
                request.tool_choice == "auto"
                and not (self.enable_auto_tools and tool_parser is not None)
                and not isinstance(tokenizer, MistralTokenizer)
                and not self.use_harmony
            ):
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
                )

            if request.tools is None or (request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none):
                tool_dicts = None
            else:
                tool_dicts = [tool.model_dump() for tool in request.tools]

            if not self.use_harmony:
                # Common case.
                error_check_ret = self._validate_chat_template(
                    request_chat_template=request.chat_template,
                    chat_template_kwargs=request.chat_template_kwargs,
                    trust_request_chat_template=self.trust_request_chat_template,
                )
                if error_check_ret is not None:
                    return error_check_ret
                conversation, engine_prompts = await self._preprocess_chat(
                    request,
                    tokenizer,
                    request.messages,
                    chat_template=request.chat_template or self.chat_template,
                    chat_template_content_format=self.chat_template_content_format,
                    add_generation_prompt=request.add_generation_prompt,
                    continue_final_message=request.continue_final_message,
                    tool_dicts=tool_dicts,
                    documents=request.documents,
                    chat_template_kwargs=request.chat_template_kwargs,
                    tool_parser=tool_parser,
                    add_special_tokens=request.add_special_tokens,
                )
            else:
                # For GPT-OSS.
                conversation, engine_prompts = self._make_request_with_harmony(request)
        except (ValueError, TypeError, RuntimeError, jinja2.TemplateError) as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(f"{e} {e.__cause__}")

        # In-place override the engine_prompts with the tokens from the request
        assert len(engine_prompts) == 1
        if engine_prompts[0]["prompt_token_ids"] != request.tokens:
            logger.warning(
                "Prompt tokens provided in request do not match the engine prompt tokens. This may happen due to retokenization discrepancies in multi-turn conversations. Since you are using the /v1/chat/completions/tokens endpoint, we assume you want this behavior and use the provided prompt tokens. If this is undesired, use the standard /v1/chat/completions endpoint instead."
            )
            logger.debug(f"engine_prompt_tokens:\n{engine_prompts[0]['prompt_token_ids']}")
            logger.debug(f"request_tokens:\n{request.tokens}")

        # For VLM models: collapse pre-expanded image placeholders before _process_inputs.
        # In multi-turn token prompts, previous turns' image placeholders are already expanded
        # (e.g., 64 consecutive <|image_pad|>). Without collapsing, _process_inputs re-expands
        # each token, inflating counts (64 → 127 → 253 → ...) across turns.
        if engine_prompts[0].get("multi_modal_data"):
            override_tokens = _collapse_image_placeholders(engine_prompts[0]["prompt_token_ids"], request.tokens)
        else:
            override_tokens = request.tokens

        engine_prompts[0]["prompt_token_ids"] = override_tokens

        request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_text, _, _ = self._get_prompt_components(engine_prompts[i])
                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=len(engine_prompt["prompt_token_ids"]),
                    default_sampling_params=self.default_sampling_params,
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
                    engine_prompts[i],
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
                    engine_request, tokenization_kwargs = await self._process_inputs(
                        sub_request_id,
                        engine_prompt,
                        sampling_params,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                    )

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
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        assert len(generators) == 1
        (result_generator,) = generators

        # Streaming response
        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
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
            )
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))
