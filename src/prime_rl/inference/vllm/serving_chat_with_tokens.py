import asyncio
import base64
import json
import math
import os
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import Any, ClassVar, Optional, Union
from uuid import uuid4

from fastapi import Request
from pydantic import Field
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.openai.engine.serving import GenerationError
from vllm.entrypoints.utils import get_max_tokens
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.reasoning import ReasoningParser
from vllm.sampling_params import BeamSearchParams, SamplingParams

from prime_rl.inference.vllm.serving_tokens import _RoutedExpertsCaptureBase

logger = init_logger(__name__)
CHAT_REPLAY_DIR_ENV = "PRIME_RL_CHAT_REPLAY_DIR"


class _RoutedExpertsCapture(_RoutedExpertsCaptureBase):
    """Chat-endpoint variant: mutates choices in-place because
    ``ChatCompletionResponseChoice`` is ``extra='allow'``, so an extra
    ``routed_experts`` attribute survives serialization."""

    def post_process(self, response: ChatCompletionResponse) -> None:
        for choice in response.choices:
            if choice.index in self.routed_experts:
                choice.routed_experts = self.routed_experts[choice.index]


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    field_names: ClassVar[Optional[set[str]]] = None
    tokens: list[int] = Field(description=("Prompt tokens to use for the request."))


class OpenAIServingChatWithTokens(OpenAIServingChat):
    """OpenAI-compatible generate API that allows token-in and routed experts capture."""

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
        try:
            response = await super().create_chat_completion(request, raw_request)
        except Exception as exc:
            await _dump_chat_replay(
                request=request,
                raw_request=raw_request,
                data_parallel_rank=_get_data_parallel_rank(self, raw_request),
                response=None,
                error={"type": type(exc).__name__, "message": str(exc)},
                nonfinite=None,
            )
            raise

        if isinstance(response, ChatCompletionResponse):
            nonfinite = _find_non_finite_chat_value(response)
            if nonfinite is not None:
                replay_dump_path = await _dump_chat_replay(
                    request=request,
                    raw_request=raw_request,
                    data_parallel_rank=_get_data_parallel_rank(self, raw_request),
                    response=response,
                    error=None,
                    nonfinite=nonfinite,
                )
                if replay_dump_path is not None:
                    nonfinite["replay_dump_path"] = replay_dump_path
                logger.warning(
                    "Non-finite /v1/chat/completions response value before JSON serialization: %s",
                    nonfinite,
                )

        return response

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation,
        tokenizer,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        # We need to override the full_generator to be able to capture the routed experts
        # By default, VLLM does not save the routed experts into ChatCompletionResponse.choices, so we need to capture them manually
        # How this works:
        # 1. We create a custom generator that encapsulates the original result_generator in self._generator
        # 2. We override it's __aiter__ method to also capture the routed experts as an extra field in ChatCompletionResponse.choices
        # 3. We override the full_generator method to use the custom generator instead of the original one if expert routing is enabled
        if self.model_config.enable_return_routed_experts:
            capture = _RoutedExpertsCapture(result_generator)
            result_generator = capture
        else:
            capture = None

        response = await super().chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )

        if capture and isinstance(response, ChatCompletionResponse):
            capture.post_process(response)

        return response

    async def create_chat_completion_with_tokens(
        self,
        request: ChatCompletionRequestWithTokens,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], ChatCompletionResponse, ErrorResponse]:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
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

        # We override prompt tokens directly.
        # VLM conversations use MITO (message-based) instead of TITO, so
        # multi_modal_data is not expected here.
        engine_prompts[0]["prompt_token_ids"] = request.tokens  # type: ignore

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
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                prompt_token_ids = self._extract_prompt_components(engine_prompt).token_ids

                # If we are creating sub requests for multiple prompts, ensure that they
                # have unique request ids.
                sub_request_id = request_id if len(engine_prompts) == 1 else f"{request_id}_{i}"

                prompt_len = self._extract_prompt_len(engine_prompt)
                if prompt_len >= max_model_len:
                    raise VLLMValidationError(
                        f"This model's maximum context length is "
                        f"{max_model_len} tokens. However, your request has "
                        f"{prompt_len} input tokens. Please reduce the length of "
                        "the input messages.",
                        parameter="input_tokens",
                        value=prompt_len,
                    )

                max_tokens = get_max_tokens(
                    max_model_len,
                    request.max_completion_tokens if request.max_completion_tokens is not None else request.max_tokens,
                    self._extract_prompt_len(engine_prompt),
                    self.default_sampling_params,
                    self.override_max_tokens,
                )

                sampling_params: SamplingParams | BeamSearchParams
                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens,
                        self.default_sampling_params,
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
                    reasoning_ended = (
                        reasoning_parser.is_reasoning_end(prompt_token_ids or []) if reasoning_parser else None
                    )

                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        sub_request_id,
                        lora_request=lora_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        data_parallel_rank=data_parallel_rank,
                        reasoning_ended=reasoning_ended,
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
        except GenerationError:
            raise  # Let FastAPI's global generation_error_handler handle it
        except ValueError as e:
            return self.create_error_response(e)


async def _dump_chat_replay(
    *,
    request: ChatCompletionRequest,
    raw_request: Optional[Request],
    data_parallel_rank: int | None,
    response: ChatCompletionResponse | None,
    error: dict[str, Any] | None,
    nonfinite: dict[str, Any] | None,
) -> str | None:
    dump_dir = _chat_replay_dir()
    if dump_dir is None:
        return None

    request_id = _request_id(request, raw_request)
    record = {
        "schema": "prime_rl.chat_completions_replay.v1",
        "created_unix": time.time(),
        "request_id": request_id,
        "endpoint": "/v1/chat/completions",
        "client": _safe_client(raw_request),
        "headers": _safe_replay_headers(raw_request),
        "data_parallel_rank": data_parallel_rank,
        "request": request.model_dump(mode="python", exclude_none=True),
        "response": response.model_dump(mode="python") if response is not None else None,
        "error": error,
        "nonfinite": nonfinite,
    }

    try:
        return await asyncio.to_thread(_write_chat_replay, dump_dir, request_id, record)
    except Exception:
        logger.exception("Failed to dump /v1/chat/completions replay record")
        return None


def _chat_replay_dir() -> Path | None:
    raw_dir = os.environ.get(CHAT_REPLAY_DIR_ENV)
    if not raw_dir:
        output_dir = os.environ.get("OUTPUT_DIR")
        if output_dir:
            raw_dir = str(Path(output_dir) / "replay_dumps" / "chat_completions")
    return Path(raw_dir).expanduser() if raw_dir else None


def _write_chat_replay(dump_dir: Path, request_id: str, record: dict[str, Any]) -> str:
    request_dir = dump_dir / "requests"
    request_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)[:180] or f"chatcmpl-{uuid4().hex[:16]}"
    path = request_dir / f"{filename}.json"
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(_json_safe(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return str(path)


def _request_id(request: ChatCompletionRequest, raw_request: Optional[Request]) -> str:
    if raw_request is not None:
        metadata = getattr(raw_request.state, "request_metadata", None)
        metadata_request_id = getattr(metadata, "request_id", None)
        if metadata_request_id:
            return str(metadata_request_id)
    if request.request_id:
        return str(request.request_id)
    return f"chatcmpl-dump-{uuid4().hex[:16]}"


def _safe_client(raw_request: Optional[Request]) -> dict[str, Any] | None:
    if raw_request is None or raw_request.client is None:
        return None
    return {
        "host": raw_request.client.host,
        "port": raw_request.client.port,
    }


def _safe_replay_headers(raw_request: Optional[Request]) -> dict[str, str]:
    if raw_request is None:
        return {}
    allowlist = {
        "endpoint-load-metrics-format",
        "traceparent",
        "x-b3-traceid",
        "x-data-parallel-rank",
        "x-request-id",
        "x-session-id",
        "x-trace-id",
    }
    return {key: value for key, value in raw_request.headers.items() if key.lower() in allowlist}


def _get_data_parallel_rank(handler: OpenAIServingChat, raw_request: Optional[Request]) -> int | None:
    if raw_request is None:
        return None
    try:
        return handler._get_data_parallel_rank(raw_request)
    except Exception:
        return None


def _find_non_finite_chat_value(response: ChatCompletionResponse) -> dict[str, Any] | None:
    payload = response.model_dump(mode="python")
    paths = _find_non_finite_paths(payload)
    if not paths:
        return None
    return {
        "paths": paths[:64],
        "path_count": len(paths),
    }


def _find_non_finite_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, float):
        return [] if math.isfinite(value) else [path]
    if isinstance(value, np.floating):
        return [] if math.isfinite(float(value)) else [path]
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            paths.extend(_find_non_finite_paths(child, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for idx, child in enumerate(value):
            paths.extend(_find_non_finite_paths(child, f"{path}[{idx}]"))
        return paths
    return []


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__nonfinite_float__": repr(value)}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        if math.isfinite(float_value):
            return float_value
        return {"__nonfinite_float__": repr(float_value)}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value
