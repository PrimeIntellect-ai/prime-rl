"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal content parts and features,
prompt logprobs, priority, ``data_parallel_rank`` header routing, server-side
``max_tokens`` defaulting and ``usage`` reporting. We subclass it for the bits
still missing from the upstream handler:

1. Compact ``routed_experts`` export — when the engine emits routing
   decisions, surface them as ``{data, shape, start, dtype}`` base64 raw-byte
   objects (the form the PD router can merge and the renderers parse) instead
   of upstream's single ``.npy`` base64 string.

2. ``kv_transfer_params`` bridging — upstream ``ServingTokens.serve_tokens``
   parses ``request.kv_transfer_params`` but never threads it into the engine.
   Fixed upstream by https://github.com/vllm-project/vllm/pull/42644, which
   missed the 0.28.0 cut.

3. Prompt metadata — return the effective engine prompt and authoritative
   multimodal placeholder ranges after expansion. Drop this once
   https://github.com/vllm-project/vllm/pull/53187 is available in a release.

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from contextvars import ContextVar
from typing import Any

from fastapi import Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    # Overrides upstream's base64 ``.npy`` string form with the compact object
    # the PD router merges and the renderers parse.
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]
    prompt_token_ids: list[int] | None = None
    mm_placeholders: dict[str, list[PlaceholderRangeInfo]] | None = None


_response_mm_placeholders: ContextVar[dict[str, list[PlaceholderRangeInfo]] | None] = ContextVar(
    "response_mm_placeholders", default=None
)


def _extract_mm_placeholders(
    engine_input: Any,
) -> dict[str, list[PlaceholderRangeInfo]] | None:
    if not isinstance(engine_input, dict) or engine_input.get("type") != "multimodal":
        return None
    return {
        modality: [
            PlaceholderRangeInfo(offset=placeholder.offset, length=placeholder.length)
            for placeholder in sorted(ranges, key=lambda placeholder: placeholder.offset)
        ]
        for modality, ranges in engine_input["mm_placeholders"].items()
    }


class _GenerateRoutedExpertsCapture(RoutedExpertsCapture):
    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts"}),
                routed_experts=self.routed_experts.get(choice.index),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


class _PromptTokenIdsCapture:
    def __init__(self, source: AsyncIterable[RequestOutput]) -> None:
        self._source = source
        self.prompt_token_ids: list[int] | None = None

    async def __aiter__(self) -> AsyncGenerator[RequestOutput, None]:
        async for output in self._source:
            self.prompt_token_ids = output.prompt_token_ids
            yield output


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with Prime's remaining response and PD extensions."""

    def _log_inputs(
        self,
        request_id: str,
        inputs: Any,
        params: Any,
        lora_request: Any,
    ) -> None:
        _response_mm_placeholders.set(_extract_mm_placeholders(inputs))
        super()._log_inputs(request_id, inputs, params, lora_request)

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        # Fixed upstream by vllm#42644; drop once it is included in the pin.
        if request.kv_transfer_params is not None:
            extra = request.sampling_params.extra_args or {}
            extra["kv_transfer_params"] = request.kv_transfer_params
            request.sampling_params.extra_args = extra

        return await super().serve_tokens(request, raw_request)

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        routed_experts: _GenerateRoutedExpertsCapture | None = None
        if self.model_config.enable_return_routed_experts:
            routed_experts = _GenerateRoutedExpertsCapture(
                result_generator,
                start=request.sampling_params.routed_experts_prompt_start,
            )
            result_generator = routed_experts

        prompt_capture = _PromptTokenIdsCapture(result_generator)
        response = await super().serve_tokens_full_generator(
            request,
            prompt_capture,
            request_id,
            model_name,
            request_metadata,
        )

        if not isinstance(response, GenerateResponse):
            return response

        if routed_experts is not None:
            response = routed_experts.post_process(response)
        else:
            response = PrimeRlGenerateResponse(**response.model_dump())
        response.prompt_token_ids = prompt_capture.prompt_token_ids
        response.mm_placeholders = _response_mm_placeholders.get()
        return response
