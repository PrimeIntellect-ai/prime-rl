"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal content parts and features,
prompt logprobs, priority, ``data_parallel_rank`` header routing, server-side
``max_tokens`` defaulting, ``usage`` reporting, and expanded prompt metadata.
We subclass it for compact routed experts and opt-in numeric logprob transport.
When the engine emits routing decisions, surface them
as ``{data, shape, start, dtype}`` base64 raw-byte objects (the form the PD
router can merge and the renderers parse) instead of upstream's single ``.npy``
base64 string. Requests with ``extra_args.prl_compact_logprobs`` also export
logprobs as flat numeric buffers, bypassing per-candidate JSON objects.

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from copy import copy
from typing import Any

from fastapi import Request
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.outputs import CompletionOutput, RequestOutput

from prime_rl.inference.vllm.compact_logprobs import serialize_compact_logprobs
from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    # Overrides upstream's base64 ``.npy`` string form with the compact object
    # the PD router merges and the renderers parse.
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    compact_logprobs: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


class _GenerateOutputCapture(RoutedExpertsCapture):
    def __init__(
        self, generator: AsyncIterator[RequestOutput], start: int = 0, compact_top_logprobs: int | None = None
    ):
        super().__init__(generator, start=start)
        self.compact_top_logprobs = compact_top_logprobs
        self.outputs: dict[int, CompletionOutput] = {}

    async def __aiter__(self):
        async for response in super().__aiter__():
            if self.compact_top_logprobs is not None:
                for output in response.outputs:
                    self.outputs[output.index] = output
            yield response

    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        compact = {}
        if self.compact_top_logprobs is not None:
            for index, output in self.outputs.items():
                assert output.logprobs is not None, "Did not output logprobs"
                assert len(output.logprobs) == len(output.token_ids), "Logprob count does not match token count"
                compact[index] = serialize_compact_logprobs(output.logprobs, self.compact_top_logprobs)
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts"}),
                routed_experts=self.routed_experts.get(choice.index),
                compact_logprobs=compact.get(choice.index),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with compact routed experts and opt-in logprob buffers."""

    async def serve_tokens(
        self, request: GenerateRequest, raw_request: Request | None = None
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        if (request.sampling_params.extra_args or {}).get("prl_compact_logprobs", False):
            if request.stream or request.sampling_params.logprobs is None:
                return self.create_error_response(
                    "Compact logprobs require a non-streaming request with logprobs enabled"
                )
            request.sampling_params.flat_logprobs = True
        return await super().serve_tokens(request, raw_request)

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        compact = (request.sampling_params.extra_args or {}).get("prl_compact_logprobs", False)
        capture: _GenerateOutputCapture | None = None
        if self.engine_client.vllm_config.aux_output_config.enable_return_routed_experts or compact:
            capture = _GenerateOutputCapture(
                result_generator,
                start=request.sampling_params.routed_experts_prompt_start,
                compact_top_logprobs=request.sampling_params.logprobs if compact else None,
            )
            result_generator = capture

        if compact:
            # Disable only the response formatter; the engine still needs logprobs.
            response_sampling_params = copy(request.sampling_params)
            response_sampling_params.logprobs = None
            request = request.model_copy(update={"sampling_params": response_sampling_params})

        response = await super().serve_tokens_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            request_metadata,
        )

        if capture is not None and isinstance(response, GenerateResponse):
            response = capture.post_process(response)

        return response
