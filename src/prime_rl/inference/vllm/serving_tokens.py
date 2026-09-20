"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs,
priority, ``data_parallel_rank`` header routing, server-side ``max_tokens``
defaulting and ``usage`` reporting. We subclass it for the bits still missing
from the upstream handler:

1. Compact ``routed_experts`` export — when the engine emits routing
   decisions, surface them as ``{data, shape, start, dtype}`` base64 raw-byte
   objects (the form the PD router can merge and the renderers parse) instead
   of upstream's single ``.npy`` base64 string.

2. ``kv_transfer_params`` bridging — upstream ``ServingTokens.serve_tokens``
   parses ``request.kv_transfer_params`` but never threads it into the engine,
   so PD disagg never fires on ``/inference/v1/generate``. Fixed upstream by
   https://github.com/vllm-project/vllm/pull/42644, which missed the 0.28.0
   cut — drop the bridge once we pin a release that includes it.

3. Opt-in compact logprobs — export flat numeric buffers before upstream
   constructs per-candidate response objects.

Other response fields and error handling delegate to upstream.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from copy import copy
from typing import Any

from fastapi import Request
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    RequestResponseMetadata,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.outputs import CompletionOutput, RequestOutput

from prime_rl.inference.vllm.compact_logprobs import serialize_compact_logprobs
from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    # Overrides upstream's base64 ``.npy`` string form with the compact
    # ``{data, shape, start, dtype}`` object the PD router merges and the
    # renderers parse.
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
        return PrimeRlGenerateResponse(**{**dict(response), "choices": choices})


class PrimeRlServingTokens(ServingTokens):
    """Token serving with compact metadata and PD kv_transfer_params bridging."""

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        compact = (request.sampling_params.extra_args or {}).get("prl_compact_logprobs", False)
        if compact:
            if request.stream or request.sampling_params.logprobs is None:
                return self.create_error_response(
                    "Compact logprobs require a non-streaming request with logprobs enabled"
                )
            request.sampling_params.flat_logprobs = True
        # Upstream parses ``request.kv_transfer_params`` but never threads it
        # into the engine, so decode receives an empty NIXL handshake and
        # re-prefills the prompt locally (~100x slower under concurrency).
        # Bridge it through ``sampling_params.extra_args`` so the engine's KV
        # connector picks the params up. Fixed upstream by vllm#42644 (merged
        # after 0.28.0) — drop once we pin a release that includes it.
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
        capture: _GenerateOutputCapture | None = None
        compact = (request.sampling_params.extra_args or {}).get("prl_compact_logprobs", False)
        if self.model_config.enable_return_routed_experts or compact:
            capture = _GenerateOutputCapture(
                result_generator,
                start=request.sampling_params.routed_experts_prompt_start,
                compact_top_logprobs=request.sampling_params.logprobs if compact else None,
            )
            result_generator = capture

        if compact:
            # The engine already has its sampling params. Disable only the
            # upstream response formatter's nested logprob construction.
            sampling_params = copy(request.sampling_params)
            sampling_params.logprobs = None
            request = request.model_copy(update={"sampling_params": sampling_params})

        response = await super().serve_tokens_full_generator(
            request, result_generator, request_id, model_name, request_metadata
        )

        if capture is not None and isinstance(response, GenerateResponse):
            response = capture.post_process(response)

        return response
