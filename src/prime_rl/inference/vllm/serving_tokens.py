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

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
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
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    # Overrides upstream's base64 ``.npy`` string form with the compact
    # ``{data, shape, start, dtype}`` object the PD router merges and the
    # renderers parse.
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


class _GenerateRoutedExpertsCapture(RoutedExpertsCapture):
    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts"}),
                routed_experts=self.routed_experts.get(choice.index),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**dict(response), "choices": choices})


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens + compact routed experts + PD kv_transfer_params bridging."""

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
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
        # Capture routed_experts as vLLM streams request outputs, then post-process
        # the final response into our GenerateResponse subclass so the encoded
        # experts surface in the JSON.
        capture: _GenerateRoutedExpertsCapture | None = None
        if self.model_config.enable_return_routed_experts:
            capture = _GenerateRoutedExpertsCapture(
                result_generator,
                start=request.sampling_params.routed_experts_prompt_start,
            )
            result_generator = capture

        response = await super().serve_tokens_full_generator(
            request, result_generator, request_id, model_name, request_metadata
        )

        if capture is not None and isinstance(response, GenerateResponse):
            response = capture.post_process(response)

        return response
