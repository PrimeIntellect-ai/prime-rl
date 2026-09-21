"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs,
priority, ``data_parallel_rank`` header routing, server-side ``max_tokens``
defaulting and ``usage`` reporting. We subclass it for the one bit still
missing from the upstream handler: compact ``routed_experts`` export — when the
engine emits routing decisions, surface them as ``{data, shape, start, dtype}``
base64 raw-byte objects (the form the PD router can merge and the renderers
parse) instead of upstream's single ``.npy`` base64 string.

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
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
    """ServingTokens + compact routed experts."""

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
