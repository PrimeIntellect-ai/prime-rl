"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal content parts and features,
prompt logprobs, priority, ``data_parallel_rank`` header routing, server-side
``max_tokens`` defaulting, ``usage`` reporting, and expanded prompt metadata.
We subclass it for two bits still missing from the upstream handler: compact
``routed_experts`` export, and the sampler logprobs at the sampling-mask ids. When
the engine emits routing decisions, surface them as ``{data, shape, start, dtype}``
base64 raw-byte objects (the form the PD router can merge and the renderers parse)
instead of upstream's single ``.npy`` base64 string. Sampling masks arrive packed
with their logprobs (see ``monkey_patch_sampling_mask_logprobs``) and are split
into ``sampling_mask`` / ``sampling_mask_logprobs``.

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
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
    # Overrides upstream's base64 ``.npy`` string form with the compact object
    # the PD router merges and the renderers parse.
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    sampling_mask_logprobs: list[list[float]] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


def _unpack_sampling_mask(packed: list[list[int]]) -> tuple[list[list[int]], list[list[float]] | None]:
    """Split ``float32 bits << 32 | id`` mask entries into ids and logprobs. All-zero high
    bits mean the logprob patch did not run (or every kept set is a singleton, which
    score centering ignores anyway)."""
    flat = np.array([v for row in packed for v in row], dtype=np.int64).view(np.uint64)
    if not (flat >> np.uint64(32)).any():
        return packed, None
    ids = (flat & np.uint64(0xFFFFFFFF)).astype(np.int64).tolist()
    logprobs = (flat >> np.uint64(32)).astype(np.uint32).view(np.float32).tolist()
    bounds = np.cumsum([0, *map(len, packed)]).tolist()
    rows = list(zip(bounds[:-1], bounds[1:]))
    return [ids[a:b] for a, b in rows], [logprobs[a:b] for a, b in rows]


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


def _split_sampling_masks(response: GenerateResponse) -> PrimeRlGenerateResponse:
    choices = []
    for choice in response.choices:
        fields = choice.model_dump(exclude={"sampling_mask"})
        if choice.sampling_mask is not None:
            fields["sampling_mask"], fields["sampling_mask_logprobs"] = _unpack_sampling_mask(choice.sampling_mask)
        choices.append(PrimeRlGenerateResponseChoice(**fields))
    return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with compact routed experts."""

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

        response = await super().serve_tokens_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            request_metadata,
        )

        if routed_experts is not None and isinstance(response, GenerateResponse):
            response = routed_experts.post_process(response)
        if os.environ.get("PRIME_RETURN_SAMPLING_MASK_LOGPROBS") == "1" and isinstance(response, GenerateResponse):
            response = _split_sampling_masks(response)

        return response
