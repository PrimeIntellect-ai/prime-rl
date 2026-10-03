"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal content parts and features,
prompt logprobs, priority, ``data_parallel_rank`` header routing, server-side
``max_tokens`` defaulting, ``usage`` reporting, and expanded prompt metadata.
We subclass it to return the per-token payloads as packed arrays instead of
per-token JSON: ``routed_experts`` as ``{data, shape, start, dtype}`` base64
raw bytes (the form the PD router can merge and the renderers parse),
``completion_logprobs`` as a ``{data, shape, dtype}`` float32 array instead of
``logprobs.content``, and ``sampling_mask`` as CSR ``{ids, counts}`` int32 arrays
instead of one list per token, plus ``sampling_mask_logprobs`` (float32, parallel
to the ids) when the sampler logprobs at the mask ids are captured for score
centering.

Per-token Python objects are what makes the API server slow under RL load:
~100+ concurrent 16k-token requests keep tens of millions of them alive, and
every gen-2 GC pass walks all of them. So the handler also turns off
detokenization (``/generate`` returns token ids only; requests with stop strings
keep it) and asks for flat logprobs (a few lists per request instead of a dict
and ``Logprob`` objects per token).

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

import numpy as np
import pybase64
from fastapi import Request
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logprobs import FlatLogprobs
from vllm.outputs import RequestOutput

from prime_rl.inference.patches import NEG_INF_BITS
from prime_rl.inference.vllm.routed_experts import serialize_routed_experts

# vLLM's clamp for missing or -inf logprobs; renderers treat it as "no sampling evidence".
LOGPROB_SENTINEL = -9999.0


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    completion_logprobs: dict[str, Any] | None = None
    sampling_mask: dict[str, Any] | None = None  # type: ignore[assignment]
    sampling_mask_logprobs: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


def encode_array(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    return {
        "data": pybase64.b64encode(memoryview(array)).decode("ascii"),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }


def pack_sampled_logprobs(logprobs: FlatLogprobs) -> np.ndarray:
    """The sampled token's logprob per position (vLLM stores it first), clamped like upstream."""
    flat = np.asarray(logprobs.logprobs, dtype=np.float32)
    starts = np.asarray(logprobs.start_indices, dtype=np.int64)
    has_entry = starts < np.asarray(logprobs.end_indices, dtype=np.int64)
    values = np.full(len(starts), LOGPROB_SENTINEL, dtype=np.float32)
    values[has_entry] = flat[starts[has_entry]]
    return np.maximum(values, LOGPROB_SENTINEL)


def unpack_sampling_mask_logprobs(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split int64 mask entries (``monkey_patch_sampling_mask_logprobs``) into int32 ids
    and float32 sampler logprobs."""
    bits = packed.view(np.uint64)
    ids = (bits & np.uint64(0xFFFFFFFF)).astype(np.int32)
    logprobs = ((bits >> np.uint64(32)).astype(np.uint32).view(np.int32) ^ np.int32(NEG_INF_BITS)).view(np.float32)
    return ids, logprobs


class _PackedOutputs:
    """Wraps the result generator: takes the per-token payloads off each final
    output (so upstream builds no per-token objects) and keeps them packed."""

    def __init__(self, generator: AsyncIterator[RequestOutput], request: GenerateRequest):
        self._generator = generator
        self._request = request
        self._routed_experts_start = request.sampling_params.routed_experts_prompt_start
        self.fields: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                fields = self.fields.setdefault(output.index, {})
                routed_experts = serialize_routed_experts(output.routed_experts, start=self._routed_experts_start)
                if routed_experts is not None:
                    fields["routed_experts"] = routed_experts
                    output.routed_experts = None
                if isinstance(output.logprobs, FlatLogprobs):
                    fields["completion_logprobs"] = encode_array(pack_sampled_logprobs(output.logprobs))
                    output.logprobs = None
                    # Upstream would build ``logprobs.content`` from the logprobs we took.
                    self._request.sampling_params.logprobs = None
                mask = output.sampling_mask
                if mask is not None:
                    ids = mask.ids
                    if ids.dtype == np.int64:
                        ids, logprobs = unpack_sampling_mask_logprobs(ids)
                        fields["sampling_mask_logprobs"] = encode_array(logprobs)
                    fields["sampling_mask"] = {"ids": encode_array(ids), "counts": encode_array(mask.counts)}
                    output.sampling_mask = None
            yield request_output

    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts", "sampling_mask"}),
                **self.fields.get(choice.index, {}),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with packed per-token payloads."""

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        sampling_params = request.sampling_params
        if not request.stream:
            # Upstream's prompt-logprobs response path expects the per-token dicts.
            sampling_params.flat_logprobs = sampling_params.prompt_logprobs is None
            if not sampling_params.stop:
                sampling_params.detokenize = False
        return await super().serve_tokens(request, raw_request)

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        packed = _PackedOutputs(result_generator, request)
        response = await super().serve_tokens_full_generator(
            request,
            packed,
            request_id,
            model_name,
            request_metadata,
        )
        if isinstance(response, GenerateResponse):
            response = packed.post_process(response)
        return response
