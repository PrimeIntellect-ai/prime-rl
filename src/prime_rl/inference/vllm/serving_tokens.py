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
``logprobs.content``, opt-in ``compact_logprobs`` for sampled and top-k evidence,
and ``sampling_mask`` as CSR ``{ids, counts}`` int32 arrays
instead of one list per token.

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
from copy import copy
from typing import Any

import numpy as np
import pybase64
from fastapi import Request
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateTokensChoice,
    GenerateTokensResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.entrypoints.serve.engine.protocol import ErrorResponse
from vllm.logprobs import FlatLogprobs
from vllm.outputs import RequestOutput

from prime_rl.inference.vllm.paired_capture import serialize_paired_support

# vLLM's clamp for missing or -inf logprobs; renderers treat it as "no sampling evidence".
LOGPROB_SENTINEL = -9999.0


class PrimeRlGenerateResponseChoice(GenerateTokensChoice):
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    completion_logprobs: dict[str, Any] | None = None
    compact_logprobs: dict[str, Any] | None = None
    sampling_mask: dict[str, Any] | None = None  # type: ignore[assignment]


class PrimeRlGenerateResponse(GenerateTokensResponse):
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


class _PackedOutputs:
    """Wraps the result generator: takes the per-token payloads off each final
    output (so upstream builds no per-token objects) and keeps them packed."""

    def __init__(self, generator: AsyncIterator[RequestOutput], request: GenerateRequest):
        self._generator = generator
        self._request = request
        self._num_logprobs = request.sampling_params.logprobs
        extra_args = request.sampling_params.extra_args or {}
        self._paired = extra_args.get("prl_paired_capture", False)
        self._sampler_rows = extra_args.get("prl_sampler_rows", 0)
        self._draws = bool(extra_args.get("prl_sampler_draws"))
        self.fields: dict[int, dict[str, Any]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            for output in request_output.outputs:
                fields = self.fields.setdefault(output.index, {})
                if output.routed_experts is not None:
                    # vLLM's capturer picks uint8 or uint16 from the model's expert count. Rows end at the
                    # last forwarded token; a P/D decode instance's rows begin at its first forward.
                    num_tokens = len(request_output.prompt_token_ids) + len(output.token_ids)
                    fields["routed_experts"] = {
                        **encode_array(output.routed_experts),
                        "start": num_tokens - 1 - len(output.routed_experts),
                    }
                    output.routed_experts = None
                if self._paired or self._sampler_rows:
                    mask = output.sampling_mask
                    if mask is None or mask.ids.dtype != np.int64:
                        raise ValueError("Missing paired sampling-support evidence (needs enable_return_sampling_mask)")
                    ids, fields["compact_logprobs"] = serialize_paired_support(mask.ids, mask.counts)
                    if self._paired:
                        mask.ids = ids
                    else:
                        # Sampler rows are compact logprobs, not a support to replay.
                        output.sampling_mask = None
                        if self._draws:
                            fields["compact_logprobs"]["samples"] = True
                    output.logprobs = None
                elif isinstance(output.logprobs, FlatLogprobs) and (self._num_logprobs or 0) <= 1:
                    fields["completion_logprobs"] = encode_array(pack_sampled_logprobs(output.logprobs))
                    output.logprobs = None
                    # Upstream would build ``logprobs.content`` from the logprobs we took.
                    self._request.sampling_params.logprobs = None
                mask = output.sampling_mask
                if mask is not None:
                    fields["sampling_mask"] = {
                        "ids": encode_array(mask.ids.astype(np.int32, copy=False)),
                        "counts": encode_array(mask.counts),
                    }
                    output.sampling_mask = None
            yield request_output

    def post_process(self, response: GenerateTokensResponse) -> PrimeRlGenerateResponse:
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
        extra_args = sampling_params.extra_args or {}
        if extra_args.get("prl_compact_logprobs", False):
            if request.stream or sampling_params.logprobs is None:
                return self.create_error_response(
                    "Compact logprobs require a non-streaming request with logprobs enabled"
                )
            # Captured on the GPU: the replayed support, or the sampled token plus K draws or the top ids.
            if extra_args.get("prl_replay", False):
                sampling_params.extra_args = {**extra_args, "prl_paired_capture": True}
            else:
                rows = extra_args.get("prl_sampler_draws") or sampling_params.logprobs
                sampling_params.extra_args = {**extra_args, "prl_sampler_rows": rows}
            sampling_params.logprobs = None
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
        # The engine still needs the original sampling params while formatting strips payloads.
        request = request.model_copy(update={"sampling_params": copy(request.sampling_params)})
        packed = _PackedOutputs(result_generator, request)
        response = await super().serve_tokens_full_generator(
            request,
            packed,
            request_id,
            model_name,
            request_metadata,
        )
        if isinstance(response, GenerateTokensResponse):
            response = packed.post_process(response)
        return response
