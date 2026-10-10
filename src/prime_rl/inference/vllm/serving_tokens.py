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
instead of one list per token. A request whose ``sampling_params.extra_args``
carries a ``payload_dir`` (the orchestrator's train rollouts on multi-node runs)
gets its ``routed_experts`` and ``sampling_mask`` written to one file there and
returned as ``payload`` segments instead.

Per-token Python objects are what makes the API server slow under RL load:
~100+ concurrent 16k-token requests keep tens of millions of them alive, and
every gen-2 GC pass walks all of them. So the handler also turns off
detokenization (``/generate`` returns token ids only; requests with stop strings
keep it) and asks for flat logprobs (a few lists per request instead of a dict
and ``Logprob`` objects per token).

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
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

from prime_rl.inference.patches import PackedSamplingMask
from prime_rl.inference.vllm.routed_experts import compact_routed_experts, serialize_routed_experts

# vLLM's clamp for missing or -inf logprobs; renderers treat it as "no sampling evidence".
LOGPROB_SENTINEL = -9999.0


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    completion_logprobs: dict[str, Any] | None = None
    sampling_mask: dict[str, Any] | None = None  # type: ignore[assignment]
    payload: list[dict[str, Any]] | None = None


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


def mask_rows(mask: PackedSamplingMask) -> np.ndarray:
    """CSR sampling masks as ``[completion tokens, widest]`` int32 rows padded with -1."""
    rows = np.full((len(mask.counts), max(int(mask.counts.max(initial=0)), 1)), -1, dtype=np.int32)
    rows[np.arange(rows.shape[1]) < mask.counts[:, None]] = mask.ids
    return rows


class _PackedOutputs:
    """Wraps the result generator: takes the per-token payloads off each final
    output (so upstream builds no per-token objects) and keeps them packed, or
    collects them for a by-handle file when the request has a ``payload_dir``."""

    def __init__(self, generator: AsyncIterator[RequestOutput], request: GenerateRequest):
        self._generator = generator
        self._request = request
        self._routed_experts_start = request.sampling_params.routed_experts_prompt_start
        payload_dir = (request.sampling_params.extra_args or {}).get("payload_dir")
        self._payload_dir = Path(payload_dir) if payload_dir is not None else None
        self.fields: dict[int, dict[str, Any]] = {}
        # Choice index -> (field, first token position, rows) for the by-handle file.
        self.arrays: dict[int, list[tuple[str, int, np.ndarray]]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            prompt_len = len(request_output.prompt_token_ids or ())
            for output in request_output.outputs:
                fields = self.fields.setdefault(output.index, {})
                arrays = self.arrays[output.index] = []
                if output.routed_experts is not None:
                    if self._payload_dir is None:
                        fields["routed_experts"] = serialize_routed_experts(
                            output.routed_experts, start=self._routed_experts_start
                        )
                    else:
                        rows = compact_routed_experts(output.routed_experts)
                        # Anchored at the last forwarded token: under P/D a decode instance
                        # emits rows from its first forward (prompt_len - 1), not from start.
                        arrays.append(("routed_experts", prompt_len + len(output.token_ids) - 1 - len(rows), rows))
                    output.routed_experts = None
                if isinstance(output.logprobs, FlatLogprobs):
                    fields["completion_logprobs"] = encode_array(pack_sampled_logprobs(output.logprobs))
                    output.logprobs = None
                    # Upstream would build ``logprobs.content`` from the logprobs we took.
                    self._request.sampling_params.logprobs = None
                mask = output.sampling_mask
                if mask is not None:
                    if self._payload_dir is None:
                        fields["sampling_mask"] = {"ids": encode_array(mask.ids), "counts": encode_array(mask.counts)}
                    else:
                        # Mask row i is completion token i.
                        arrays.append(("sampling_mask", prompt_len, mask_rows(mask)))
                    output.sampling_mask = None
            yield request_output

    async def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = []
        for choice in response.choices:
            fields = self.fields.get(choice.index, {})
            if arrays := self.arrays.get(choice.index):
                fields["payload"] = await asyncio.to_thread(_write_payload, self._payload_dir, arrays)
            choices.append(
                PrimeRlGenerateResponseChoice(
                    **choice.model_dump(exclude={"routed_experts", "sampling_mask"}), **fields
                )
            )
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


def _write_payload(directory: Path, arrays: list[tuple[str, int, np.ndarray]]) -> list[dict[str, Any]]:
    """Write ``(field, first position, rows)`` arrays back to back into one new file and return
    their segments. The file is synced and closed before the response, so readers on other
    nodes see it; so is the directory when this call creates it."""
    created = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    path = str(directory / f"{uuid.uuid4().hex}.bin")
    segments = []
    offset = 0
    with open(path, "wb") as f:
        for field, pos, rows in arrays:
            f.write(rows.data)
            segments.append(
                dict(
                    field=field,
                    file=path,
                    offset=offset,
                    pos=pos,
                    rows=len(rows),
                    dtype=rows.dtype.name,
                    shape=list(rows.shape[1:]),
                )
            )
            offset += rows.nbytes
        os.fsync(f.fileno())
    if created:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return segments


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with packed or by-handle per-token payloads."""

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
            response = await packed.post_process(response)
        return response
