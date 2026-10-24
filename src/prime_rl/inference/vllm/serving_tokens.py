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
and ``sampling_mask`` as CSR ``{ids, counts}`` int32 arrays instead of one list per
token. A request whose ``sampling_params.extra_args`` carries a ``payload_dir`` (the
orchestrator's train rollouts on multi-node runs) gets its ``routed_experts`` and
``sampling_mask`` written to one file there and returned as ``payload`` segments instead.

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
from copy import copy
from functools import cached_property
from pathlib import Path
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

from prime_rl.inference.vllm.compact_logprobs import serialize_compact_logprobs
from prime_rl.inference.vllm.paired_capture import serialize_paired_support
from prime_rl.transports.payload import RAGGED_FIELDS, id_bits, ragged_bytes, routed_layers

# vLLM's clamp for missing or -inf logprobs; renderers treat it as "no sampling evidence".
LOGPROB_SENTINEL = -9999.0


class PrimeRlGenerateResponseChoice(GenerateTokensChoice):
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    completion_logprobs: dict[str, Any] | None = None
    compact_logprobs: dict[str, Any] | None = None
    sampling_mask: dict[str, Any] | None = None  # type: ignore[assignment]
    payload: list[dict[str, Any]] | None = None


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
    output (so upstream builds no per-token objects) and keeps them packed, or
    collects them for a by-handle file when the request has a ``payload_dir``."""

    def __init__(
        self,
        generator: AsyncIterator[RequestOutput],
        request: GenerateRequest,
        layers: list[int] | None,
        bits: int,
    ):
        self._generator = generator
        self._request = request
        # Routing rows keep only these layers (None: all); mask ids are packed at ``bits`` bits.
        self._layers = layers
        self._bits = bits
        payload_dir = (request.sampling_params.extra_args or {}).get("payload_dir")
        self._payload_dir = Path(payload_dir) if payload_dir is not None else None
        self._num_logprobs = request.sampling_params.logprobs
        self._compact = (request.sampling_params.extra_args or {}).get("prl_compact_logprobs", False)
        self._paired = (request.sampling_params.extra_args or {}).get("prl_paired_capture", False)
        self.fields: dict[int, dict[str, Any]] = {}
        # Choice index -> (field, first token position, rows) for the by-handle file; rows of a
        # ragged field are CSR (counts, values).
        self.arrays: dict[int, list[tuple[str, int, Any]]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            prompt_len = len(request_output.prompt_token_ids or ())
            for output in request_output.outputs:
                fields = self.fields.setdefault(output.index, {})
                arrays = self.arrays[output.index] = []
                if output.routed_experts is not None:
                    # vLLM's capturer picks uint8 or uint16 from the model's expert count. Rows end at the
                    # last forwarded token; a P/D decode instance's rows begin at its first forward.
                    rows = output.routed_experts
                    start = prompt_len + len(output.token_ids) - 1 - len(rows)
                    if self._layers is not None:
                        rows = rows[:, self._layers]
                    if self._payload_dir is None:
                        fields["routed_experts"] = {**encode_array(rows), "start": start}
                    else:
                        arrays.append(("routed_experts", start, np.ascontiguousarray(rows)))
                    output.routed_experts = None
                if self._paired:
                    mask = output.sampling_mask
                    if mask is None or mask.ids.dtype != np.int64:
                        raise ValueError("Missing paired sampling-support evidence")
                    mask.ids, fields["compact_logprobs"] = serialize_paired_support(mask.ids, mask.counts)
                    output.logprobs = None
                elif self._compact:
                    assert output.logprobs is not None, "Did not output logprobs"
                    assert len(output.logprobs) == len(output.token_ids), "Logprob count does not match token count"
                    fields["compact_logprobs"] = serialize_compact_logprobs(
                        output.logprobs, self._num_logprobs, drop_filtered=output.sampling_mask is not None
                    )
                    output.logprobs = None
                    self._request.sampling_params.logprobs = None
                elif isinstance(output.logprobs, FlatLogprobs) and (self._num_logprobs or 0) <= 1:
                    fields["completion_logprobs"] = encode_array(pack_sampled_logprobs(output.logprobs))
                    output.logprobs = None
                    # Upstream would build ``logprobs.content`` from the logprobs we took.
                    self._request.sampling_params.logprobs = None
                mask = output.sampling_mask
                if mask is not None:
                    if self._payload_dir is None:
                        fields["sampling_mask"] = {
                            "ids": encode_array(mask.ids.astype(np.int32, copy=False)),
                            "counts": encode_array(mask.counts),
                        }
                    else:
                        # Mask row i is completion token i.
                        arrays.append(("sampling_mask", prompt_len, (mask.counts, mask.ids)))
                    output.sampling_mask = None
            yield request_output

    async def post_process(self, response: GenerateTokensResponse) -> PrimeRlGenerateResponse:
        choices = []
        for choice in response.choices:
            fields = self.fields.get(choice.index, {})
            if arrays := self.arrays.get(choice.index):
                fields["payload"] = await asyncio.to_thread(_write_payload, self._payload_dir, arrays, self._bits)
            choices.append(
                PrimeRlGenerateResponseChoice(
                    **choice.model_dump(exclude={"routed_experts", "sampling_mask"}), **fields
                )
            )
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


def _write_payload(directory: Path, arrays: list[tuple[str, int, Any]], bits: int) -> list[dict[str, Any]]:
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
            if field in RAGGED_FIELDS:
                counts, values = rows
                data = ragged_bytes(counts, values, offset, bits)
                segment = dict(rows=len(counts), dtype="uint32", shape=[])
            else:
                data = memoryview(rows).cast("B")
                segment = dict(rows=len(rows), dtype=rows.dtype.name, shape=list(rows.shape[1:]))
            f.write(data)
            segments.append(dict(field=field, file=path, offset=offset, pos=pos, **segment))
            offset += len(data)
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

    @cached_property
    def _payload_layout(self) -> tuple[list[int] | None, int]:
        """Routing rows' layers (None when every layer is routed) and mask id bits."""
        config = self.model_config.hf_text_config
        layers = routed_layers(config)
        return (None if len(layers) == config.num_hidden_layers else layers), id_bits(
            self.model_config.get_vocab_size()
        )

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        sampling_params = request.sampling_params
        if (sampling_params.extra_args or {}).get("prl_compact_logprobs", False):
            if request.stream or sampling_params.logprobs is None:
                return self.create_error_response(
                    "Compact logprobs require a non-streaming request with logprobs enabled"
                )
        extra_args = sampling_params.extra_args or {}
        paired = (
            not request.stream
            and extra_args.get("prl_replay", False)
            and extra_args.get("prl_compact_logprobs", False)
            and os.environ.get("PRL_TOP_P_BEFORE_TOP_K") == "1"
        )
        sampling_params.extra_args = {**extra_args, "prl_paired_capture": paired}
        if paired:
            # Full support evidence supplies both sampled and candidate logprobs.
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
        packed = _PackedOutputs(result_generator, request, *self._payload_layout)
        response = await super().serve_tokens_full_generator(
            request,
            packed,
            request_id,
            model_name,
            request_metadata,
        )
        if isinstance(response, GenerateTokensResponse):
            response = await packed.post_process(response)
        return response
