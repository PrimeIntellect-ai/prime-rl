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

Raw media (``content_parts``) is expanded only at the placeholder ranges the renderer
declared per part, so a placeholder token id sampled earlier or carried in text stays text.

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

import itertools
from collections.abc import AsyncGenerator, AsyncIterator
from contextvars import ContextVar
from copy import copy
from typing import Any

import numpy as np
import pybase64
from fastapi import Request
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
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
    output (so upstream builds no per-token objects) and keeps them packed.

    ``request`` is the engine's request; ``serve_tokens_full_generator`` hands upstream a copy
    whose logprobs are already off when they are packed here, since upstream reads
    ``sampling_params.logprobs`` before it consumes the generator."""

    def __init__(self, generator: AsyncIterator[RequestOutput], request: GenerateRequest):
        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args or {}
        self._generator = generator
        self._paired = extra_args.get("prl_paired_capture", False)
        self._sampler_rows = extra_args.get("prl_sampler_rows", 0)
        # ``serve_tokens`` asks for flat logprobs, so the engine returns FlatLogprobs here.
        self._flat = (
            sampling_params.flat_logprobs
            and sampling_params.logprobs in (0, 1)
            and not sampling_params.logprob_token_ids
        )
        self.fields: dict[int, dict[str, Any]] = {}

    @property
    def strips_logprobs(self) -> bool:
        return self._flat

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
                mask = output.sampling_mask
                if mask is not None and (mask.counts <= 0).any():
                    raise ValueError(
                        "Missing sampling support: the step's supports exceeded inference.sampling_support_capacity"
                    )
                if self._paired or self._sampler_rows:
                    if mask is None or mask.ids.dtype != np.int64:
                        raise ValueError("Missing paired sampling-support evidence (needs enable_return_sampling_mask)")
                    ids, fields["compact_logprobs"] = serialize_paired_support(mask.ids, mask.counts)
                    if self._paired:
                        mask.ids = ids
                    else:
                        # Sampler rows are compact logprobs, not a support to replay.
                        output.sampling_mask = None
                    output.logprobs = None
                elif self._flat:
                    assert isinstance(output.logprobs, FlatLogprobs), "Did not output flat logprobs"
                    fields["completion_logprobs"] = encode_array(pack_sampled_logprobs(output.logprobs))
                    output.logprobs = None
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


# Media placeholder ranges a request declared in its content parts; read by the
# ``render_cmpl_async`` wrapper that ``install_placed_media`` puts on the renderer.
_PLACED_MEDIA: ContextVar[list[tuple[int, int]] | None] = ContextVar("prl_placed_media", default=None)


def hide_stray_placeholders(
    token_ids: list[int], placed: list[tuple[int, int]]
) -> tuple[list[int], list[int], int] | None:
    """Swap placeholder token ids outside the placed ranges for a token absent from the prompt.

    vLLM expands media at the first N placeholder tokens it finds, so a placeholder id
    sampled earlier or carried in text would take an image. Returns the masked ids, the
    swapped positions and the stand-in token, or None when nothing needs hiding.
    """
    placed_positions = {i for offset, length in placed for i in range(offset, offset + length)}
    placeholder_ids = {token_ids[i] for i in placed_positions}
    strays = [i for i, token in enumerate(token_ids) if token in placeholder_ids and i not in placed_positions]
    if not strays:
        return None
    present = set(token_ids)
    stand_in = next(token for token in itertools.count() if token not in present)
    masked = list(token_ids)
    for i in strays:
        masked[i] = stand_in
    return masked, strays, stand_in


def restore_stray_placeholders(expanded: list[int], token_ids: list[int], strays: list[int], stand_in: int) -> None:
    """Put the original ids back where the stand-in landed after media expansion."""
    positions = [i for i, token in enumerate(expanded) if token == stand_in]
    if len(positions) != len(strays):
        raise ValueError(f"Expected {len(strays)} hidden placeholder tokens after expansion, found {len(positions)}")
    for position, stray in zip(positions, strays):
        expanded[position] = token_ids[stray]


def install_placed_media(renderer: Any) -> None:
    """Expand media only at the ranges the request declared (see ``hide_stray_placeholders``)."""
    render_cmpl_async = renderer.render_cmpl_async

    async def render_placed(prompts, *args, **kwargs):
        placed = _PLACED_MEDIA.get()
        hidden = hide_stray_placeholders(prompts[0]["prompt_token_ids"], placed) if placed else None
        if hidden is None:
            return await render_cmpl_async(prompts, *args, **kwargs)
        masked, strays, stand_in = hidden
        token_ids = prompts[0]["prompt_token_ids"]
        (engine_input,) = await render_cmpl_async([{**prompts[0], "prompt_token_ids": masked}], *args, **kwargs)
        restore_stray_placeholders(engine_input["prompt_token_ids"], token_ids, strays, stand_in)
        return [engine_input]

    renderer.render_cmpl_async = render_placed


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with packed per-token payloads, expanding media only where it was placed."""

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        sampling_params = request.sampling_params
        extra_args = sampling_params.extra_args or {}
        captures = extra_args.get("prl_replay", False) or extra_args.get("prl_sampler_rows", 0)
        if captures and (request.stream or request.output_mode != "tokens"):
            return self.create_error_response("Sampling-support capture requires a non-streaming tokens request")
        if extra_args.get("prl_compact_logprobs", False):
            if request.stream or request.output_mode != "tokens" or sampling_params.logprobs is None:
                return self.create_error_response(
                    "Compact logprobs require a non-streaming tokens request with logprobs enabled"
                )
            # Captured on the GPU: the replayed support, or the sampled token plus the top ids.
            # logprobs = 0 (sampled token only) keeps vLLM's flat logprobs.
            if extra_args.get("prl_replay", False):
                sampling_params.extra_args = {**extra_args, "prl_paired_capture": True}
                sampling_params.logprobs = None
            elif rows := sampling_params.logprobs:
                sampling_params.extra_args = {**extra_args, "prl_sampler_rows": rows}
                sampling_params.logprobs = None
        if not request.stream and request.output_mode == "tokens":
            # Upstream's prompt-logprobs response path expects the per-token dicts.
            sampling_params.flat_logprobs = sampling_params.prompt_logprobs is None
            if not sampling_params.stop:
                sampling_params.detokenize = False
        parts = request.content_parts or []
        placed = (
            [(part["placeholder"]["offset"], part["placeholder"]["length"]) for part in parts]
            if parts and all("placeholder" in part for part in parts)
            else None
        )
        token = _PLACED_MEDIA.set(placed)
        try:
            return await super().serve_tokens(request, raw_request)
        finally:
            _PLACED_MEDIA.reset(token)

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
        derender_request: ChatCompletionRequest | None = None,
    ) -> ErrorResponse | GenerateResponse:
        if request.output_mode != "tokens":
            return await super().serve_tokens_full_generator(
                request, result_generator, request_id, model_name, request_metadata, derender_request=derender_request
            )
        packed = _PackedOutputs(result_generator, request)
        # The engine still needs the original sampling params; upstream formats from a copy without
        # the logprobs we pack, or it would build ``logprobs.content`` from them.
        sampling_params = copy(request.sampling_params)
        if packed.strips_logprobs:
            sampling_params.logprobs = None
        request = request.model_copy(update={"sampling_params": sampling_params})
        response = await super().serve_tokens_full_generator(
            request,
            packed,
            request_id,
            model_name,
            request_metadata,
            derender_request=derender_request,
        )
        if isinstance(response, GenerateTokensResponse):
            response = packed.post_process(response)
        return response
