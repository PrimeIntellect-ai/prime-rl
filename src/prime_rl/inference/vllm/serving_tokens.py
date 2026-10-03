"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.scale_out.token_in_token_out.serving.ServingTokens`` that covers
prefix-cache salting, lora dispatch, multimodal content parts and features,
prompt logprobs, priority, ``data_parallel_rank`` header routing, server-side
``max_tokens`` defaulting, ``usage`` reporting, and expanded prompt metadata.
We subclass it for the per-token side arrays. A request whose
``sampling_params.extra_args`` carries a ``payload_dir`` (the orchestrator's train
rollouts on multi-node runs) gets its ``routed_experts`` and ``sampling_mask``
written to one file there and returned as ``payload`` segments. Otherwise
``routed_experts`` is surfaced as a ``{data, shape, start, dtype}`` base64 raw-byte
object (the form the PD router can merge and the renderers parse) instead of
upstream's single ``.npy`` base64 string.

Everything else delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
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

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture, compact_routed_experts


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    # Overrides upstream's base64 ``.npy`` string form with the compact object
    # the PD router merges and the renderers parse.
    routed_experts: dict[str, Any] | None = None  # type: ignore[assignment]
    payload: list[dict[str, Any]] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


class _GenerateRoutedExpertsCapture(RoutedExpertsCapture):
    async def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts"}),
                routed_experts=self.routed_experts.get(choice.index),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


class _PayloadCapture:
    """Takes each choice's routed_experts and sampling_mask off the engine output, so upstream
    encodes nothing inline, and writes them to one file per choice under ``directory``."""

    def __init__(self, generator: AsyncIterator[RequestOutput], start: int, directory: Path):
        self._generator = generator
        self._start = start
        self._directory = directory
        self.prompt_len = 0
        self.routed_experts: dict[int, Any] = {}
        self.sampling_masks: dict[int, list[list[int]]] = {}

    async def __aiter__(self):
        async for request_output in self._generator:
            self.prompt_len = len(request_output.prompt_token_ids)
            for output in request_output.outputs:
                if output.routed_experts is not None:
                    self.routed_experts[output.index] = output.routed_experts
                if output.sampling_mask is not None:
                    self.sampling_masks[output.index] = output.sampling_mask.token_ids
                output.routed_experts = None
                output.sampling_mask = None
            yield request_output

    async def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = []
        for choice in response.choices:
            # Routing row i is position start + i; mask row i is completion token i.
            arrays = []
            if (routed_experts := self.routed_experts.get(choice.index)) is not None:
                arrays.append(("routed_experts", self._start, compact_routed_experts(routed_experts)))
            if sampling_mask := self.sampling_masks.get(choice.index):
                rows = np.full((len(sampling_mask), max(map(len, sampling_mask))), -1, dtype=np.int32)
                for index, row in enumerate(sampling_mask):
                    rows[index, : len(row)] = row
                arrays.append(("sampling_mask", self.prompt_len, rows))
            payload = await asyncio.to_thread(_write_payload, self._directory, arrays) if arrays else None
            choices.append(PrimeRlGenerateResponseChoice(**choice.model_dump(), payload=payload))
        return PrimeRlGenerateResponse(**{**response.model_dump(exclude={"choices"}), "choices": choices})


def _write_payload(directory: Path, arrays: list[tuple[str, int, np.ndarray]]) -> list[dict[str, Any]]:
    """Write ``(field, first position, rows)`` arrays back to back into one new file and return
    their segments. The file is closed before the response, so readers on other nodes see it."""
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
    return segments


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens with compact routed experts and by-handle payloads."""

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | GenerateResponse:
        capture: _PayloadCapture | _GenerateRoutedExpertsCapture | None = None
        start = request.sampling_params.routed_experts_prompt_start
        payload_dir = (request.sampling_params.extra_args or {}).get("payload_dir")
        if payload_dir is not None:
            capture = _PayloadCapture(result_generator, start, Path(payload_dir))
        elif self.model_config.enable_return_routed_experts:
            capture = _GenerateRoutedExpertsCapture(result_generator, start=start)
        if capture is not None:
            result_generator = capture

        response = await super().serve_tokens_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            request_metadata,
        )

        if capture is not None and isinstance(response, GenerateResponse):
            response = await capture.post_process(response)

        return response
