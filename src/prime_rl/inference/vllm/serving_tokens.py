"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM 0.20 ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.serve.disagg.serving.ServingTokens`` that already covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs and
priority. Two prime-RL features are not in the upstream protocol though, so
we subclass it to add them back:

1. ``data_parallel_rank`` routing — read from the ``X-data-parallel-rank``
   header and forwarded to ``engine_client.generate``. The DP-replicated
   inference servers prime-RL runs need this to target a specific replica.

2. ``routed_experts`` per-token export — when the engine emits routing
   decisions (``enable_return_routed_experts``), surface them on each choice.
   This is what the trainer's router-replay path consumes.

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncGenerator

import numpy as np
from fastapi import Request
from pydantic import Field
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import clamp_prompt_logprobs
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.utils.collection_utils import as_list


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    routed_experts: dict | None = Field(
        default=None,
        description=(
            "Per-token expert routing decisions (base85-encoded int32 array + shape). "
            "Populated only when the engine was launched with "
            "``enable_return_routed_experts=True``; otherwise ``None``."
        ),
    )


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


def _encode_routed_experts(arr: np.ndarray) -> dict:
    return {
        "data": base64.b85encode(arr.tobytes()).decode("ascii"),
        "shape": list(arr.shape),
    }


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens + DP-rank routing + routed_experts export."""

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> PrimeRlGenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        # Mirrors upstream ``ServingTokens.serve_tokens`` (vllm 0.20). Only
        # diffs are: (a) inject ``data_parallel_rank`` from the inbound header
        # into ``engine_client.generate``, and (b) dispatch to our overridden
        # response builder so ``routed_experts`` makes it into the JSON.
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        if self.engine_client.errored:
            raise self.engine_client.dead_error

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)
        model_name = self.models.model_name(lora_request)

        request_id = f"generate-tokens-{self._base_request_id(raw_request, request.request_id)}"
        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        # Build the engine input — features-aware (MM) or text-only fallback.
        # Identical to upstream so we keep tracking it.
        if features := request.features:
            from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item
            from vllm.inputs import mm_input
            from vllm.multimodal.inputs import (
                MultiModalKwargsItem,
                PlaceholderRange,
            )

            mm_placeholders = {
                modality: [PlaceholderRange(offset=p.offset, length=p.length) for p in ranges]
                for modality, ranges in features.mm_placeholders.items()
            }
            mm_kwargs: dict[str, list[MultiModalKwargsItem | None]] = {}
            if features.kwargs_data is not None:
                for modality, items in features.kwargs_data.items():
                    mm_kwargs[modality] = [decode_mm_kwargs_item(item) if item is not None else None for item in items]
            else:
                for modality, hashes in features.mm_hashes.items():
                    mm_kwargs[modality] = [None] * len(hashes)
            engine_input = mm_input(
                prompt_token_ids=request.token_ids,
                mm_kwargs=mm_kwargs,  # type: ignore[arg-type]
                mm_hashes=features.mm_hashes,
                mm_placeholders=mm_placeholders,
                cache_salt=request.cache_salt,
            )
        else:
            (engine_input,) = await self.openai_serving_render.preprocess_completion(
                request,
                prompt_input=request.token_ids,
                prompt_embeds=None,
                skip_mm_cache=True,
            )

        sampling_params: SamplingParams = request.sampling_params
        if self.force_no_detokenize:
            sampling_params.detokenize = False
        if request.stream:
            sampling_params.output_kind = RequestOutputKind.DELTA

        self._log_inputs(
            request_id,
            engine_input,
            params=sampling_params,
            lora_request=lora_request,
        )

        trace_headers = None if raw_request is None else await self._get_trace_headers(raw_request.headers)

        result_generator = self.engine_client.generate(
            engine_input,
            sampling_params,
            request_id,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=request.priority,
            data_parallel_rank=self._get_data_parallel_rank(raw_request),
        )

        if request.stream:
            # Streaming path: defer to upstream — prime-RL's renderer client
            # only consumes the full response, so adding routed_experts to the
            # streaming choice schema is unnecessary churn.
            return self.serve_tokens_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                request_metadata,
            )

        return await self.serve_tokens_full_generator(
            request, result_generator, request_id, model_name, request_metadata
        )

    async def serve_tokens_full_generator(  # type: ignore[override]
        self,
        request: GenerateRequest,
        result_generator: AsyncGenerator[RequestOutput, None],
        request_id: str,
        model_name: str,
        request_metadata: RequestResponseMetadata,
    ) -> ErrorResponse | PrimeRlGenerateResponse:
        # Same shape as upstream's full generator — the only diff is we lift
        # ``routed_experts`` off each completion output into the choice.
        sampling_params: SamplingParams = request.sampling_params
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        assert final_res is not None

        choices: list[PrimeRlGenerateResponseChoice] = []
        num_generated_tokens = 0
        for output in final_res.outputs:
            token_ids = output.token_ids
            out_logprobs = output.logprobs

            if sampling_params.logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_tokens_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=sampling_params.logprobs,
                )
            else:
                logprobs = None

            routed_experts = None
            re_arr = getattr(output, "routed_experts", None)
            if re_arr is not None:
                routed_experts = _encode_routed_experts(re_arr)

            choices.append(
                PrimeRlGenerateResponseChoice(
                    index=output.index,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason if output.finish_reason else "stop",
                    token_ids=as_list(output.token_ids),
                    routed_experts=routed_experts,
                )
            )
            num_generated_tokens += len(output.token_ids)

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)

        request_metadata.final_usage_info = usage

        # Upstream constructs GenerateResponse with id=/created=/model=/usage=
        # which Pydantic silently drops (they're not declared on the model).
        # We only set the declared fields plus the request_id for traceability.
        _ = (model_name, time.time())  # touched for symmetry with upstream call site
        return PrimeRlGenerateResponse(
            request_id=request_id,
            choices=choices,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            kv_transfer_params=final_res.kv_transfer_params,
        )
