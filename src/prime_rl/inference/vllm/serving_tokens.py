"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM 0.20 ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.serve.disagg.serving.ServingTokens`` that already covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs and
priority. Two prime-RL features are not in the upstream protocol though, so
we subclass it to add them back:

1. ``data_parallel_rank`` routing — read from the ``X-data-parallel-rank``
   header and forwarded to ``engine_client.generate``. The DP-replicated
   inference servers prime-RL runs need this to target a specific replica.

2. Server-side ``max_tokens`` defaulting — ``ServingTokens`` hands the
   client-supplied ``SamplingParams`` to the engine verbatim, and
   ``SamplingParams.max_tokens`` defaults to ``16`` (a dataclass-level
   default that predates the OpenAI-compat layer). Every other vLLM
   endpoint masks this server-side via
   ``vllm.entrypoints.utils.get_max_tokens`` (see e.g.
   ``OpenAIServingChat`` at ``serving.py:284``); the disagg endpoint
   skips that path. Mirror it here so callers that omit ``max_tokens``
   don't silently truncate at 16. Drop once vLLM patches upstream.

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import cached_property

from fastapi import Request
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    GenerateResponse,
)
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.entrypoints.utils import get_max_tokens
from vllm.sampling_params import RequestOutputKind, SamplingParams


async def _client_set_max_tokens(raw_request: Request | None) -> bool:
    """Whether the inbound JSON body carried ``sampling_params.max_tokens``.

    ``GenerateRequest.sampling_params`` is parsed into a ``SamplingParams``
    instance, which means an unset ``max_tokens`` is indistinguishable from
    an explicit ``max_tokens=16`` once the request reaches the handler —
    both surface as ``sampling_params.max_tokens == 16``. We re-read the
    cached body to recover that distinction. When we can't (no raw_request,
    non-JSON body, or read error), pessimistically assume the client did
    set it so we never clobber an explicit value.
    """
    if raw_request is None:
        return True
    try:
        body = await raw_request.json()
    except Exception:
        return True
    if not isinstance(body, dict):
        return True
    sp = body.get("sampling_params")
    return isinstance(sp, dict) and "max_tokens" in sp


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens + DP-rank routing + max_tokens defaulting."""

    @cached_property
    def _max_tokens_defaults(self) -> tuple[dict, int | None]:
        """Server-side ``max_tokens`` defaulting inputs, mirroring ``OpenAIServingChat``.

        Computed lazily because ``custom_init_app_state`` swaps in this
        subclass via ``object.__new__`` + ``__dict__.update`` (so our
        ``__init__`` never runs).
        """
        diff = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        override = (
            diff.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            # Upstream uses ``getattr(..., {})`` directly. Defensive ``or {}``
            # in case a downstream caller ever sets the attribute to ``None``
            # (``getattr``'s default only fires when the attribute is missing,
            # not when it exists with a ``None`` value).
            else (getattr(mc, "override_generation_config", None) or {}).get("max_new_tokens")
        )
        return diff, override

    async def serve_tokens(
        self,
        request: GenerateRequest,
        raw_request: Request | None = None,
    ) -> GenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        # Mirrors upstream ``ServingTokens.serve_tokens`` (vllm 0.20). Diffs:
        # (a) inject ``data_parallel_rank`` from the inbound header into
        # ``engine_client.generate``; (b) default ``sampling_params.max_tokens``
        # to ``max_model_len - prompt_len`` when the caller didn't set it.
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

        # Server-side ``max_tokens`` defaulting — see module docstring.
        # Mirrors ``OpenAIServingChat`` (vllm/entrypoints/openai/chat_completion/
        # serving.py:284) so callers that omit ``max_tokens`` don't get capped
        # at vLLM's 16-token ``SamplingParams`` default.
        if not await _client_set_max_tokens(raw_request):
            diff_sp, override = self._max_tokens_defaults
            sampling_params.max_tokens = get_max_tokens(
                max_model_len=self.model_config.max_model_len,
                max_tokens=None,
                input_length=len(request.token_ids),
                default_sampling_params=diff_sp,
                override_max_tokens=override,
            )

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
