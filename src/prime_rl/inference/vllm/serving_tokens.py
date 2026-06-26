"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM 0.22 ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.serve.disagg.serving.ServingTokens`` that already covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs,
priority, ``data_parallel_rank`` header routing and server-side ``max_tokens``
defaulting. We subclass it for the bits still missing from the upstream handler:

1. ``data_parallel_rank`` routing — read from the ``X-data-parallel-rank``
   header and forwarded to ``engine_client.generate``. Upstream ``ServingTokens``
   now does this too; we keep the equivalent path for the DP-replicated
   inference servers prime-RL runs.

2. Compact ``routed_experts`` export — when the engine emits routing
   decisions, surface them as base64 raw-byte payloads without requiring a vLLM
   source fork.

3. Server-side ``max_tokens`` defaulting — upstream ``ServingTokens`` now applies
   this itself (via ``GenerateRequest.is_sampling_param_provided`` +
   ``get_max_tokens``); we keep an equivalent guard so callers that omit
   ``max_tokens`` don't truncate at vLLM's 16-token ``SamplingParams`` default.

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable
from functools import cached_property
from typing import Any

from fastapi import Request
from vllm.entrypoints.openai.engine.protocol import (
    ErrorResponse,
    PromptTokenUsageInfo,
    RequestResponseMetadata,
    UsageInfo,
)
from vllm.entrypoints.serve.disagg.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
)
from vllm.entrypoints.serve.disagg.serving import ServingTokens
from vllm.entrypoints.utils import get_max_tokens
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    routed_experts: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]
    # Upstream ``GenerateResponse`` doesn't declare a ``usage`` field, so the
    # parent ``ServingTokens.serve_tokens_full_generator`` constructs it and
    # Pydantic silently drops it on serialization. Declare it here so the
    # router can extract per-run token counts (and cached-prefix tokens) for
    # platform billing — see https://github.com/PrimeIntellect-ai/router/pull/43.
    usage: UsageInfo | None = None


class _GenerateRoutedExpertsCapture(RoutedExpertsCapture):
    def post_process(self, response: GenerateResponse) -> PrimeRlGenerateResponse:
        choices = [
            PrimeRlGenerateResponseChoice(
                **choice.model_dump(exclude={"routed_experts"}),
                routed_experts=self.routed_experts.get(choice.index),
            )
            for choice in response.choices
        ]
        return PrimeRlGenerateResponse(
            request_id=response.request_id,
            choices=choices,
            prompt_logprobs=response.prompt_logprobs,
            kv_transfer_params=response.kv_transfer_params,
        )


class _FinalOutputCapture:
    """Wraps a ``RequestOutput`` async generator to record the last yielded item.

    Needed so the response builder can construct a ``usage`` block from
    ``final_res.prompt_token_ids`` / ``output.token_ids`` / ``num_cached_tokens``
    after delegating iteration to upstream.
    """

    def __init__(self, source: AsyncIterable[RequestOutput]) -> None:
        # ``source`` may be any async-iterable — including
        # ``_GenerateRoutedExpertsCapture``, which exposes the protocol via
        # ``async def __aiter__`` (an async generator function) and has no
        # ``__anext__`` method. Drive it through ``async for`` rather than
        # poking ``__anext__`` directly so both shapes work.
        self._source = source
        self.final_res: RequestOutput | None = None

    async def __aiter__(self) -> AsyncGenerator[RequestOutput, None]:
        async for item in self._source:
            self.final_res = item
            yield item


def _build_usage(final_res: RequestOutput) -> UsageInfo:
    assert final_res.prompt_token_ids is not None
    num_prompt_tokens = len(final_res.prompt_token_ids)
    if final_res.encoder_prompt_token_ids is not None:
        num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
    num_generated_tokens = sum(len(output.token_ids) for output in final_res.outputs)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    # Always emit cached tokens when vLLM reports any. Upstream gates this on
    # ``enable_prompt_tokens_details`` (default False) for OpenAI-API compat,
    # but ``/inference/v1/generate`` is prime-rl internal — the cache-discount
    # billing pipeline always wants the cached subset surfaced.
    if final_res.num_cached_tokens:
        usage.prompt_tokens_details = PromptTokenUsageInfo(cached_tokens=final_res.num_cached_tokens)
    return usage


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
    """ServingTokens + DP-rank routing + compact routed experts + max_tokens defaulting."""

    @cached_property
    def _max_tokens_defaults(self) -> tuple[dict, int | None]:
        """Server-side ``max_tokens`` defaulting inputs, mirroring upstream ``ServingTokens``.

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
    ) -> PrimeRlGenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        # Mirrors upstream ``ServingTokens.serve_tokens`` (vllm 0.22). Diffs:
        # (a) inject ``data_parallel_rank`` from the inbound header into
        # ``engine_client.generate``; (b) default ``sampling_params.max_tokens``
        # to ``max_model_len - prompt_len`` when the caller didn't set it; and
        # (c) dispatch to our overridden response builder so ``routed_experts``
        # makes it into the JSON.
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

        # Upstream ``ServingTokens.serve_tokens`` parses ``request.kv_transfer_params``
        # but never threads it into the engine, so PD disagg never fires on
        # ``/inference/v1/generate`` — decode receives an empty NIXL handshake
        # and ends up re-prefilling the prompt locally (~100× slower under
        # concurrency). Bridge it through ``sampling_params.extra_args`` so the
        # engine's KV connector picks the params up.
        #
        # Upstream fix: https://github.com/vllm-project/vllm/pull/42644 — drop
        # this block once we pin a vLLM version that includes it.
        if request.kv_transfer_params is not None:
            extra = sampling_params.extra_args or {}
            extra["kv_transfer_params"] = request.kv_transfer_params
            sampling_params.extra_args = extra

        # Server-side ``max_tokens`` defaulting — see module docstring. Upstream
        # ``ServingTokens`` now does this too; kept here so callers that omit
        # ``max_tokens`` don't get capped at vLLM's 16-token ``SamplingParams``
        # default.
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
    ) -> ErrorResponse | GenerateResponse:
        # Capture routed_experts as vLLM streams request outputs, then post-process
        # the final response into our GenerateResponse subclass so the encoded
        # experts surface in the JSON.
        capture: _GenerateRoutedExpertsCapture | None = None
        if self.model_config.enable_return_routed_experts:
            start = request.sampling_params.routed_experts_prompt_start
            capture = _GenerateRoutedExpertsCapture(
                result_generator,
                start=start,
            )
            result_generator = capture

        # Always capture the final ``RequestOutput`` so we can attach a
        # ``usage`` block to the response. The router parses ``usage`` for
        # per-run billing metrics; without it the cache-discount counter
        # (``vllm_router_run_cached_prompt_tokens_total``) stays at zero.
        final_capture = _FinalOutputCapture(result_generator)
        result_generator = final_capture

        response = await super().serve_tokens_full_generator(
            request, result_generator, request_id, model_name, request_metadata
        )

        if not isinstance(response, GenerateResponse):
            return response

        if capture is not None:
            response = capture.post_process(response)
        elif not isinstance(response, PrimeRlGenerateResponse):
            # Upgrade to the prime-rl subclass so the declared ``usage`` field
            # actually surfaces in JSON (the parent class would drop it).
            response = PrimeRlGenerateResponse(
                request_id=response.request_id,
                choices=[PrimeRlGenerateResponseChoice(**choice.model_dump()) for choice in response.choices],
                prompt_logprobs=response.prompt_logprobs,
                kv_transfer_params=response.kv_transfer_params,
            )

        if final_capture.final_res is not None:
            response.usage = _build_usage(final_capture.final_res)

        return response
