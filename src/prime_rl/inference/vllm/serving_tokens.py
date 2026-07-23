"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM 0.22 ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.serve.disagg.serving.ServingTokens`` that already covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs,
priority, ``data_parallel_rank`` header routing and server-side ``max_tokens``
defaulting. We subclass it for prime-RL behavior that is still missing or
customized:

1. ``data_parallel_rank`` routing — read from the ``X-data-parallel-rank``
   header and forwarded to ``engine_client.generate``. Upstream ``ServingTokens``
   now does this too; we keep the equivalent path for the DP-replicated
   inference servers prime-RL runs.

2. Compact ``routed_experts`` export — when the engine emits routing
   decisions, surface them as base64 raw-byte payloads without requiring a vLLM
   source fork.

3. Raw image refs for multimodal rollouts — renderers send a lightweight raw
   descriptor ref at every image slot (current and prior turns alike). This
   handler materializes every ref through multimodal adapters before vLLM sees
   the request.

4. Server-side ``max_tokens`` defaulting — upstream ``ServingTokens`` now applies
   this itself (via ``GenerateRequest.is_sampling_param_provided`` +
   ``get_max_tokens``); we keep an equivalent guard so callers that omit
   ``max_tokens`` don't truncate at vLLM's 16-token ``SamplingParams`` default.

Everything else (request/response schema, sampling params, error handling)
delegates to upstream so we track future vLLM changes for free.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from dataclasses import dataclass
from functools import cached_property, lru_cache
from http import HTTPStatus
from io import BytesIO
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
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.multimodal.cache import MultiModalCache
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from prime_rl.inference.vllm.routed_experts import RoutedExpertsCapture
from prime_rl.multimodal.registry import get_multimodal_adapter
from prime_rl.multimodal.schema import RawMMItem
from prime_rl.utils.mm import file_uri_to_path

logger = logging.getLogger(__name__)


@dataclass
class _MMImageRefError(Exception):
    message: str
    err_type: str = "invalid_mm_image_ref"
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST


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


@lru_cache(maxsize=8)
def _load_image_processor(model_name: str, trust_remote_code: bool):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError(f"{model_name!r} does not expose an image_processor")
    return image_processor


def _parse_raw_image_ref(raw_ref: str, *, feature_modality: str, mm_hash: str):
    from renderers.mm_store import split_raw_mm_ref

    try:
        ref = split_raw_mm_ref(raw_ref)
    except ValueError as exc:
        raise _MMImageRefError(str(exc)) from exc

    if ref.modality != feature_modality:
        raise _MMImageRefError(f"Expected feature modality {feature_modality!r}, got {ref.modality!r}")
    if ref.mm_hash != mm_hash:
        raise _MMImageRefError(f"Expected image hash {mm_hash}, got {ref.mm_hash}")
    return ref


def _read_verified_raw_image(ref) -> bytes:
    try:
        raw = file_uri_to_path(ref.raw_image_uri).read_bytes()
    except OSError as exc:
        raise _MMImageRefError(f"Unable to read raw image asset {ref.raw_image_uri!r}: {exc}") from exc

    actual_hash = hashlib.sha256(raw).hexdigest()[:32]
    if actual_hash != ref.mm_hash:
        raise _MMImageRefError(f"Raw image hash mismatch: expected {ref.mm_hash}, got {actual_hash}")
    return raw


def _decode_raw_image(raw: bytes, *, raw_image_uri: str):
    from PIL import Image

    try:
        return Image.open(BytesIO(raw)).convert("RGB")
    except OSError as exc:
        raise _MMImageRefError(f"Unable to decode raw image asset {raw_image_uri!r}: {exc}") from exc


def _materialize_raw_image_ref_sync(
    raw_ref: str,
    *,
    feature_modality: str,
    mm_hash: str,
    expected_placeholder_length: int,
    processor_model_name: str,
    trust_remote_code: bool,
):
    ref = _parse_raw_image_ref(raw_ref, feature_modality=feature_modality, mm_hash=mm_hash)
    raw = _read_verified_raw_image(ref)
    image = _decode_raw_image(raw, raw_image_uri=ref.raw_image_uri)
    image_processor = _load_image_processor(processor_model_name, trust_remote_code)
    item = RawMMItem(
        modality=ref.modality,
        family=ref.family,
        layout_fingerprint=ref.fingerprint,
        raw_image_uri=ref.raw_image_uri,
        payload=dict(ref.payload),
        raw_ref=raw_ref,
    )
    adapter = get_multimodal_adapter(ref.family)
    return adapter.materialize_for_vllm(
        image_processor,
        item,
        image,
        expected_placeholder_length,
    )


# (raw_ref, expected_placeholder_length, processor_model_name). The ref string is
# content-addressed (it embeds the image hash, layout fingerprint, family, and URI),
# so identical keys describe byte-identical materialization work.
_MaterializeKey = tuple[str, int, str]

_MM_MATERIALIZE_CACHE_GB_ENV = "PRIME_RL_MM_MATERIALIZE_CACHE_GB"
_MM_MATERIALIZE_LOG_EVERY = 1000


class _MaterializedRefCache:
    """Byte-bounded LRU of materialized raw image refs, with single-flight misses.

    Every request carries a ref for every image in its prompt (prior turns included),
    so multi-turn rollouts re-materialize the same images once per turn per rollout —
    this cache turns those repeats into lookups. Keys are content-addressed, so
    entries can only go cold, never stale: a post-eviction request simply misses and
    re-materializes from the durable ``file://`` asset.

    All mutation happens on the event loop thread (materialization itself runs in a
    worker thread via ``asyncio.to_thread``, but lookup/insertion/eviction happen in
    the async caller) — so no lock is needed. Do not touch this cache from sync code.

    ``max_bytes == 0`` disables caching and single-flight entirely: every call runs
    the materializer, byte-identical to the uncached path.

    Host-RAM note: this budget is additive to vLLM's own processor cache
    (``mm_processor_cache_gb × (api_server_count + data_parallel_size)``).
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._items: OrderedDict[_MaterializeKey, tuple[Any, int]] = OrderedDict()
        self._inflight: dict[_MaterializeKey, asyncio.Future] = {}
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    async def get_or_materialize(self, key: _MaterializeKey, materialize_fn: Callable[[], Any]) -> Any:
        if self.max_bytes == 0:
            return await asyncio.to_thread(materialize_fn)
        if key in self._items:
            self._items.move_to_end(key)
            self.hits += 1
            self._maybe_log()
            return self._items[key][0]
        if (inflight := self._inflight.get(key)) is not None:
            self.hits += 1
            self._maybe_log()
            return await inflight
        self.misses += 1
        self._maybe_log()
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            item = await asyncio.to_thread(materialize_fn)
        except BaseException as exc:
            # Never cache a failure: the exception propagates to every awaiter and
            # the next request for this key retries cleanly.
            future.set_exception(exc)
            # A never-awaited errored future logs a spurious warning at GC.
            future.exception()
            del self._inflight[key]
            raise
        future.set_result(item)
        del self._inflight[key]
        self._insert(key, item)
        return item

    def _insert(self, key: _MaterializeKey, item: Any) -> None:
        nbytes = MultiModalCache.get_item_size(item)
        if nbytes > self.max_bytes:
            # Don't churn the whole cache for one oversized entry.
            return
        self._items[key] = (item, nbytes)
        self._bytes += nbytes
        while self._bytes > self.max_bytes:
            _, (_, evicted_bytes) = self._items.popitem(last=False)
            self._bytes -= evicted_bytes
            self.evictions += 1

    def _maybe_log(self) -> None:
        total = self.hits + self.misses
        if total % _MM_MATERIALIZE_LOG_EVERY == 0:
            logger.info(
                "mm materialize cache: hits=%d misses=%d hit_rate=%.1f%% bytes=%d/%d evictions=%d",
                self.hits,
                self.misses,
                100.0 * self.hits / total,
                self._bytes,
                self.max_bytes,
                self.evictions,
            )


def _raw_ref_payloads_for_feature(
    features: Any, feature_modality: str, hashes: list[str]
) -> tuple[list[str], list[Any]]:
    from renderers.mm_store import is_raw_mm_ref

    kwargs_data = features.kwargs_data
    if kwargs_data is None or feature_modality not in kwargs_data:
        raise _MMImageRefError(f"v1 raw multimodal: modality {feature_modality!r} arrived with no raw refs")

    raw_refs = kwargs_data[feature_modality]
    placeholders = features.mm_placeholders.get(feature_modality)
    if placeholders is None:
        raise _MMImageRefError(f"v1 raw multimodal: modality {feature_modality!r} arrived with no placeholders")
    if len(raw_refs) != len(hashes):
        raise _MMImageRefError(
            f"Multimodal kwargs/hash length mismatch for {feature_modality}: {len(raw_refs)} != {len(hashes)}"
        )
    if len(placeholders) != len(hashes):
        raise _MMImageRefError(
            f"Multimodal placeholder/hash length mismatch for {feature_modality}: {len(placeholders)} != {len(hashes)}"
        )
    if not all(is_raw_mm_ref(item) for item in raw_refs):
        raise _MMImageRefError("v1 multimodal inference accepts raw descriptor refs only")
    return raw_refs, placeholders


async def _decode_raw_mm_kwargs(
    features: Any,
    *,
    processor_model_name: str,
    trust_remote_code: bool,
    cache: _MaterializedRefCache,
) -> dict[str, list[Any]]:
    # Flatten across modalities so every image in the request materializes
    # concurrently; single-flight in the cache dedupes identical refs across
    # (and within) requests. Fresh lists per request — vLLM's cache-injection
    # path replaces list elements in place, so never hand it cache-owned lists.
    flat: list[tuple[str, str, str, Any]] = []
    for feature_modality, hashes in features.mm_hashes.items():
        raw_refs, placeholders = _raw_ref_payloads_for_feature(features, feature_modality, hashes)
        flat.extend(zip([feature_modality] * len(hashes), raw_refs, hashes, placeholders, strict=True))

    def _materialize(feature_modality: str, raw_ref: str, mm_hash: str, placeholder: Any):
        return _materialize_raw_image_ref_sync(
            raw_ref,
            feature_modality=feature_modality,
            mm_hash=mm_hash,
            expected_placeholder_length=placeholder.length,
            processor_model_name=processor_model_name,
            trust_remote_code=trust_remote_code,
        )

    decoded = await asyncio.gather(
        *(
            cache.get_or_materialize(
                (raw_ref, placeholder.length, processor_model_name),
                lambda fm=feature_modality, r=raw_ref, h=mm_hash, p=placeholder: _materialize(fm, r, h, p),
            )
            for feature_modality, raw_ref, mm_hash, placeholder in flat
        )
    )

    mm_kwargs: dict[str, list[Any]] = {feature_modality: [] for feature_modality in features.mm_hashes}
    for (feature_modality, _, _, _), item in zip(flat, decoded, strict=True):
        mm_kwargs[feature_modality].append(item)
    return mm_kwargs


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens + DP-rank routing + routed experts + raw image refs + max_tokens defaulting."""

    @cached_property
    def _mm_materialize_cache(self) -> _MaterializedRefCache:
        """Materialized-ref cache, sized by ``PRIME_RL_MM_MATERIALIZE_CACHE_GB``
        (float GiB, default 2.0, ``0`` disables — the kill switch: disabled is
        byte-identical to the uncached path).

        A ``cached_property`` because ``custom_init_app_state`` grafts this
        subclass via ``object.__new__`` + ``__dict__.update``, so ``__init__``
        never runs (see ``_max_tokens_defaults``). One frontend process serves
        all DP ranks of an engine, so the cache is shared across ranks.
        """
        gb = float(os.environ.get(_MM_MATERIALIZE_CACHE_GB_ENV, "2.0"))
        return _MaterializedRefCache(max_bytes=int(gb * (1 << 30)))

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
        if features := request.features:
            from vllm.inputs import mm_input
            from vllm.multimodal.inputs import PlaceholderRange

            mm_placeholders = {
                modality: [PlaceholderRange(offset=p.offset, length=p.length) for p in ranges]
                for modality, ranges in features.mm_placeholders.items()
            }
            processor_model_name = getattr(self.model_config, "model", None) or model_name
            trust_remote_code = bool(getattr(self.model_config, "trust_remote_code", False))
            try:
                mm_kwargs = await _decode_raw_mm_kwargs(
                    features,
                    processor_model_name=processor_model_name,
                    trust_remote_code=trust_remote_code,
                    cache=self._mm_materialize_cache,
                )
            except _MMImageRefError as exc:
                return self.create_error_response(
                    message=exc.message,
                    err_type=exc.err_type,
                    status_code=exc.status_code,
                )
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
