"""Prime-RL extensions to vLLM's `/inference/v1/generate` handler.

vLLM 0.20 ships a generic tokens-in / tokens-out handler at
``vllm.entrypoints.serve.disagg.serving.ServingTokens`` that already covers
prefix-cache salting, lora dispatch, multimodal features, prompt logprobs and
priority. Three prime-RL features are not in the upstream protocol though, so
we subclass it to add them back:

1. ``data_parallel_rank`` routing — read from the ``X-data-parallel-rank``
   header and forwarded to ``engine_client.generate``. The DP-replicated
   inference servers prime-RL runs need this to target a specific replica.

2. Compact ``routed_experts`` export — when the engine emits routing
   decisions, surface them as base64 raw-byte payloads without requiring a vLLM
   source fork.

3. Server-side ``max_tokens`` defaulting — ``ServingTokens`` hands the
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

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncGenerator
from functools import cached_property
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import Request
from renderers.mm_store import (
    _SAFE_FINGERPRINT_RE,
    _SAFE_MM_HASH_RE,
    _SAFE_RUN_ID_RE,
    MMFILE_PREFIX,
    MMRAW_PREFIX,
    mm_feature_envelope_matches,
    mm_feature_fingerprint,
    mm_feature_path,
    mm_processor_fingerprint,
    raw_image_path,
    split_mmfile_ref,
    split_mmraw_ref,
)
from vllm.entrypoints.openai.engine.protocol import ErrorResponse, RequestResponseMetadata
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

logger = logging.getLogger(__name__)

_MM_FEATURE_LOAD_WORKERS_ENV = "PRIME_RL_MM_FEATURE_LOAD_WORKERS"
_MM_FEATURE_LOAD_RETRIES = 3
_MM_FEATURE_LOAD_BACKOFF_S = 0.02
_mm_feature_executor: concurrent.futures.ThreadPoolExecutor | None = None
_mm_raw_processors: dict[str, Any] = {}
_mm_raw_processors_lock = threading.Lock()


class PrimeRlGenerateResponseChoice(GenerateResponseChoice):
    routed_experts: dict[str, Any] | None = None


class PrimeRlGenerateResponse(GenerateResponse):
    choices: list[PrimeRlGenerateResponseChoice]


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


class _MMFeatureArtifactError(Exception):
    def __init__(
        self,
        *,
        error_type: str,
        message: str,
        missing: list[dict[str, str]] | None = None,
        status_code: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.missing = missing or []
        self.status_code = status_code

    def response_message(self) -> str:
        return json.dumps(
            {
                "error_type": self.error_type,
                "message": str(self),
                "missing": self.missing,
            },
            separators=(",", ":"),
        )


def _mm_feature_load_workers() -> int:
    raw = os.getenv(_MM_FEATURE_LOAD_WORKERS_ENV, "8").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


def _get_mm_feature_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _mm_feature_executor
    if _mm_feature_executor is None:
        _mm_feature_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=_mm_feature_load_workers(),
            thread_name_prefix="prime-mmfile",
        )
    return _mm_feature_executor


def _mm_feature_env_run_id() -> str:
    run_id = os.environ.get("RUN_ID", "").strip()
    if not run_id or not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_store",
            message="RUN_ID must be set to a safe run id for legacy mmfile refs.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    return run_id


def _parse_mmfile_ref(ref: str, *, expected_modality: str, expected_hash: str) -> tuple[str, str, str, str]:
    try:
        run_id, fingerprint, modality, mm_hash = split_mmfile_ref(ref)
    except ValueError as exc:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message=f"Invalid mmfile ref shape for {expected_modality}.",
            status_code=HTTPStatus.BAD_REQUEST,
        ) from exc
    if run_id is None:  # legacy 5-part ref: run_id comes from this process's env
        run_id = _mm_feature_env_run_id()
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message="mmfile run_id contains unsafe characters.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if modality != expected_modality:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message=(f"mmfile modality {modality!r} does not match slot modality {expected_modality!r}."),
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if mm_hash != expected_hash:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message="mmfile hash does not match the slot mm_hash.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if not _SAFE_FINGERPRINT_RE.fullmatch(fingerprint):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message="mmfile fingerprint contains unsafe characters.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if modality != "image":
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message=f"Unsupported mmfile modality: {modality!r}.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if not _SAFE_MM_HASH_RE.fullmatch(mm_hash):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message="mmfile hash contains unsafe characters.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    expected_fingerprint = mm_feature_fingerprint(family="qwen_vl", spatial_merge_size=2)
    if fingerprint != expected_fingerprint:
        raise _MMFeatureArtifactError(
            error_type="incompatible_mm_feature_artifact",
            message=(
                "mmfile fingerprint is not compatible with this vLLM process "
                f"(got {fingerprint}, expected {expected_fingerprint})."
            ),
            status_code=HTTPStatus.BAD_REQUEST,
        )
    return run_id, fingerprint, modality, mm_hash


def _parse_mmraw_ref(
    ref: str, *, expected_modality: str, expected_hash: str
) -> tuple[str, str, str, str, str, list[int]]:
    try:
        run_id, fingerprint, modality, mm_hash, raw_image_id, grid_thw = split_mmraw_ref(ref)
    except ValueError as exc:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message=f"Invalid mmraw ref shape for {expected_modality}.",
            status_code=HTTPStatus.BAD_REQUEST,
        ) from exc
    if not _SAFE_RUN_ID_RE.fullmatch(run_id):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message="mmraw run_id contains unsafe characters.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if modality != expected_modality:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message=(f"mmraw modality {modality!r} does not match slot modality {expected_modality!r}."),
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if mm_hash != expected_hash:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message="mmraw hash does not match the slot mm_hash.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if not _SAFE_FINGERPRINT_RE.fullmatch(fingerprint):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message="mmraw fingerprint contains unsafe characters.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if modality != "image":
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message=f"Unsupported mmraw modality: {modality!r}.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    if not _SAFE_MM_HASH_RE.fullmatch(mm_hash):
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message="mmraw hash contains unsafe characters.",
            status_code=HTTPStatus.BAD_REQUEST,
        )
    return run_id, fingerprint, modality, mm_hash, raw_image_id, grid_thw


def _mm_feature_path(*, run_id: str, fingerprint: str, modality: str, mm_hash: str) -> Path:
    # ``_parse_mmfile_ref`` validates run_id/fingerprint/modality/mm_hash and the
    # traversal guard before we reach here, so ``mm_store.mm_feature_path``'s
    # ValueError paths are unreachable; surface any as the reader's domain error.
    try:
        return mm_feature_path(run_id=run_id, fingerprint=fingerprint, modality=modality, mm_hash=mm_hash)
    except ValueError as exc:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_feature_ref",
            message=str(exc),
            status_code=HTTPStatus.BAD_REQUEST,
        ) from exc


def _decoded_image_grid_thw(item: Any) -> list[int]:
    elem = item.get("image_grid_thw")
    data = getattr(elem, "data", elem)
    if hasattr(data, "detach"):
        data = data.detach().cpu()
    if hasattr(data, "tolist"):
        data = data.tolist()
    grid = data[0] if isinstance(data, list) and data and isinstance(data[0], list) else data
    if not isinstance(grid, list) or len(grid) != 3:
        raise ValueError("decoded image_grid_thw does not have shape [T,H,W]")
    return [int(grid[0]), int(grid[1]), int(grid[2])]


def _decoded_image_placeholder_length(item: Any, *, spatial_merge_size: int) -> int:
    grid = _decoded_image_grid_thw(item)
    return int(grid[0]) * int(grid[1]) * int(grid[2]) // (spatial_merge_size**2)


def _processor_size_value(size: Any, key: str) -> int:
    value = getattr(size, key, None)
    if value is None and isinstance(size, dict):
        value = size.get(key)
    if value is None:
        raise ValueError(f"image processor size missing {key!r}")
    return int(value)


def _processor_fingerprint(processor: Any) -> tuple[str, int]:
    image_processor = processor.image_processor
    merge_size = int(getattr(image_processor, "merge_size"))
    fingerprint = mm_processor_fingerprint(
        family="qwen_vl",
        patch_size=int(getattr(image_processor, "patch_size")),
        merge_size=merge_size,
        temporal_patch_size=int(getattr(image_processor, "temporal_patch_size")),
        min_pixels=_processor_size_value(image_processor.size, "shortest_edge"),
        max_pixels=_processor_size_value(image_processor.size, "longest_edge"),
    )
    return fingerprint, merge_size


def _get_mm_raw_processor(model_name: str) -> Any:
    with _mm_raw_processors_lock:
        processor = _mm_raw_processors.get(model_name)
        if processor is not None:
            return processor
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name)
    with _mm_raw_processors_lock:
        return _mm_raw_processors.setdefault(model_name, processor)


def _load_mmfile_ref_sync(
    ref: str,
    *,
    expected_modality: str,
    expected_hash: str,
    expected_placeholder_length: int,
):
    import msgpack
    from vllm.multimodal.inputs import MultiModalKwargsItem
    from vllm.v1.serial_utils import MsgpackDecoder

    run_id, fingerprint, modality, mm_hash = _parse_mmfile_ref(
        ref, expected_modality=expected_modality, expected_hash=expected_hash
    )
    path = _mm_feature_path(run_id=run_id, fingerprint=fingerprint, modality=modality, mm_hash=mm_hash)
    missing = [
        {
            "run_id": run_id,
            "modality": modality,
            "mm_hash": mm_hash,
            "fingerprint": fingerprint,
        }
    ]

    packed: bytes | None = None
    for attempt in range(_MM_FEATURE_LOAD_RETRIES):
        try:
            packed = path.read_bytes()
            break
        except FileNotFoundError:
            if attempt + 1 == _MM_FEATURE_LOAD_RETRIES:
                raise _MMFeatureArtifactError(
                    error_type="missing_mm_feature_artifact",
                    message=f"Missing mmfile artifact: {path}",
                    missing=missing,
                ) from None
            time.sleep(_MM_FEATURE_LOAD_BACKOFF_S * (attempt + 1))

    try:
        artifact = msgpack.unpackb(packed, raw=False)
        envelope = artifact.get("envelope") if isinstance(artifact, dict) else None
        payload = artifact.get("payload") if isinstance(artifact, dict) else None
        if not isinstance(envelope, dict) or not isinstance(payload, bytes):
            raise ValueError("artifact must contain envelope and binary payload")
        if not mm_feature_envelope_matches(
            envelope,
            run_id=run_id,
            fingerprint=fingerprint,
            modality=modality,
            mm_hash=mm_hash,
            payload=payload,
            require_run_id=False,
        ):
            raise ValueError("artifact envelope does not match requested mmfile")

        decoder = MsgpackDecoder(t=MultiModalKwargsItem)
        item = decoder.decode(payload)
        placeholder_length = _decoded_image_placeholder_length(item, spatial_merge_size=2)
        if int(envelope.get("placeholder_length", -1)) != expected_placeholder_length:
            raise ValueError("artifact placeholder length does not match envelope")
        if placeholder_length != expected_placeholder_length:
            raise ValueError("decoded image_grid_thw does not match placeholder length")
        return item
    except _MMFeatureArtifactError:
        raise
    except Exception as exc:
        raise _MMFeatureArtifactError(
            error_type="corrupt_mm_feature_artifact",
            message=f"Corrupt mmfile artifact for {modality}:{mm_hash}: {exc}",
            missing=missing,
        ) from exc


def _load_mmraw_ref_sync(
    ref: str,
    *,
    expected_modality: str,
    expected_hash: str,
    expected_placeholder_length: int,
    processor_model_name: str,
):
    from PIL import Image
    from renderers.qwen3_vl import _image_hash
    from vllm.model_executor.models.qwen2_vl import _create_qwen2vl_field_factory
    from vllm.multimodal.inputs import MultiModalKwargsItems

    run_id, fingerprint, modality, mm_hash, raw_image_id, expected_grid = _parse_mmraw_ref(
        ref,
        expected_modality=expected_modality,
        expected_hash=expected_hash,
    )
    missing = [
        {
            "run_id": run_id,
            "modality": modality,
            "mm_hash": mm_hash,
            "fingerprint": fingerprint,
            "raw_image_id": raw_image_id,
        }
    ]
    try:
        path = raw_image_path(run_id=run_id, raw_image_id=raw_image_id)
    except ValueError as exc:
        raise _MMFeatureArtifactError(
            error_type="invalid_mm_raw_ref",
            message=str(exc),
            status_code=HTTPStatus.BAD_REQUEST,
        ) from exc

    pil = None
    for attempt in range(_MM_FEATURE_LOAD_RETRIES):
        try:
            with Image.open(path) as img:
                pil = img.convert("RGB")
            break
        except FileNotFoundError:
            if attempt + 1 == _MM_FEATURE_LOAD_RETRIES:
                raise _MMFeatureArtifactError(
                    error_type="missing_mm_raw_image",
                    message=f"Missing mmraw image: {path}",
                    missing=missing,
                ) from None
            time.sleep(_MM_FEATURE_LOAD_BACKOFF_S * (attempt + 1))
        except Exception as exc:
            raise _MMFeatureArtifactError(
                error_type="corrupt_mm_raw_image",
                message=f"Corrupt mmraw image for {modality}:{mm_hash}: {exc}",
                missing=missing,
            ) from exc
    if pil is None:
        raise _MMFeatureArtifactError(
            error_type="missing_mm_raw_image",
            message=f"Missing mmraw image: {path}",
            missing=missing,
        )

    actual_hash = _image_hash(pil)
    if actual_hash != mm_hash:
        raise _MMFeatureArtifactError(
            error_type="raw_mm_hash_mismatch",
            message=f"mmraw image hash mismatch for {modality}:{mm_hash}; got {actual_hash}",
            missing=missing,
            status_code=HTTPStatus.BAD_REQUEST,
        )

    try:
        processor = _get_mm_raw_processor(processor_model_name)
        expected_fingerprint, merge_size = _processor_fingerprint(processor)
        if fingerprint != expected_fingerprint:
            raise _MMFeatureArtifactError(
                error_type="incompatible_mm_raw_fingerprint",
                message=(
                    "mmraw fingerprint is not compatible with this vLLM process "
                    f"(got {fingerprint}, expected {expected_fingerprint})."
                ),
                status_code=HTTPStatus.BAD_REQUEST,
            )

        hf_inputs = processor.image_processor(images=[pil], return_tensors="pt")
        config = _create_qwen2vl_field_factory(merge_size)(hf_inputs)
        item = MultiModalKwargsItems.from_hf_inputs(hf_inputs, config)["image"][0]
        actual_grid = _decoded_image_grid_thw(item)
        actual_placeholder_length = _decoded_image_placeholder_length(item, spatial_merge_size=merge_size)
        if actual_grid != expected_grid:
            raise ValueError(f"processed image_grid_thw {actual_grid!r} != ref {expected_grid!r}")
        if actual_placeholder_length != expected_placeholder_length:
            raise ValueError(
                "processed image_grid_thw does not match placeholder length "
                f"({actual_placeholder_length} != {expected_placeholder_length})"
            )
        return item
    except _MMFeatureArtifactError:
        raise
    except Exception as exc:
        raise _MMFeatureArtifactError(
            error_type="raw_mm_grid_mismatch",
            message=f"mmraw materialization failed for {modality}:{mm_hash}: {exc}",
            missing=missing,
            status_code=HTTPStatus.BAD_REQUEST,
        ) from exc


async def _load_mmfile_ref(
    ref: str,
    *,
    expected_modality: str,
    expected_hash: str,
    expected_placeholder_length: int,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_mm_feature_executor(),
        lambda: _load_mmfile_ref_sync(
            ref,
            expected_modality=expected_modality,
            expected_hash=expected_hash,
            expected_placeholder_length=expected_placeholder_length,
        ),
    )


async def _load_mmraw_ref(
    ref: str,
    *,
    expected_modality: str,
    expected_hash: str,
    expected_placeholder_length: int,
    processor_model_name: str,
):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _get_mm_feature_executor(),
        lambda: _load_mmraw_ref_sync(
            ref,
            expected_modality=expected_modality,
            expected_hash=expected_hash,
            expected_placeholder_length=expected_placeholder_length,
            processor_model_name=processor_model_name,
        ),
    )


def _missing_cache_error_from_exception(exc: Exception, features: Any) -> _MMFeatureArtifactError | None:
    text = repr(exc)
    if "Expected a cached item" not in text:
        return None

    missing_hashes = set(re.findall(r"mm_hash=['\"]([^'\"]+)['\"]", text))
    missing: list[dict[str, str]] = []
    kwargs_data = getattr(features, "kwargs_data", None)
    hashes_by_modality = getattr(features, "mm_hashes", {}) or {}
    if isinstance(kwargs_data, dict):
        for modality, items in kwargs_data.items():
            hashes = hashes_by_modality.get(modality) or []
            for idx, item in enumerate(items):
                if item is not None or idx >= len(hashes):
                    continue
                mm_hash = hashes[idx]
                if missing_hashes and mm_hash not in missing_hashes:
                    continue
                missing.append({"modality": modality, "mm_hash": mm_hash})

    if not missing and missing_hashes:
        missing = [{"modality": "unknown", "mm_hash": h} for h in missing_hashes]

    return _MMFeatureArtifactError(
        error_type="missing_mm_cache_item",
        message=f"vLLM multimodal cache miss for cache-only slot: {exc}",
        missing=missing,
    )


class PrimeRlServingTokens(ServingTokens):
    """ServingTokens + DP-rank routing + compact routed experts + max_tokens defaulting."""

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
    ) -> PrimeRlGenerateResponse | ErrorResponse | AsyncGenerator[str, None]:
        # Mirrors upstream ``ServingTokens.serve_tokens`` (vllm 0.20). Diffs:
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
            try:
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
                slot_counts = {"none": 0, "inline": 0, "mmfile": 0, "mmraw": 0}
                load_start = time.monotonic()
                processor_model_name = str(getattr(self.model_config, "model", None) or model_name)

                async def decode_slot(modality: str, idx: int, item: str | None) -> MultiModalKwargsItem | None:
                    if item is None:
                        slot_counts["none"] += 1
                        return None
                    if item.startswith(f"{MMRAW_PREFIX}:"):
                        slot_counts["mmraw"] += 1
                        hashes = features.mm_hashes.get(modality) or []
                        placeholders = features.mm_placeholders.get(modality) or []
                        if idx >= len(hashes) or idx >= len(placeholders):
                            raise _MMFeatureArtifactError(
                                error_type="invalid_mm_raw_ref",
                                message=("mmraw slot has no matching hash or placeholder entry."),
                                status_code=HTTPStatus.BAD_REQUEST,
                            )
                        return await _load_mmraw_ref(
                            item,
                            expected_modality=modality,
                            expected_hash=hashes[idx],
                            expected_placeholder_length=placeholders[idx].length,
                            processor_model_name=processor_model_name,
                        )
                    if item.startswith(f"{MMFILE_PREFIX}:"):
                        slot_counts["mmfile"] += 1
                        hashes = features.mm_hashes.get(modality) or []
                        placeholders = features.mm_placeholders.get(modality) or []
                        if idx >= len(hashes) or idx >= len(placeholders):
                            raise _MMFeatureArtifactError(
                                error_type="invalid_mm_feature_ref",
                                message=("mmfile slot has no matching hash or placeholder entry."),
                                status_code=HTTPStatus.BAD_REQUEST,
                            )
                        return await _load_mmfile_ref(
                            item,
                            expected_modality=modality,
                            expected_hash=hashes[idx],
                            expected_placeholder_length=placeholders[idx].length,
                        )
                    slot_counts["inline"] += 1
                    return decode_mm_kwargs_item(item)

                if features.kwargs_data is not None:
                    for modality, items in features.kwargs_data.items():
                        hashes = features.mm_hashes.get(modality) or []
                        if len(items) != len(hashes):
                            raise _MMFeatureArtifactError(
                                error_type="invalid_mm_feature_ref",
                                message=(
                                    f"kwargs_data[{modality!r}] has {len(items)} items but mm_hashes has {len(hashes)}."
                                ),
                                status_code=HTTPStatus.BAD_REQUEST,
                            )
                        mm_kwargs[modality] = list(
                            await asyncio.gather(*(decode_slot(modality, idx, item) for idx, item in enumerate(items)))
                        )
                else:
                    for modality, hashes in features.mm_hashes.items():
                        slot_counts["none"] += len(hashes)
                        mm_kwargs[modality] = [None] * len(hashes)

                if any(slot_counts.values()):
                    logger.debug(
                        "decoded multimodal feature slots none=%d inline=%d mmfile=%d mmraw=%d load_ms=%.2f",
                        slot_counts["none"],
                        slot_counts["inline"],
                        slot_counts["mmfile"],
                        slot_counts["mmraw"],
                        (time.monotonic() - load_start) * 1000.0,
                    )

                engine_input = mm_input(
                    prompt_token_ids=request.token_ids,
                    mm_kwargs=mm_kwargs,  # type: ignore[arg-type]
                    mm_hashes=features.mm_hashes,
                    mm_placeholders=mm_placeholders,
                    cache_salt=request.cache_salt,
                )
            except _MMFeatureArtifactError as exc:
                return self.create_error_response(
                    exc.response_message(),
                    err_type=exc.error_type,
                    status_code=exc.status_code,
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

        try:
            response = await super().serve_tokens_full_generator(
                request, result_generator, request_id, model_name, request_metadata
            )
        except Exception as exc:
            if request.features is not None:
                mm_error = _missing_cache_error_from_exception(exc, request.features)
                if mm_error is not None:
                    return self.create_error_response(
                        mm_error.response_message(),
                        err_type=mm_error.error_type,
                        status_code=mm_error.status_code,
                    )
            raise

        if capture is not None and isinstance(response, GenerateResponse):
            response = capture.post_process(response)

        return response
