"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM 0.20's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass attaches its overrides without monkey-patching the parent.
    * ``_client_set_max_tokens`` distinguishes raw-body shapes correctly.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse, GenerateResponseChoice

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlGenerateResponse,
    PrimeRlGenerateResponseChoice,
    PrimeRlServingTokens,
    _build_usage,
    _client_set_max_tokens,
    _decode_raw_mm_kwargs,
    _FinalOutputCapture,
    _GenerateRoutedExpertsCapture,
    _MMImageRefError,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


class _FakeRawRequest:
    def __init__(self, body):
        self._body = body
        self._raise = isinstance(body, Exception)

    async def json(self):
        if self._raise:
            raise self._body
        return self._body


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_only_overrides_serve_tokens():
    assert PrimeRlServingTokens.serve_tokens is not PrimeRlServingTokens.__mro__[1].serve_tokens
    assert (
        PrimeRlServingTokens.serve_tokens_full_generator
        is not PrimeRlServingTokens.__mro__[1].serve_tokens_full_generator
    )


def test_serialize_routed_experts_uses_compact_raw_payload():
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )

    encoded = serialize_routed_experts(routed_experts)
    assert encoded is not None

    decoded = _decode_routed_experts(encoded)
    assert decoded.dtype == np.uint8
    np.testing.assert_array_equal(decoded, routed_experts)


def test_generate_response_post_process_replaces_upstream_routed_experts():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _GenerateRoutedExpertsCapture(_empty_request_outputs())
    capture.routed_experts[0] = compact_routed_experts
    response = GenerateResponse(
        request_id="request-id",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts


def test_client_set_max_tokens_recognizes_explicit_value():
    body = {"token_ids": [1, 2, 3], "sampling_params": {"max_tokens": 256}}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body))) is True


def test_client_set_max_tokens_detects_unset():
    body = {"token_ids": [1, 2, 3], "sampling_params": {}}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body))) is False

    body_without_sp = {"token_ids": [1, 2, 3]}
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(body_without_sp))) is False


class _FakeOutput:
    def __init__(self, token_ids):
        self.token_ids = token_ids


class _FakeRequestOutput:
    """Minimal stand-in for ``vllm.outputs.RequestOutput``.

    ``_build_usage`` only touches four attributes; constructing a real
    ``RequestOutput`` would require a full ``CompletionOutput`` graph and
    isn't worth it for a serialization-shape test.
    """

    def __init__(self, prompt_token_ids, output_token_ids_list, num_cached_tokens=0, encoder_prompt_token_ids=None):
        self.prompt_token_ids = prompt_token_ids
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.outputs = [_FakeOutput(t) for t in output_token_ids_list]
        self.num_cached_tokens = num_cached_tokens


def test_prime_rl_generate_response_serializes_usage_block():
    # Regression for prime-rl PR #2408: parent ``GenerateResponse`` doesn't
    # declare ``usage``, so the field must be declared on the subclass for
    # Pydantic to emit it in JSON. Without this the router can't extract
    # per-run token / cache counts for billing.
    response = PrimeRlGenerateResponse(
        request_id="req-1",
        choices=[PrimeRlGenerateResponseChoice(index=0, token_ids=[1, 2, 3])],
        usage=UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7),
    )
    payload = response.model_dump(mode="json")
    assert payload["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 3,
        "total_tokens": 7,
        "prompt_tokens_details": None,
    }


def test_build_usage_sums_prompt_and_completion_tokens():
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3, 4, 5],
        output_token_ids_list=[[10, 11], [20, 21, 22]],
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens == 5
    assert usage.completion_tokens == 5  # 2 + 3
    assert usage.total_tokens == 10
    assert usage.prompt_tokens_details is None


def test_build_usage_includes_encoder_prompt_tokens():
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3],
        output_token_ids_list=[[10]],
        encoder_prompt_token_ids=[100, 101],
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens == 5  # 3 + 2
    assert usage.total_tokens == 6


def test_build_usage_reports_cached_tokens_unconditionally():
    # Unlike upstream's ``enable_prompt_tokens_details`` gate, prime-rl always
    # surfaces cached tokens — the cache-discount billing pipeline needs them.
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3, 4],
        output_token_ids_list=[[10, 11]],
        num_cached_tokens=3,
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 3


def test_build_usage_skips_cached_tokens_when_zero():
    # Don't emit a details block with cached=0, which would be misleading
    # to the router's billing extractor.
    final_res = _FakeRequestOutput(
        prompt_token_ids=[1, 2, 3, 4],
        output_token_ids_list=[[10, 11]],
        num_cached_tokens=0,
    )
    usage = _build_usage(final_res)
    assert usage.prompt_tokens_details is None


def test_final_output_capture_records_last_item():
    async def _gen():
        for r in [
            _FakeRequestOutput(prompt_token_ids=[1], output_token_ids_list=[[1]]),
            _FakeRequestOutput(prompt_token_ids=[1, 2], output_token_ids_list=[[1, 2]]),
            _FakeRequestOutput(prompt_token_ids=[1, 2, 3], output_token_ids_list=[[1, 2, 3]]),
        ]:
            yield r

    async def _drain(capture):
        async for _ in capture:
            pass

    capture = _FinalOutputCapture(_gen())
    asyncio.run(_drain(capture))
    assert capture.final_res is not None
    assert capture.final_res.prompt_token_ids == [1, 2, 3]


def test_final_output_capture_works_over_async_def_aiter_source():
    # ``_GenerateRoutedExpertsCapture`` exposes the async-iterator protocol
    # via ``async def __aiter__`` (an async generator function) and has no
    # ``__anext__``. The wrapper must drive it through ``async for`` rather
    # than poking ``__anext__`` directly, or routed-experts runs raise
    # AttributeError before the response is built.

    class _AsyncGenAiterSource:
        def __init__(self, items):
            self._items = items

        async def __aiter__(self):
            for item in self._items:
                yield item

    items = [
        _FakeRequestOutput(prompt_token_ids=[1], output_token_ids_list=[[1]]),
        _FakeRequestOutput(prompt_token_ids=[1, 2], output_token_ids_list=[[1, 2]]),
    ]
    capture = _FinalOutputCapture(_AsyncGenAiterSource(items))

    async def _drain():
        async for _ in capture:
            pass

    asyncio.run(_drain())
    assert capture.final_res is not None
    assert capture.final_res.prompt_token_ids == [1, 2]


def test_final_output_capture_handles_empty_stream():
    capture = _FinalOutputCapture(_empty_request_outputs())

    async def _drain():
        async for _ in capture:
            pass

    asyncio.run(_drain())
    assert capture.final_res is None


def test_client_set_max_tokens_assumes_set_when_body_unreadable():
    # No raw_request → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(None)) is True

    # body read raises → can't tell, don't override.
    err = ValueError("bad json")
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(err))) is True

    # non-dict body → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest([1, 2, 3]))) is True


def test_materialize_raw_image_ref_uses_generic_family_payload(tmp_path, monkeypatch):
    from PIL import Image
    from renderers.mm_store import raw_mm_ref

    from prime_rl.inference.vllm import serving_tokens

    image_dir = tmp_path / "run_serving" / "assets" / "images"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "image.png"
    Image.new("RGB", (8, 6), color=(32, 64, 128)).save(image_path)
    monkeypatch.setenv("VF_RENDERER_IMAGE_OFFLOAD_DIR", str(image_dir))

    mm_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()[:32]
    fingerprint = "f" * 32
    raw_ref = raw_mm_ref(
        run_id="serving",
        family="test_family",
        fingerprint=fingerprint,
        modality="image",
        mm_hash=mm_hash,
        raw_image_id=image_path.name,
        payload={"adapter_owned": [1, 2, 3]},
    )
    processor = object()
    captured = {}

    class _Adapter:
        def materialize_for_vllm(self, image_processor, item, image, expected_placeholder_length):
            captured["image_processor"] = image_processor
            captured["item"] = item
            captured["image_size"] = image.size
            captured["expected_placeholder_length"] = expected_placeholder_length
            return {"materialized": True}

    def _get_adapter(family):
        captured["family"] = family
        return _Adapter()

    monkeypatch.setattr(serving_tokens, "_load_image_processor", lambda _model, _trust: processor)
    monkeypatch.setattr(serving_tokens, "get_multimodal_adapter", _get_adapter)

    out = serving_tokens._materialize_raw_image_ref_sync(
        raw_ref,
        feature_modality="image",
        mm_hash=mm_hash,
        expected_placeholder_length=7,
        processor_model_name="model",
        trust_remote_code=True,
    )

    assert out == {"materialized": True}
    assert captured["family"] == "test_family"
    assert captured["image_processor"] is processor
    assert captured["image_size"] == (8, 6)
    assert captured["expected_placeholder_length"] == 7
    item = captured["item"]
    assert item.family == "test_family"
    assert item.layout_fingerprint == fingerprint
    assert item.payload == {"adapter_owned": [1, 2, 3]}


def test_materialize_raw_image_ref_accepts_inline_data_uri(tmp_path, monkeypatch):
    from PIL import Image
    from renderers.mm_store import raw_mm_ref

    from prime_rl.inference.vllm import serving_tokens

    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 6), color=(16, 32, 48)).save(image_path)
    raw = image_path.read_bytes()
    data_uri = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"

    mm_hash = hashlib.sha256(raw).hexdigest()[:32]
    fingerprint = "f" * 32
    raw_ref = raw_mm_ref(
        run_id="serving",
        family="test_family",
        fingerprint=fingerprint,
        modality="image",
        mm_hash=mm_hash,
        raw_uri=data_uri,
        payload={"adapter_owned": [4, 5, 6]},
    )
    processor = object()
    captured = {}

    class _Adapter:
        def materialize_for_vllm(self, image_processor, item, image, expected_placeholder_length):
            captured["image_processor"] = image_processor
            captured["item"] = item
            captured["image_size"] = image.size
            captured["expected_placeholder_length"] = expected_placeholder_length
            return {"materialized": True}

    monkeypatch.setattr(serving_tokens, "_load_image_processor", lambda _model, _trust: processor)
    monkeypatch.setattr(serving_tokens, "get_multimodal_adapter", lambda _family: _Adapter())

    out = serving_tokens._materialize_raw_image_ref_sync(
        raw_ref,
        feature_modality="image",
        mm_hash=mm_hash,
        expected_placeholder_length=7,
        processor_model_name="model",
        trust_remote_code=True,
    )

    assert out == {"materialized": True}
    assert captured["image_processor"] is processor
    assert captured["image_size"] == (8, 6)
    assert captured["expected_placeholder_length"] == 7
    assert captured["item"].payload == {"adapter_owned": [4, 5, 6]}


def test_decode_raw_mm_kwargs_rejects_none_items():
    features = SimpleNamespace(
        mm_hashes={"image": ["a" * 32]},
        mm_placeholders={"image": [SimpleNamespace(length=1)]},
        kwargs_data={"image": [None]},
    )

    with pytest.raises(_MMImageRefError, match="raw descriptor refs"):
        asyncio.run(
            _decode_raw_mm_kwargs(
                features,
                processor_model_name="model",
                trust_remote_code=True,
            )
        )


def test_decode_raw_mm_kwargs_requires_parallel_placeholders():
    features = SimpleNamespace(
        mm_hashes={"image": ["a" * 32]},
        mm_placeholders={"image": []},
        kwargs_data={"image": ["mmraw:v2:anything"]},
    )

    with pytest.raises(_MMImageRefError, match="placeholder/hash length mismatch"):
        asyncio.run(
            _decode_raw_mm_kwargs(
                features,
                processor_model_name="model",
                trust_remote_code=True,
            )
        )
