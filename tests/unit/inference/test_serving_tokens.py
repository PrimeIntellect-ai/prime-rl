from __future__ import annotations

import asyncio
import io

import numpy as np
import pybase64
import pytest
from pydantic import Field
from vllm.entrypoints.openai.engine.protocol import RequestResponseMetadata, UsageInfo
from vllm.sampling_params import SamplingParams

from prime_rl.inference.vllm.compat import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    ServingTokens,
)
from prime_rl.inference.vllm.routed_experts import (
    compact_vllm_routed_experts,
    serialize_routed_experts,
)
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    build_prime_serving_tokens,
)


def _decode_compact(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


def _encode_vllm(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, array)
    return pybase64.b64encode(buffer.getvalue()).decode("ascii")


def test_routed_experts_round_trip_both_wire_formats():
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )

    compact = serialize_routed_experts(routed_experts, start=2)
    converted = compact_vllm_routed_experts(_encode_vllm(routed_experts), start=2)

    assert compact is not None
    assert converted is not None
    assert converted["start"] == 2
    np.testing.assert_array_equal(_decode_compact(compact), routed_experts)
    np.testing.assert_array_equal(_decode_compact(converted), routed_experts)


def test_serve_tokens_forwards_kv_transfer_params_without_mutating_request(monkeypatch):
    expected = object()
    observed_request = None

    async def upstream(_self, request, _raw_request=None):
        nonlocal observed_request
        observed_request = request
        assert request.sampling_params.extra_args == {
            "existing": True,
            "kv_transfer_params": {"remote": "metadata"},
        }
        return expected

    monkeypatch.setattr(ServingTokens, "serve_tokens", upstream)
    server = object.__new__(PrimeRlServingTokens)
    request = GenerateRequest(
        token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1, extra_args={"existing": True}),
        kv_transfer_params={"remote": "metadata"},
    )

    assert asyncio.run(server.serve_tokens(request)) is expected
    assert observed_request is not request
    assert request.sampling_params.extra_args == {"existing": True}


class _FutureGenerateResponseChoice(GenerateResponseChoice):
    future_choice_field: str | None = None


class _FutureGenerateResponse(GenerateResponse):
    model: str | None = None
    created: int | None = None
    usage: UsageInfo | None = None
    future_response_field: dict[str, int] = Field(default_factory=dict)


def test_full_generator_preserves_all_upstream_response_fields(monkeypatch):
    routed_experts = np.array([[[1, 2, 3]]], dtype=np.uint8)
    usage = UsageInfo(
        prompt_tokens=3,
        completion_tokens=2,
        total_tokens=5,
        prompt_tokens_details={"cached_tokens": 2},
    )
    upstream_response = _FutureGenerateResponse(
        request_id="canonical-request-id",
        model="served-model",
        created=123456789,
        usage=usage,
        future_response_field={"version": 2},
        choices=[
            _FutureGenerateResponseChoice(
                index=0,
                token_ids=[10, 11],
                routed_experts=_encode_vllm(routed_experts),
                future_choice_field="retained",
            )
        ],
    )

    async def upstream(_self, _request, result_generator, *_args):
        async for _ in result_generator:
            pass
        return upstream_response

    monkeypatch.setattr(ServingTokens, "serve_tokens_full_generator", upstream)
    server = object.__new__(PrimeRlServingTokens)
    server.enable_prompt_tokens_details = True
    request = GenerateRequest(
        token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=2, routed_experts_prompt_start=1),
    )

    async def outputs():
        if False:
            yield

    response = asyncio.run(
        server.serve_tokens_full_generator(
            request,
            outputs(),
            "input-request-id",
            "input-model",
            RequestResponseMetadata(request_id="input-request-id"),
        )
    )

    assert response.request_id == "canonical-request-id"
    assert response.model == "served-model"
    assert response.created == 123456789
    assert response.usage == usage
    assert response.future_response_field == {"version": 2}
    assert response.choices[0].future_choice_field == "retained"
    encoded = response.choices[0].routed_experts
    assert isinstance(encoded, dict)
    assert encoded["start"] == 1
    np.testing.assert_array_equal(_decode_compact(encoded), routed_experts)


def test_full_generator_fills_fields_missing_from_legacy_response(monkeypatch):
    upstream_response = GenerateResponse(
        request_id="canonical-request-id",
        choices=[GenerateResponseChoice(index=0, token_ids=[10])],
    )

    async def upstream(_self, _request, result_generator, *_args):
        async for _ in result_generator:
            pass
        return upstream_response

    monkeypatch.setattr(ServingTokens, "serve_tokens_full_generator", upstream)
    server = object.__new__(PrimeRlServingTokens)
    request = GenerateRequest(
        token_ids=[1],
        sampling_params=SamplingParams(max_tokens=1),
    )

    async def outputs():
        if False:
            yield

    response = asyncio.run(
        server.serve_tokens_full_generator(
            request,
            outputs(),
            "input-request-id",
            "served-model",
            RequestResponseMetadata(request_id="input-request-id"),
        )
    )

    assert response.model == "served-model"
    assert isinstance(response.created, int)


@pytest.mark.parametrize("renderer_attribute", ["online_renderer", "openai_serving_render"])
def test_build_prime_serving_tokens_detects_renderer_independently_of_import_layout(
    monkeypatch, renderer_attribute
):
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(PrimeRlServingTokens, "__init__", fake_init)
    renderer = object()
    attributes = {
        "engine_client": object(),
        "models": object(),
        renderer_attribute: renderer,
        "request_logger": object(),
        "return_tokens_as_token_ids": True,
        "force_no_detokenize": True,
        "enable_log_outputs": True,
    }
    upstream = type("UpstreamServingTokens", (), attributes)()

    replacement = build_prime_serving_tokens(upstream)

    assert type(replacement) is PrimeRlServingTokens
    assert captured["args"] == (
        upstream.engine_client,
        upstream.models,
        renderer,
    )
    assert captured["kwargs"] == {
        "request_logger": upstream.request_logger,
        "return_tokens_as_token_ids": True,
        "force_no_detokenize": True,
        "enable_prompt_tokens_details": True,
        "enable_log_outputs": True,
    }
