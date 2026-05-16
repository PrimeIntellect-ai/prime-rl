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

import numpy as np
import pybase64
from vllm.entrypoints.serve.disagg.protocol import GenerateResponse, GenerateResponseChoice

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _client_set_max_tokens,
    _GenerateRoutedExpertsCapture,
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
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3]}
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


def test_client_set_max_tokens_assumes_set_when_body_unreadable():
    # No raw_request → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(None)) is True

    # body read raises → can't tell, don't override.
    err = ValueError("bad json")
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest(err))) is True

    # non-dict body → can't tell, don't override.
    assert asyncio.run(_client_set_max_tokens(_FakeRawRequest([1, 2, 3]))) is True
