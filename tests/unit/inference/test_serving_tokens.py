"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM 0.20's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``_encode_routed_experts`` round-trips a numpy array as expected.
    * ``PrimeRlGenerateResponseChoice`` accepts the optional field.
    * The subclass attaches its overrides without monkey-patching the parent.
    * ``_client_set_max_tokens`` distinguishes raw-body shapes correctly.
"""

from __future__ import annotations

import asyncio
import base64

import numpy as np

from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlGenerateResponse,
    PrimeRlGenerateResponseChoice,
    PrimeRlServingTokens,
    _client_set_max_tokens,
    _encode_routed_experts,
)


class _FakeRawRequest:
    def __init__(self, body):
        self._body = body
        self._raise = isinstance(body, Exception)

    async def json(self):
        if self._raise:
            raise self._body
        return self._body


def test_encode_routed_experts_roundtrip():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    encoded = _encode_routed_experts(arr)

    assert encoded["shape"] == [2, 3]
    decoded = np.frombuffer(base64.b85decode(encoded["data"]), dtype=np.int32).reshape(encoded["shape"])
    np.testing.assert_array_equal(decoded, arr)


def test_routed_experts_choice_accepts_none_and_dict():
    no_re = PrimeRlGenerateResponseChoice(index=0, finish_reason="stop", token_ids=[1, 2])
    assert no_re.routed_experts is None

    encoded = _encode_routed_experts(np.zeros((1, 1), dtype=np.int32))
    with_re = PrimeRlGenerateResponseChoice(index=0, finish_reason="stop", token_ids=[1], routed_experts=encoded)
    assert with_re.routed_experts == encoded


def test_response_only_serializes_declared_fields():
    # Upstream silently drops id=/created=/model=/usage= because they're not
    # declared on GenerateResponse. Our subclass adds nothing to that surface
    # — it only widens the choices type — so the JSON shape stays slim.
    resp = PrimeRlGenerateResponse(
        request_id="gen-x",
        choices=[PrimeRlGenerateResponseChoice(index=0, finish_reason="stop", token_ids=[7])],
    )
    dumped = resp.model_dump()
    assert set(dumped.keys()) == {
        "request_id",
        "choices",
        "prompt_logprobs",
        "kv_transfer_params",
    }
    assert dumped["choices"][0]["routed_experts"] is None


def test_subclass_inherits_serve_tokens_full_generator():
    # The subclass adds an override; make sure we didn't accidentally rebind
    # ``serve_tokens`` to a parent attribute via __dict__-update tricks later.
    assert (
        PrimeRlServingTokens.serve_tokens_full_generator
        is not PrimeRlServingTokens.__mro__[1].serve_tokens_full_generator
    )
    assert PrimeRlServingTokens.serve_tokens is not PrimeRlServingTokens.__mro__[1].serve_tokens


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
