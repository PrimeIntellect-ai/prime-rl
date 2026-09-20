"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass attaches its overrides without monkey-patching the parent.
    * Compact logprobs preserve sampled-token evidence and top-k ordering.
    * ``post_process`` swaps in the compact routed_experts while preserving
      the rest of the upstream response (``usage`` included).
"""

from __future__ import annotations

import json

import numpy as np
import pybase64
import pytest
from renderers.client import MalformedGenerateResponseError, _parse_compact_logprobs, parse_generate_response
from vllm.entrypoints.openai.engine.protocol import UsageInfo
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateResponse, GenerateResponseChoice
from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
from vllm.logprobs import FlatLogprobs, append_logprobs_for_next_position

from prime_rl.inference.vllm.compact_logprobs import serialize_compact_logprobs
from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _GenerateOutputCapture,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


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
    capture = _GenerateOutputCapture(_empty_request_outputs())
    capture.routed_experts[0] = compact_routed_experts
    usage = UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7)
    response = GenerateResponse(
        request_id="request-id",
        model="test-model",
        choices=[
            GenerateResponseChoice(
                index=0,
                token_ids=[1, 2, 3],
                routed_experts="upstream-npy-payload",
            )
        ],
        usage=usage,
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts
    assert processed.model == "test-model"
    assert processed.usage == usage
    # The compact object form must survive JSON serialization (the parent
    # declares ``routed_experts`` as a base64 string).
    payload = processed.model_dump(mode="json")
    assert payload["choices"][0]["routed_experts"] == compact_routed_experts
    assert payload["usage"]["total_tokens"] == 7


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("sampled_ids", [[4, 9], [9, 2]])
def test_compact_logprobs_preserve_legacy_scores(flat, sampled_ids):
    rows = FlatLogprobs() if flat else []
    for sampled in sampled_ids:
        ids = [sampled, 2, 4, 6]
        values = [float(np.float32(-sampled / 10)), *np.asarray([-0.2, -0.4, -0.6], dtype=np.float32).tolist()]
        append_logprobs_for_next_position(rows, ids, values, [None] * 4, 1, 3)
    legacy = ServingTokens._create_tokens_logprobs(None, sampled_ids, rows, 3).model_dump()
    packed = serialize_compact_logprobs(rows, 3)
    payload = {
        "choices": [
            {"compact_logprobs": packed, "routed_experts": {"data": "AQ=="}},
            {"compact_logprobs": packed, "routed_experts": {"data": "Ag=="}},
        ]
    }
    parsed = parse_generate_response(json.dumps(payload, separators=(",", ":")).encode())
    for choice in parsed["choices"]:
        assert isinstance(choice["compact_logprobs"]["data"], memoryview)
        assert isinstance(choice["routed_experts"]["data"], memoryview)
        sampled, heads = _parse_compact_logprobs(choice, sampled_ids, 2)
        assert sampled == [row["logprob"] for row in legacy["content"]]
        for head, row in zip(heads, legacy["content"]):
            expected = sorted(row["top_logprobs"], key=lambda item: item["logprob"], reverse=True)[:2]
            assert head == (
                [int(item["token"].removeprefix("token_id:")) for item in expected],
                [item["logprob"] for item in expected],
            )


def test_compact_logprobs_empty_and_sliced_buffers():
    packed = serialize_compact_logprobs(FlatLogprobs(), 3)
    assert _parse_compact_logprobs({"compact_logprobs": packed}, [], 2) == ([], [])
    rows = FlatLogprobs()
    for token_id in range(5):
        append_logprobs_for_next_position(rows, [token_id], [-0.5], [None], 1, 0)
    packed = serialize_compact_logprobs(rows[1:5:2], 0)
    assert _parse_compact_logprobs({"compact_logprobs": packed}, [1, 3], 0) == ([-0.5, -0.5], None)


@pytest.mark.parametrize(
    "change",
    [
        {"format": "unknown"},
        {"offsets": [0]},
        {"offsets": [1, 1]},
        {"offsets": [0, 0]},
        {"offsets": [0, 2]},
        {"offsets": [0, True]},
        {"data": "invalid!"},
        {"num_top_logprobs": None},
    ],
)
def test_compact_logprobs_reject_malformed_buffers(change):
    rows = FlatLogprobs()
    append_logprobs_for_next_position(rows, [7], [-0.5], [None], 1, 0)
    packed = serialize_compact_logprobs(rows, 0)
    with pytest.raises(MalformedGenerateResponseError, match="sampled-token evidence"):
        _parse_compact_logprobs({"compact_logprobs": packed}, [8], 0)
    with pytest.raises(MalformedGenerateResponseError, match="Incomplete"):
        _parse_compact_logprobs({"compact_logprobs": packed}, [7], 2)
    packed.update(change)
    with pytest.raises(MalformedGenerateResponseError):
        _parse_compact_logprobs({"compact_logprobs": packed}, [7], 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), -9999.0, 0.5])
def test_compact_logprobs_reject_invalid_sampling_evidence(value):
    rows = FlatLogprobs()
    append_logprobs_for_next_position(rows, [7], [value], [None], 1, 0)
    packed = serialize_compact_logprobs(rows, 0)
    with pytest.raises(MalformedGenerateResponseError):
        _parse_compact_logprobs({"compact_logprobs": packed}, [7], 0)
    with pytest.raises(MalformedGenerateResponseError):
        _parse_compact_logprobs({"compact_logprobs": packed}, [8], 0)
