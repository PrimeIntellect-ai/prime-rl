"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass overrides ``serve_tokens_full_generator`` without
      monkey-patching the parent.
    * ``post_process`` swaps in the packed payloads while preserving
      the rest of the upstream response (``usage`` included).
    * The sampled-token logprobs pack from vLLM's flat logprobs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pybase64
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateResponse,
    GenerateResponseChoice,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.logprobs import FlatLogprobs, Logprob

from prime_rl.inference.vllm.routed_experts import serialize_routed_experts
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _PackedOutputs,
    pack_sampled_logprobs,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=np.uint8,
    ).reshape(encoded["shape"])


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_overrides_serve_tokens_full_generator():
    upstream = PrimeRlServingTokens.__mro__[1]
    assert PrimeRlServingTokens.serve_tokens_full_generator is not upstream.serve_tokens_full_generator


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


def test_generate_response_post_process_preserves_prompt_metadata():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _PackedOutputs(
        _empty_request_outputs(), SimpleNamespace(sampling_params=SimpleNamespace(routed_experts_prompt_start=0))
    )
    capture.fields[0] = {"routed_experts": compact_routed_experts}
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
        prompt_token_ids=[10, 11, 12, 13],
        mm_placeholders={"image": [PlaceholderRangeInfo(offset=1, length=2)]},
    )

    processed = capture.post_process(response)

    assert processed.choices[0].routed_experts == compact_routed_experts
    assert processed.model == "test-model"
    assert processed.usage == usage
    payload = processed.model_dump(mode="json")
    assert payload["choices"][0]["routed_experts"] == compact_routed_experts
    assert payload["prompt_token_ids"] == [10, 11, 12, 13]
    assert payload["mm_placeholders"] == {"image": [{"offset": 1, "length": 2}]}
    assert payload["usage"]["total_tokens"] == 7


def test_pack_sampled_logprobs_takes_first_entry_and_clamps():
    logprobs = FlatLogprobs()
    logprobs.append({5: Logprob(-0.5, 2, None), 7: Logprob(-0.1, 1, None)})  # sampled token first
    logprobs.append(None)
    logprobs.append({3: Logprob(float("-inf"), 9, None)})

    np.testing.assert_array_equal(pack_sampled_logprobs(logprobs), np.array([-0.5, -9999.0, -9999.0], np.float32))
