"""Sanity tests for the prime-RL ``ServingTokens`` subclass.

The full happy-path is owned upstream by vLLM's
``vllm/entrypoints/serve/disagg`` test suite. We only cover the prime-RL
deltas here:
    * ``serialize_routed_experts`` round-trips a compact raw-byte payload.
    * The subclass overrides ``serve_tokens_full_generator`` without
      monkey-patching the parent.
    * ``post_process`` swaps in the compact routed_experts while preserving
      the rest of the upstream response (``usage`` included).
    * Numeric logprob buffers preserve sampled/head evidence and replay masks
      through the real upstream response formatter and renderer decoder.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from fastapi.responses import JSONResponse
from renderers.client import (
    _parse_compact_logprobs,
    _parse_completion_logprobs,
    _parse_completion_top_logprobs,
    parse_generate_response,
)
from vllm import SamplingParams
from vllm.entrypoints.generate.base.protocol import RequestResponseMetadata
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateRequest,
    GenerateResponse,
    GenerateResponseChoice,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.outputs import CompletionOutput, RequestOutput, SamplingMask

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


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("selection", [slice(None), slice(1, 3), slice(None, None, 2), slice(0, 0)])
def test_compact_logprobs_roundtrip_ragged_slices(flat, selection):
    rows = [{7: Logprob(-0.5)}, {8: Logprob(-1.5), 9: Logprob(float("-inf"))}, {10: Logprob(-2.5)}]
    logprobs = FlatLogprobs() if flat else []
    logprobs.extend(rows)
    encoded = serialize_compact_logprobs(logprobs[selection], 2)
    raw = pybase64.b64decode(encoded["data"])
    selected = rows[selection]
    count = sum(map(len, selected))
    assert encoded["format"] == "u32-f32-le-v1"
    assert encoded["num_top_logprobs"] == 2
    assert encoded["offsets"] == [0, *np.cumsum([len(row) for row in selected]).tolist()]
    np.testing.assert_array_equal(np.frombuffer(raw, dtype="<u4", count=count), [i for row in selected for i in row])
    np.testing.assert_array_equal(
        np.frombuffer(raw, dtype="<f4", offset=count * 4),
        [max(v.logprob, -9999.0) for row in selected for v in row.values()],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("routed", [False, True])
async def test_compact_formatter_matches_native_evidence_and_metadata(flat, replay, routed):
    serving = object.__new__(PrimeRlServingTokens)
    serving.model_config = SimpleNamespace(enable_return_routed_experts=routed)
    serving.enable_prompt_tokens_details = True
    serving.enable_log_outputs = False
    sampled_ids = [2, 3] if replay else [2, 7]
    probabilities = {1: 0.6, 2: 0.2, 3: 0.2, 4: 0.0} if replay else {1: 0.6, 2: 0.25, 3: 0.1, 4: 0.04, 7: 0.01}
    with np.errstate(divide="ignore"):
        logq = {i: float(np.float32(np.log(q))) for i, q in probabilities.items()}
    logprobs = FlatLogprobs() if flat else []
    for sampled in sampled_ids:
        ids = [sampled, 1, 2, 3, 4]
        if flat:
            logprobs.append_fast(ids, [logq[i] for i in ids], iter(range(len(ids))), [None] * len(ids))
        else:
            logprobs.append({i: Logprob(logq[i]) for i in ids})
    support = SamplingMask([[3, 1, 2], [1, 3, 2]]) if replay else None
    experts = np.array([[[1, 2]], [[3, 4]]], dtype=np.int64) if routed else None
    output = RequestOutput(
        "test",
        None,
        [10, 11],
        [None, {11: Logprob(-0.5)}],
        [
            CompletionOutput(
                i,
                "",
                sampled_ids,
                None,
                logprobs,
                routed_experts=experts,
                sampling_mask=support,
                finish_reason="length",
            )
            for i in range(2)
        ],
        True,
        num_cached_tokens=1,
        kv_transfer_params={"kv": "kept"},
        ec_transfer_params={"ec": "kept"},
    )

    async def generate_outputs():
        yield output

    responses = []
    evidence = []
    for compact in (False, True):
        params = SamplingParams(logprobs=4, extra_args={"prl_compact_logprobs": True} if compact else None)
        request = GenerateRequest(model="test", token_ids=[10, 11], sampling_params=params, return_token_ids=True)
        request._response_mm_placeholders = {"image": [PlaceholderRangeInfo(offset=0, length=1)]}
        metadata = RequestResponseMetadata(request_id="test")
        response = await serving.serve_tokens_full_generator(request, generate_outputs(), "test", "test", metadata)
        assert isinstance(response, GenerateResponse)
        assert params.logprobs == 4
        assert request.sampling_params is params
        assert output.outputs[0].logprobs is logprobs
        assert response.usage == metadata.final_usage_info
        payload = parse_generate_response(JSONResponse(content=response.model_dump()).body, compact_logprobs=compact)
        for choice in payload["choices"]:
            if compact:
                assert choice["logprobs"] is None
                assert isinstance(choice["compact_logprobs"]["data"], memoryview)
                evidence.append(_parse_compact_logprobs(choice, sampled_ids, 2))
            else:
                assert choice.get("compact_logprobs") is None
                evidence.append(
                    (
                        _parse_completion_logprobs(choice, sampled_ids),
                        *_parse_completion_top_logprobs(choice, sampled_ids, 2),
                    )
                )
            choice.pop("logprobs")
            choice.pop("compact_logprobs", None)
        payload.pop("created")
        responses.append(payload)
    assert all(item == evidence[0] for item in evidence)
    assert evidence[0][1] == ([[1, 2, 3]] * 2 if replay else [[1, 2]] * 2)
    assert responses[0] == responses[1]
    assert responses[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert responses[1]["kv_transfer_params"] == {"kv": "kept"}
    assert responses[1]["ec_transfer_params"] == {"ec": "kept"}
