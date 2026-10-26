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

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import torch
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
    GenerateTokensChoice,
    GenerateTokensResponse,
    PlaceholderRangeInfo,
)
from vllm.entrypoints.serve.engine.protocol import UsageInfo
from vllm.logprobs import FlatLogprobs, Logprob
from vllm.outputs import CompletionOutput, RequestOutput

from prime_rl.inference.patches import PackedSamplingMask
from prime_rl.inference.vllm.paired_capture import NEG_INF_BITS, pack_sampler_rows, serialize_paired_support
from prime_rl.inference.vllm.serving_tokens import (
    PrimeRlServingTokens,
    _PackedOutputs,
    encode_array,
    pack_sampled_logprobs,
)


def _decode_routed_experts(encoded: dict) -> np.ndarray:
    return np.frombuffer(
        pybase64.b64decode_as_bytearray(encoded["data"]),
        dtype=encoded["dtype"],
    ).reshape(encoded["shape"])


async def _empty_request_outputs():
    if False:
        yield


def test_subclass_overrides_serve_tokens_full_generator():
    upstream = PrimeRlServingTokens.__mro__[1]
    assert PrimeRlServingTokens.serve_tokens_full_generator is not upstream.serve_tokens_full_generator


def test_serialize_routed_experts_keeps_the_capture_dtype():
    # A >256-expert model captures uint16 even when every id of a request is below 256.
    routed_experts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.uint16,
    )

    encoded = encode_array(routed_experts)
    assert encoded is not None

    decoded = _decode_routed_experts(encoded)
    assert decoded.dtype == np.uint16
    np.testing.assert_array_equal(decoded, routed_experts)


def test_generate_response_post_process_preserves_prompt_metadata():
    compact_routed_experts = {"data": "AQID", "shape": [1, 1, 3], "start": 0}
    capture = _PackedOutputs(
        _empty_request_outputs(), GenerateRequest(model="test", token_ids=[1], sampling_params=SamplingParams())
    )
    capture.fields[0] = {"routed_experts": compact_routed_experts}
    usage = UsageInfo(prompt_tokens=4, completion_tokens=3, total_tokens=7)
    response = GenerateTokensResponse(
        request_id="request-id",
        model="test-model",
        choices=[
            GenerateTokensChoice(
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


def _gpu_capture(logq: dict[int, float], sampled_ids: list[int], replay: bool, draws: bool = False):
    """The packed int64 mask the patched sampler hands to the server (pinned host copies need CUDA)."""
    logits = torch.full((len(sampled_ids), 8), -torch.inf)
    for i, q in logq.items():
        logits[:, i] = q
    if replay:
        ids = torch.isfinite(logits).nonzero()[:, 1].view(len(sampled_ids), -1)
        values = logits.gather(1, ids).log_softmax(-1)
        packed = ((values.contiguous().view(torch.int32) ^ NEG_INF_BITS).long() << 32) | ids
    else:
        cu, num_sampled = torch.arange(len(sampled_ids) + 1), torch.ones(len(sampled_ids), dtype=torch.int32)
        sampled = torch.tensor(sampled_ids).view(-1, 1)
        rows = np.full(len(sampled_ids), 4)
        _, packed, _ = pack_sampler_rows(logits, cu, num_sampled, sampled, rows, np.full(len(sampled_ids), draws))
    return PackedSamplingMask(
        ids=packed.flatten().numpy(), counts=np.full(len(sampled_ids), packed.shape[1], dtype=np.int32)
    )


@pytest.mark.gpu
@pytest.mark.asyncio
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("routed", [False, True])
async def test_gpu_capture_matches_native_evidence_and_metadata(flat, replay, routed):
    serving = object.__new__(PrimeRlServingTokens)
    serving.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(aux_output_config=SimpleNamespace(enable_return_routed_experts=routed))
    )
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
    support = (
        PackedSamplingMask(ids=np.array([1, 2, 3, 1, 2, 3], dtype=np.int32), counts=np.array([3, 3], dtype=np.int32))
        if replay
        else None
    )
    experts = np.array([[[1, 2]], [[3, 4]]], dtype=np.uint8) if routed else None

    def request_output(logprobs, mask):
        return RequestOutput(
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
                    sampling_mask=deepcopy(mask),
                    finish_reason="length",
                )
                for i in range(2)
            ],
            True,
            num_cached_tokens=1,
            kv_transfer_params={"kv": "kept"},
            ec_transfer_params={"ec": "kept"},
        )

    responses = []
    evidence = []
    for captured in (False, True):
        output = (
            request_output(None, _gpu_capture(logq, sampled_ids, replay))
            if captured
            else request_output(logprobs, support)
        )

        async def generate_outputs():
            yield deepcopy(output)

        extra_args = {"prl_paired_capture": True} if replay else {"prl_sampler_rows": 4}
        params = SamplingParams(logprobs=None if captured else 4, extra_args=extra_args if captured else None)
        request = GenerateRequest(model="test", token_ids=[10, 11], sampling_params=params, return_token_ids=True)
        request._response_mm_placeholders = {"image": [PlaceholderRangeInfo(offset=0, length=1)]}
        metadata = RequestResponseMetadata(request_id="test")
        response = await serving.serve_tokens_full_generator(request, generate_outputs(), "test", "test", metadata)
        assert isinstance(response, GenerateTokensResponse)
        assert request.sampling_params is params
        assert response.usage == metadata.final_usage_info
        payload = parse_generate_response(JSONResponse(content=response.model_dump()).body)
        for choice in payload["choices"]:
            if captured:
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
    evidence = [(np.round(logprobs, 6).tolist(), *heads) for logprobs, *heads in evidence]
    assert all(item[0] == evidence[0][0] and item[1] == evidence[0][1] for item in evidence)
    np.testing.assert_allclose(np.array(evidence[-1][2]), np.array(evidence[0][2]), atol=1e-6)
    assert evidence[0][1] == ([[1, 2, 3]] * 2 if replay else [[1, 2]] * 2)
    assert responses[0] == responses[1]
    assert responses[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert responses[1]["kv_transfer_params"] == {"kv": "kept"}
    assert responses[1]["ec_transfer_params"] == {"ec": "kept"}


@pytest.mark.gpu
def test_gpu_capture_sampler_draws_round_trip():
    logq = {1: float(np.log(0.6)), 2: float(np.log(0.3)), 5: float(np.log(0.1))}
    mask = _gpu_capture(logq, [2, 5], replay=False, draws=True)
    ids, compact = serialize_paired_support(mask.ids, mask.counts)
    raw = pybase64.b64decode(compact["data"])
    values = np.frombuffer(raw, dtype="<f4", offset=len(ids) * 4)
    assert compact["offsets"] == [0, 5, 10] and ids[0] == 2 and ids[5] == 5
    np.testing.assert_allclose(values, [logq[i] for i in ids], atol=1e-6)


def test_pack_sampled_logprobs_takes_first_entry_and_clamps():
    logprobs = FlatLogprobs()
    logprobs.append({5: Logprob(-0.5, 2, None), 7: Logprob(-0.1, 1, None)})  # sampled token first
    logprobs.append(None)
    logprobs.append({3: Logprob(float("-inf"), 9, None)})

    np.testing.assert_array_equal(pack_sampled_logprobs(logprobs), np.array([-0.5, -9999.0, -9999.0], np.float32))


@pytest.mark.gpu
def test_capture_supports_is_the_masked_set():
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    from prime_rl.inference.vllm.paired_capture import capture_supports

    torch.manual_seed(0)
    num_reqs, vocab = 6, 50_000
    logits = torch.randn(num_reqs, vocab, device="cuda") * torch.tensor([1, 2, 4, 8, 0.1, 3], device="cuda")[:, None]
    logits[3] = logits[3].round()
    top_p = torch.tensor([0.95, 0.9, 0.97, 0.5, 0.99, 0.95], device="cuda")
    masked = apply_top_k_top_p_triton(logits.clone(), None, top_p)
    cu = torch.arange(num_reqs + 1, device="cuda", dtype=torch.int32)
    committed = torch.tensor([1, 1, 0, 1, 1, 1], device="cuda", dtype=torch.int32)
    rows = (
        torch.tensor([2], device="cuda"),
        torch.tensor([[7, 8, 9]], device="cuda"),
        torch.tensor([3], device="cuda"),
    )
    captured = capture_supports(masked, cu, committed, 1 << 20, rows).to_cpu_nonblocking()
    torch.cuda.synchronize()
    supports = captured.tolists()

    offsets = supports.offsets
    for req in range(num_reqs):
        entries = supports.token_ids[offsets[req] : offsets[req + 1]]
        if req == 2:
            assert entries.tolist() == [7, 8, 9]
            continue
        kept = torch.isfinite(masked[req]).nonzero().squeeze(-1)
        ids = entries & 0xFFFFFFFF
        values = ((entries >> 32).astype(np.int32) ^ NEG_INF_BITS).view(np.float32)
        assert ids.tolist() == kept.tolist()
        np.testing.assert_allclose(values, masked[req, kept].log_softmax(-1).cpu().numpy(), atol=1e-5)

    overflow = capture_supports(masked, cu, committed, int(offsets[2]), rows).to_cpu_nonblocking()
    torch.cuda.synchronize()
    assert overflow.counts_cpu.tolist()[:2] == [offsets[1], offsets[2] - offsets[1]]
    assert (overflow.counts_cpu[2:] == -1).all()
