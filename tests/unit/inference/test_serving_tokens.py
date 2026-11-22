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
    _parse_replayed_support,
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
from prime_rl.inference.vllm.paired_capture import NEG_INF_BITS, pack_sampler_rows
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


@pytest.mark.asyncio
async def test_replay_mask_rows_dropped_by_capacity_fail_the_request():
    # A replay request without compact logprobs: an overflowed step leaves an empty mask row.
    mask = PackedSamplingMask(ids=np.array([1, 2], dtype=np.int64), counts=np.array([2, 0], dtype=np.int32))
    output = CompletionOutput(0, "", [1, 2], None, None, sampling_mask=mask, finish_reason="length")

    async def outputs():
        yield RequestOutput("test", None, [10], None, [output], True)

    params = SamplingParams(extra_args={"prl_replay": True})
    packed = _PackedOutputs(outputs(), GenerateRequest(model="test", token_ids=[10], sampling_params=params))
    with pytest.raises(ValueError, match="sampling_support_capacity"):
        async for _ in packed:
            pass


def _rows_by_id(ids_per_row: list[list[int]], logprobs_per_row: list[list[float]]):
    """Sort each row by token id, since rows are sets and only native heads come ranked by probability."""
    rows = [sorted(zip(ids, logprobs)) for ids, logprobs in zip(ids_per_row, logprobs_per_row)]
    return [[i for i, _ in row] for row in rows], [[p for _, p in row] for row in rows]


def _gpu_capture(logq: dict[int, float], sampled_ids: list[int], replay: bool):
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
        _, packed, _ = pack_sampler_rows(logits, cu, num_sampled, sampled, rows)
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
    serving.return_tokens_as_token_ids = False
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
                if replay:
                    sampled, replayed = _parse_replayed_support(choice, sampled_ids)
                    cuts = np.cumsum(replayed["counts"])[:-1]
                    evidence.append(
                        (
                            sampled,
                            [row.tolist() for row in np.split(replayed["ids"], cuts)],
                            [row.tolist() for row in np.split(replayed["logprobs"], cuts)],
                        )
                    )
                else:
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
    evidence = [(np.round(logprobs, 6).tolist(), *_rows_by_id(ids, values)) for logprobs, ids, values in evidence]
    assert all(item[0] == evidence[0][0] and item[1] == evidence[0][1] for item in evidence)
    np.testing.assert_allclose(np.array(evidence[-1][2]), np.array(evidence[0][2]), atol=1e-6)
    assert evidence[0][1] == ([[1, 2, 3]] * 2 if replay else [[1, 2]] * 2)
    assert responses[0] == responses[1]
    assert responses[1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 1
    assert responses[1]["kv_transfer_params"] == {"kv": "kept"}
    assert responses[1]["ec_transfer_params"] == {"ec": "kept"}


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


@pytest.mark.gpu
def test_verified_supports_follow_accepted_positions_across_chunks():
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

    from prime_rl.inference.vllm.paired_capture import VerifiedSupports, capture_verified_supports

    torch.manual_seed(0)
    # Request 0 verifies 3 positions, request 1 only its bonus row, request 2 (not replayed) 3 positions.
    logits = torch.randn(7, 50_000, device="cuda")
    masked = apply_top_k_top_p_triton(logits, None, torch.full((7,), 0.9, device="cuda"))
    cu = torch.tensor([0, 3, 4, 7], device="cuda", dtype=torch.int32)
    num_sampled = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int32)
    replay = np.array([True, True, False])
    whole = capture_verified_supports(masked, cu, num_sampled, replay, 3, 1 << 20)
    chunked = VerifiedSupports.cat(
        [
            capture_verified_supports(masked[:3], cu[:2], num_sampled[:1], replay[:1], 3, 1 << 20),
            capture_verified_supports(masked[3:], cu[1:] - 3, num_sampled[1:], replay[1:], 3, 1 << 20),
        ]
    )
    for supports in (whole, chunked):
        supports.to_cpu_nonblocking()
        torch.cuda.synchronize()
        lists = supports.tolists(np.array([3, 1, 2]))
        assert lists.cu_num_generated_tokens == [0, 3, 4, 6]
        for position, row in enumerate([0, 1, 2, 3]):
            entries = lists.token_ids[lists.offsets[position] : lists.offsets[position + 1]]
            kept = torch.isfinite(masked[row]).nonzero().squeeze(-1)
            values = ((entries >> 32).astype(np.int32) ^ NEG_INF_BITS).view(np.float32)
            assert (entries & 0xFFFFFFFF).tolist() == kept.tolist()
            np.testing.assert_allclose(values, masked[row, kept].log_softmax(-1).cpu().numpy(), atol=1e-5)
        assert np.diff(lists.offsets)[4:].tolist() == [0, 0]


@pytest.mark.asyncio
async def test_flat_sampled_logprobs_are_packed():
    """The default RL path (flat logprobs, logprobs<=1): upstream reads ``logprobs`` before it
    consumes the outputs, so the override must turn them off up front."""
    serving = object.__new__(PrimeRlServingTokens)
    serving.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(aux_output_config=SimpleNamespace(enable_return_routed_experts=False))
    )
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    serving.return_tokens_as_token_ids = False
    logprobs = FlatLogprobs()
    for sampled, value in [(2, -0.25), (7, -20000.0)]:
        logprobs.append_fast([sampled], [value], iter(range(1)), [None])
    output = RequestOutput(
        "test",
        None,
        [10, 11],
        None,
        [CompletionOutput(0, "", [2, 7], None, logprobs, finish_reason="length")],
        True,
    )

    async def generate_outputs():
        yield output

    params = SamplingParams(logprobs=1, flat_logprobs=True)
    request = GenerateRequest(model="test", token_ids=[10, 11], sampling_params=params, return_token_ids=True)
    metadata = RequestResponseMetadata(request_id="test")
    response = await serving.serve_tokens_full_generator(request, generate_outputs(), "test", "test", metadata)
    assert isinstance(response, GenerateTokensResponse)
    assert params.logprobs == 1
    [choice] = response.model_dump()["choices"]
    assert choice["logprobs"] is None
    packed = choice["completion_logprobs"]
    values = np.frombuffer(pybase64.b64decode(packed["data"]), dtype=packed["dtype"])
    np.testing.assert_array_equal(values, np.array([-0.25, -9999.0], dtype=np.float32))


@pytest.mark.parametrize(
    ("logprobs", "logprob_token_ids", "packed"),
    [(0, None, True), (1, None, True), (1, [4, 5], False), (-1, None, False), (2, None, False)],
)
def test_only_sampled_token_logprobs_are_packed_flat(logprobs, logprob_token_ids, packed):
    params = SamplingParams(logprobs=logprobs, logprob_token_ids=logprob_token_ids, flat_logprobs=True)
    request = GenerateRequest(model="test", token_ids=[10, 11], sampling_params=params)
    assert _PackedOutputs(_empty_request_outputs(), request).strips_logprobs is packed


def _make_serving(*, routed: bool = False) -> PrimeRlServingTokens:
    serving = object.__new__(PrimeRlServingTokens)
    serving.engine_client = SimpleNamespace(
        vllm_config=SimpleNamespace(aux_output_config=SimpleNamespace(enable_return_routed_experts=routed))
    )
    serving.enable_prompt_tokens_details = False
    serving.enable_log_outputs = False
    serving.return_tokens_as_token_ids = False
    return serving


def _cpu_paired_mask(rows_ids: list[list[int]], rows_logprobs: list[list[float]]) -> PackedSamplingMask:
    flat_ids = np.asarray([i for row in rows_ids for i in row], dtype=np.int64)
    flat_vals = np.asarray([p for row in rows_logprobs for p in row], dtype=np.float32)
    packed = ((flat_vals.view(np.int32).astype(np.int64) ^ NEG_INF_BITS) << 32) | flat_ids
    counts = np.asarray([len(row) for row in rows_ids], dtype=np.int32)
    return PackedSamplingMask(ids=packed, counts=counts)


@pytest.mark.asyncio
async def test_unified_sampled_only_format_and_legacy_renderer_compat():
    from prime_rl.inference.vllm.logprobs import (
        SAMPLED_LOGPROBS_FORMAT,
        parse_completion_logprobs,
        parse_generate_response_packed,
    )

    serving = _make_serving()
    logprobs = FlatLogprobs()
    for sampled, value in [(2, -0.25), (7, -1.5)]:
        logprobs.append_fast([sampled], [value], iter(range(1)), [None])
    output = RequestOutput(
        "test",
        None,
        [10, 11],
        None,
        [CompletionOutput(0, "", [2, 7], None, logprobs, finish_reason="length")],
        True,
    )

    async def generate_outputs():
        yield output

    params = SamplingParams(logprobs=1, flat_logprobs=True)
    request = GenerateRequest(model="test", token_ids=[10, 11], sampling_params=params, return_token_ids=True)
    response = await serving.serve_tokens_full_generator(
        request, generate_outputs(), "test", "test", RequestResponseMetadata(request_id="test")
    )
    assert isinstance(response, GenerateTokensResponse)
    raw = JSONResponse(content=response.model_dump()).body
    payload = parse_generate_response_packed(raw)
    [choice] = payload["choices"]
    assert choice["logprobs"] is None
    assert choice.get("compact_logprobs") is None
    packed = choice["completion_logprobs"]
    assert list(packed)[0] == "data"
    assert isinstance(packed["data"], memoryview)
    assert packed["format"] == SAMPLED_LOGPROBS_FORMAT
    assert packed["shape"] == [2]
    assert packed["dtype"] == "float32"
    assert len(pybase64.b64decode(packed["data"], validate=True)) == 2 * 4

    # Both the unified parser and existing unmigrated renderers parse the same payload.
    unified_sampled, support = parse_completion_logprobs(choice, [2, 7])
    legacy_sampled = _parse_completion_logprobs(choice, [2, 7])
    assert support is None
    np.testing.assert_array_equal(unified_sampled, np.array([-0.25, -1.5], dtype=np.float32))
    np.testing.assert_array_equal(legacy_sampled, unified_sampled)

    # Unified parser also reads legacy sampled-only payloads that omit the format field.
    legacy_only_choice = {"completion_logprobs": {k: v for k, v in packed.items() if k != "format"}}
    legacy_only_sampled, legacy_only_support = parse_completion_logprobs(legacy_only_choice, [2, 7])
    assert legacy_only_support is None
    np.testing.assert_array_equal(legacy_only_sampled, unified_sampled)


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("mode", ["unified", "dual_compat", "legacy"])
async def test_packed_logprobs_unifies_sampler_rows_and_replayed_support(replay, mode):
    from prime_rl.inference.vllm.logprobs import (
        PACKED_LOGPROBS_FORMAT,
        parse_completion_logprobs,
        parse_generate_response_packed,
    )

    serving = _make_serving()
    # For replay=False, token 7 is outside the top-4 head [1, 2, 3, 4].
    sampled_ids = [2, 3] if replay else [2, 7]
    probs = {1: 0.6, 2: 0.2, 3: 0.2, 4: 0.0} if replay else {1: 0.6, 2: 0.25, 3: 0.1, 4: 0.04, 7: 0.01}
    with np.errstate(divide="ignore"):
        logq = {i: float(np.float32(np.log(q))) for i, q in probs.items()}

    if replay:
        rows_ids = [[1, 2, 3], [1, 2, 3]]
        rows_vals = [[logq[i] for i in row] for row in rows_ids]
        native_support = PackedSamplingMask(
            ids=np.array([1, 2, 3, 1, 2, 3], dtype=np.int32), counts=np.array([3, 3], dtype=np.int32)
        )
    else:
        rows_ids = [[sampled, 1, 2, 3, 4] for sampled in sampled_ids]
        rows_vals = [[logq[i] for i in row] for row in rows_ids]
        native_support = None

    # Reference native vLLM response.
    native_logprobs = FlatLogprobs()
    for sampled in sampled_ids:
        ids = [sampled, 1, 2, 3, 4]
        native_logprobs.append_fast(ids, [logq[i] for i in ids], iter(range(len(ids))), [None] * len(ids))
    native_out = RequestOutput(
        "test",
        None,
        [10, 11],
        None,
        [
            CompletionOutput(
                0, "", sampled_ids, None, native_logprobs, sampling_mask=native_support, finish_reason="length"
            )
        ],
        True,
    )

    async def native_gen():
        yield deepcopy(native_out)

    native_req = GenerateRequest(
        model="test", token_ids=[10, 11], sampling_params=SamplingParams(logprobs=4), return_token_ids=True
    )
    native_resp = await serving.serve_tokens_full_generator(
        native_req, native_gen(), "test", "test", RequestResponseMetadata(request_id="test")
    )
    native_choice = parse_generate_response(JSONResponse(content=native_resp.model_dump()).body)["choices"][0]
    ref_sampled = _parse_completion_logprobs(native_choice, sampled_ids)
    ref_top_ids, ref_top_lps = _parse_completion_top_logprobs(native_choice, sampled_ids, 2)

    # Captured response in unified, dual_compat, or legacy mode.
    extra_args: dict[str, object] = {"prl_paired_capture": True} if replay else {"prl_sampler_rows": 4}
    if mode == "unified":
        extra_args["prl_packed_logprobs"] = True
    elif mode == "dual_compat":
        extra_args["prl_packed_logprobs"] = True
        extra_args["prl_compact_logprobs"] = True
    else:
        extra_args["prl_compact_logprobs"] = True

    captured_out = RequestOutput(
        "test",
        None,
        [10, 11],
        None,
        [
            CompletionOutput(
                0,
                "",
                sampled_ids,
                None,
                None,
                sampling_mask=_cpu_paired_mask(rows_ids, rows_vals),
                finish_reason="length",
            )
        ],
        True,
    )

    async def captured_gen():
        yield deepcopy(captured_out)

    captured_req = GenerateRequest(
        model="test",
        token_ids=[10, 11],
        sampling_params=SamplingParams(logprobs=None, extra_args=extra_args),
        return_token_ids=True,
    )
    captured_resp = await serving.serve_tokens_full_generator(
        captured_req, captured_gen(), "test", "test", RequestResponseMetadata(request_id="test")
    )
    raw = JSONResponse(content=captured_resp.model_dump()).body

    if mode in ("unified", "dual_compat"):
        choice = parse_generate_response_packed(raw)["choices"][0]
        assert choice["logprobs"] is None
        assert choice.get("compact_logprobs") is None
        packed = choice["completion_logprobs"]
        assert list(packed)[0] == "data"
        assert isinstance(packed["data"], memoryview)
        assert packed["format"] == PACKED_LOGPROBS_FORMAT
        raw_bytes = pybase64.b64decode(packed["data"], validate=True)
        if not replay:
            # Leading sampled slot stripped into leading f32[T]: 4 bytes/token smaller than legacy u32-f32-le-v1.
            assert packed["offsets"] == [0, 4, 8]
            assert len(raw_bytes) == 2 * 4 + 8 * 8
        else:
            assert packed["offsets"] == [0, 3, 6]
            assert len(raw_bytes) == 2 * 4 + 6 * 8

        got_sampled, support = parse_completion_logprobs(
            choice, sampled_ids, head_size=None if replay else 2, replayed=replay
        )
        assert support is not None
        assert support["replayed"] is replay
        cuts = np.cumsum(support["counts"])[:-1]
        got_ids = [row.tolist() for row in np.split(support["ids"], cuts)]
        got_lps = [row.tolist() for row in np.split(support["logprobs"], cuts)]
    else:
        choice = parse_generate_response(raw)["choices"][0]
        assert choice["logprobs"] is None
        assert choice.get("completion_logprobs") is None
        assert isinstance(choice["compact_logprobs"]["data"], memoryview)
        if replay:
            got_sampled, support = _parse_replayed_support(choice, sampled_ids)
            cuts = np.cumsum(support["counts"])[:-1]
            got_ids = [row.tolist() for row in np.split(support["ids"], cuts)]
            got_lps = [row.tolist() for row in np.split(support["logprobs"], cuts)]
        else:
            got_sampled, got_ids, got_lps = _parse_compact_logprobs(choice, sampled_ids, 2)
        # Unified parser also handles legacy compact_logprobs payloads during migration.
        compat_sampled, compat_support = parse_completion_logprobs(
            choice, sampled_ids, head_size=None if replay else 2, replayed=replay
        )
        np.testing.assert_allclose(compat_sampled, got_sampled, atol=1e-6)
        assert compat_support is not None and compat_support["replayed"] is replay

    np.testing.assert_allclose(got_sampled, ref_sampled, atol=1e-6)
    got_sorted_ids, got_sorted_lps = _rows_by_id(got_ids, got_lps)
    ref_sorted_ids, ref_sorted_lps = _rows_by_id(ref_top_ids, ref_top_lps)
    assert got_sorted_ids == ref_sorted_ids
    np.testing.assert_allclose(np.array(got_sorted_lps), np.array(ref_sorted_lps), atol=1e-6)


@pytest.mark.parametrize("use_flat", [True, False])
def test_unified_serialize_sample_logprobs_ragged_duplicates_and_ties(use_flat):
    from prime_rl.inference.vllm.logprobs import (
        PACKED_LOGPROBS_FORMAT,
        SAMPLED_LOGPROBS_FORMAT,
        parse_completion_logprobs,
        serialize_sample_logprobs,
    )

    # Row 0: sampled (2) duplicated inside top-4; row 1: sampled (7) outside top-4 with filtered -inf candidate;
    # row 2: ragged row of 5 unique candidates with a tie at boundary k=2 (tokens 2 and 3 tied at -1.2).
    token_ids = [2, 7, 1]

    def _lp(prob: float) -> float:
        return float(np.float32(np.log(prob)))

    row_defs = [
        [(2, _lp(0.25)), (1, _lp(0.55)), (2, _lp(0.25)), (3, _lp(0.12)), (4, _lp(0.05))],
        [(7, _lp(0.01)), (1, _lp(0.60)), (2, _lp(0.25)), (3, _lp(0.10)), (4, float("-inf"))],
        [(1, _lp(0.50)), (2, _lp(0.20)), (3, _lp(0.20)), (4, _lp(0.06)), (5, _lp(0.03))],
    ]
    if use_flat:
        lp = FlatLogprobs()
        # Insert a skipped gap entry between row 0 and row 1 to exercise disjoint FlatLogprobs slices.
        lp.append_fast([i for i, _ in row_defs[0]], [p for _, p in row_defs[0]], iter(range(5)), [None] * 5)
        lp.token_ids.append(999)
        lp.logprobs.append(-0.01)
        lp.ranks.append(0)
        lp.decoded_tokens.append(None)
        for row in row_defs[1:]:
            lp.append_fast([i for i, _ in row], [p for _, p in row], iter(range(len(row))), [None] * len(row))
    else:
        lp = [{i: Logprob(p) for i, p in row} for row in row_defs]

    packed = serialize_sample_logprobs(lp, token_ids, num_top_logprobs=4)
    assert packed["format"] == PACKED_LOGPROBS_FORMAT
    sampled, support = parse_completion_logprobs(
        {"completion_logprobs": packed}, token_ids, head_size=2, replayed=False
    )
    expected_sampled = [_lp(0.25), _lp(0.01), _lp(0.50)]
    np.testing.assert_allclose(sampled, expected_sampled, atol=1e-6)
    assert support is not None
    cuts = np.cumsum(support["counts"])[:-1]
    ids_rows = [row.tolist() for row in np.split(support["ids"], cuts)]
    lps_rows = [row.tolist() for row in np.split(support["logprobs"], cuts)]
    assert ids_rows == [[1, 2], [1, 2], [1]]  # row 2 excludes boundary tie at k=2
    np.testing.assert_allclose(lps_rows[0], [_lp(0.55), _lp(0.25)], atol=1e-6)
    np.testing.assert_allclose(lps_rows[1], [_lp(0.60), _lp(0.25)], atol=1e-6)
    np.testing.assert_allclose(lps_rows[2], [_lp(0.50)], atol=1e-6)

    # Sampled-only mode and empty completion round-trip.
    sampled_only = serialize_sample_logprobs(lp, token_ids, num_top_logprobs=0)
    assert sampled_only["format"] == SAMPLED_LOGPROBS_FORMAT
    s_only, sup_only = parse_completion_logprobs({"completion_logprobs": sampled_only}, token_ids)
    assert sup_only is None
    np.testing.assert_allclose(s_only, expected_sampled, atol=1e-6)

    empty_lp = FlatLogprobs() if use_flat else []
    empty_sampled = serialize_sample_logprobs(empty_lp, [], num_top_logprobs=0)
    empty_vals, empty_sup = parse_completion_logprobs({"completion_logprobs": empty_sampled}, [])
    assert len(empty_vals) == 0 and empty_sup is None


@pytest.mark.asyncio
async def test_serve_tokens_negotiation_for_packed_logprobs(monkeypatch):
    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens

    captured_requests: list[GenerateRequest] = []

    async def fake_super_serve_tokens(self, request, raw_request=None):
        captured_requests.append(request)
        return "ok"

    monkeypatch.setattr(ServingTokens, "serve_tokens", fake_super_serve_tokens)
    serving = _make_serving()
    serving.create_error_response = lambda msg: SimpleNamespace(error=msg)

    # Rejects streaming, non-tokens output_mode, or logprobs=None.
    for bad_req in [
        GenerateRequest(
            model="m",
            token_ids=[1],
            stream=True,
            sampling_params=SamplingParams(logprobs=4, extra_args={"prl_packed_logprobs": True}),
        ),
        GenerateRequest(
            model="m",
            token_ids=[1],
            output_mode="text",
            sampling_params=SamplingParams(logprobs=4, extra_args={"prl_packed_logprobs": True}),
        ),
        GenerateRequest(
            model="m",
            token_ids=[1],
            sampling_params=SamplingParams(logprobs=None, extra_args={"prl_packed_logprobs": True}),
        ),
    ]:
        err = await serving.serve_tokens(bad_req)
        assert hasattr(err, "error")

    # Top-k packed -> prl_sampler_rows=4, logprobs=None.
    req_topk = GenerateRequest(
        model="m", token_ids=[1], sampling_params=SamplingParams(logprobs=4, extra_args={"prl_packed_logprobs": True})
    )
    assert await serving.serve_tokens(req_topk) == "ok"
    assert req_topk.sampling_params.extra_args["prl_sampler_rows"] == 4
    assert req_topk.sampling_params.logprobs is None

    # Replay + packed -> prl_paired_capture=True, logprobs=None.
    req_replay = GenerateRequest(
        model="m",
        token_ids=[1],
        sampling_params=SamplingParams(logprobs=1, extra_args={"prl_packed_logprobs": True, "prl_replay": True}),
    )
    assert await serving.serve_tokens(req_replay) == "ok"
    assert req_replay.sampling_params.extra_args["prl_paired_capture"] is True
    assert req_replay.sampling_params.logprobs is None

    # Sampled-only (logprobs=0) + packed -> keeps flat_logprobs, no GPU sampler rows.
    req_sampled = GenerateRequest(
        model="m", token_ids=[1], sampling_params=SamplingParams(logprobs=0, extra_args={"prl_packed_logprobs": True})
    )
    assert await serving.serve_tokens(req_sampled) == "ok"
    assert req_sampled.sampling_params.logprobs == 0
    assert req_sampled.sampling_params.flat_logprobs is True
    assert "prl_sampler_rows" not in req_sampled.sampling_params.extra_args


def test_unified_logprobs_reject_malformed_and_missing_evidence():
    from prime_rl.inference.vllm.logprobs import (
        PackedLogprobsError,
        encode_packed_logprobs,
        encode_sampled_logprobs,
        parse_completion_logprobs,
    )

    valid_sampled = encode_sampled_logprobs(np.array([-0.2], dtype=np.float32))
    valid_packed = encode_packed_logprobs(
        np.array([-0.2], dtype=np.float32),
        np.array([7, 1, 2, 3], dtype="<u4"),
        np.array([-0.2, -1.0, -2.0, -3.0], dtype="<f4"),
        [0, 4],
        4,
    )

    # Sampled-only rejections.
    for bad in [
        {**valid_sampled, "format": "unknown-v99"},
        {**valid_sampled, "dtype": "float16"},
        {**valid_sampled, "shape": [2]},
        {**valid_sampled, "data": "not!base64"},
        {**valid_sampled, "data": pybase64.b64encode(b"123").decode("ascii")},
        encode_sampled_logprobs(np.array([-9999.0], dtype=np.float32)),
        encode_sampled_logprobs(np.array([float("nan")], dtype=np.float32)),
        encode_sampled_logprobs(np.array([0.5], dtype=np.float32)),
    ]:
        with pytest.raises(PackedLogprobsError):
            parse_completion_logprobs({"completion_logprobs": bad}, [7])

    # Requesting top-k head from sampled-only payload must fail.
    with pytest.raises(PackedLogprobsError):
        parse_completion_logprobs({"completion_logprobs": valid_sampled}, [7], head_size=2)

    # Candidate-row rejections.
    for bad in [
        {**valid_packed, "offsets": [1, 4]},
        {**valid_packed, "offsets": [0]},
        {**valid_packed, "offsets": [0, 2, 1]},
        {**valid_packed, "offsets": [0, True]},
        {**valid_packed, "num_top_logprobs": -2},
        {**valid_packed, "num_top_logprobs": True},
        {**valid_packed, "data": "bad!b64"},
        {**valid_packed, "data": pybase64.b64encode(b"\x00" * 4).decode("ascii")},
        # Sampled logprob disagrees with candidate row entry for token 7.
        encode_packed_logprobs(
            np.array([-0.9], dtype=np.float32),
            np.array([7, 1, 2, 3], dtype="<u4"),
            np.array([-0.2, -1.0, -2.0, -3.0], dtype="<f4"),
            [0, 4],
            4,
        ),
        # Repeated candidate IDs with disagreeing logprobs.
        encode_packed_logprobs(
            np.array([-0.2], dtype=np.float32),
            np.array([7, 1, 1, 2], dtype="<u4"),
            np.array([-0.2, -1.0, -1.5, -2.0], dtype="<f4"),
            [0, 4],
            4,
        ),
        # Positive candidate logprob.
        encode_packed_logprobs(
            np.array([-0.2], dtype=np.float32),
            np.array([7, 1, 2, 3], dtype="<u4"),
            np.array([-0.2, 0.25, -2.0, -3.0], dtype="<f4"),
            [0, 4],
            4,
        ),
        # Head probability mass > 1.0001.
        encode_packed_logprobs(
            np.array([-0.1], dtype=np.float32),
            np.array([7, 1, 2, 3], dtype="<u4"),
            np.array([-0.1, -0.1, -2.0, -3.0], dtype="<f4"),
            [0, 4],
            4,
        ),
    ]:
        with pytest.raises(PackedLogprobsError):
            parse_completion_logprobs({"completion_logprobs": bad}, [7], head_size=2)

    # Fewer than k + 2 candidates in top-k mode.
    too_few = encode_packed_logprobs(
        np.array([-0.2], dtype=np.float32),
        np.array([7, 1, 2], dtype="<u4"),
        np.array([-0.2, -1.0, -2.0], dtype="<f4"),
        [0, 3],
        3,
    )
    with pytest.raises(PackedLogprobsError):
        parse_completion_logprobs({"completion_logprobs": too_few}, [7], head_size=2)

    # Replayed support rejections: unnormalized mass, missing sampled token, mismatched sampling_mask.
    unnormalized = encode_packed_logprobs(
        np.array([-0.5], dtype=np.float32),
        np.array([1, 7], dtype="<u4"),
        np.array([-0.5, -0.5], dtype="<f4"),
        [0, 2],
        2,
    )
    with pytest.raises(PackedLogprobsError):
        parse_completion_logprobs({"completion_logprobs": unnormalized}, [7], replayed=True)

    normalized = encode_packed_logprobs(
        np.array([float(np.log(0.4))], dtype=np.float32),
        np.array([1, 7], dtype="<u4"),
        np.array([float(np.log(0.6)), float(np.log(0.4))], dtype="<f4"),
        [0, 2],
        2,
    )
    with pytest.raises(PackedLogprobsError):
        parse_completion_logprobs({"completion_logprobs": normalized}, [9], replayed=True)
    wrong_mask = {
        "ids": encode_array(np.array([1, 8], dtype=np.int32)),
        "counts": encode_array(np.array([2], dtype=np.int32)),
    }
    with pytest.raises(PackedLogprobsError):
        parse_completion_logprobs({"completion_logprobs": normalized, "sampling_mask": wrong_mask}, [7], replayed=True)
