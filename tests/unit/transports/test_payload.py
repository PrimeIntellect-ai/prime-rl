"""Training arrays kept by env servers: the trainer must assemble each branch's arrays,
cut, packed and padded like its tokens, and the store must serve and free them."""

import base64
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.episode import EnvInfo, Episode
from verifiers.v1.types import SamplingMask, TopLogprobs, TurnTokens

from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.transports import payload
from prime_rl.transports.batch import types as wire
from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY, MicroBatch, PayloadSlice, TrainingSample, encode_array
from prime_rl.transports.payload import (
    PayloadHook,
    PayloadServer,
    delete_payloads,
    encode_branch_arrays,
    payload_host,
    resolve_payloads,
    wait_for_deletes,
)


def _episode(
    rng: np.random.Generator, prompt_len: int, completion_len: int, *, routed: bool = True, masked: bool = False
) -> vf.Episode:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    rows = rng.integers(0, 128, size=(prompt_len + completion_len - 1, 3, 2)).astype(np.uint8)
    counts = rng.integers(1, 5, size=completion_len).astype(np.int32)
    head = TopLogprobs(
        ids=rng.integers(0, 1000, size=int(counts.sum())).astype(np.int32),
        logprobs=-rng.random(int(counts.sum())).astype(np.float32),
        counts=counts,
    )
    graph.prepare_turn(trace, [vf.UserMessage(content="u")]).commit(
        vf.Response(
            id="a",
            created=0,
            model="t",
            message=vf.AssistantMessage(content="a"),
            finish_reason="stop",
            tokens=TurnTokens.model_construct(
                prompt_ids=list(range(prompt_len)),
                completion_ids=list(range(completion_len)),
                completion_logprobs=[-0.5] * completion_len,
                message_spans=[(0, prompt_len)],
                routed_experts={
                    "data": base64.b64encode(rows.tobytes()).decode(),
                    "shape": list(rows.shape),
                    "start": 0,
                }
                if routed
                else None,
                top_logprobs=head,
                # A mask the head covers exactly carries the head's logprobs.
                sampling_mask=SamplingMask(ids=head.ids, counts=head.counts) if masked else None,
            ),
        )
    )
    trace.stop("agent_completed")
    trace.ok = True
    return Episode(env=EnvInfo(id="e"), task=trace.task, ok=True, traces=[trace])


def _samples(episodes: list[vf.Episode]):
    samples = []
    for episode in episodes:
        for sample in trace_to_samples(episode.traces[0], env_name="env"):
            sample.temperatures = np.ones(len(sample.token_ids), dtype=np.float32)
            sample.advantages = np.ones(len(sample.token_ids), dtype=np.float32)
            samples.append(sample)
    return samples


def _rows(counts: bytes, *columns: bytes | None) -> list[tuple]:
    """Per token, its row of each column (ids, logprobs) as Python values."""
    counts = np.frombuffer(counts, np.int32)
    ends = np.cumsum(counts)
    out = []
    for start, end in zip(ends - counts, ends):
        out.append(
            tuple(
                None if c is None else tuple(np.frombuffer(c, np.int32 if i == 0 else np.float32)[start:end].tolist())
                for i, c in enumerate(columns)
            )
        )
    return out


def test_fetched_micro_batches_hold_each_branchs_arrays():
    rng = np.random.default_rng(0)
    episodes = [
        _episode(rng, 40, 300),
        _episode(rng, 5, 30, masked=True),
        _episode(rng, 20, 90),
        _episode(rng, 8, 12, masked=True),
        _episode(rng, 6, 40, routed=False),
        _episode(rng, 9, 25, routed=False, masked=True),
    ]
    expected = {
        episode.traces[0].id: encode_branch_arrays(branch, len(branch.token_ids))
        for episode in episodes
        for branch in episode.traces[0].branches
    }
    hook = PayloadHook()
    for episode in episodes:
        hook.keep(episode, vf.SamplingConfig())
        assert all(node.routed_experts is None for node in episode.traces[0].nodes)
    samples = _samples(episodes)
    assert all(sample.payload is not None for sample in samples)

    for pad in (1, 64):
        kwargs = dict(seq_len=256, num_train_workers=2, bin_cost=sum, pad_to_multiple_of=pad)
        with ThreadPoolExecutor(4) as pool:
            for rank in prepare_batch(samples, **kwargs):
                layout = [(mb.trace_ids, list(mb.sequence_lengths), mb.padding) for mb in rank]
                failed, urls = resolve_payloads(rank, pool)
                assert failed == [] and urls
                for mb, (trace_ids, lengths, padding) in zip(rank, layout, strict=True):
                    lengths[-1] -= padding
                    want = [(expected[trace_id], n) for trace_id, n in zip(trace_ids, lengths, strict=True)]
                    # Routing replays per micro batch: all of its samples have rows or none.
                    assert len({arrays.routed_experts is None for arrays, _ in want}) == 1
                    if want[0][0].routed_experts is None:
                        assert mb.routed_experts is None
                    else:
                        rows = b"".join(
                            arrays.routed_experts.data[
                                : n * len(arrays.routed_experts.data) // arrays.routed_experts.shape[0]
                            ]
                            for arrays, n in want
                        )
                        rows += payload._tile(rows, padding * len(rows) // sum(lengths))
                        assert mb.routed_experts.data == rows
                    for name, empty in (("sampling_mask", (0, 0)), ("top_logprobs", (0, 0))):
                        got = getattr(mb, name)
                        heads = [getattr(arrays, name) for arrays, _ in want]
                        if all(head is None for head in heads):
                            assert got is None
                            continue
                        got_rows = _rows(got.counts, got.ids, got.logprobs)
                        want_rows = []
                        for head, (_, n) in zip(heads, want):
                            want_rows += (
                                _rows(head.counts, head.ids, head.logprobs)[:n]
                                if head is not None
                                else [((), None)] * n
                            )
                        want_rows += [((), None)] * padding
                        assert [ids for ids, _ in got_rows] == [ids for ids, _ in want_rows]
                        for (_, got_lp), (ids, want_lp) in zip(got_rows, want_rows):
                            if want_lp is not None:
                                assert got_lp == want_lp
                            elif got_lp is not None:
                                assert all(np.isnan(got_lp)) and len(got_lp) == len(ids)
                    assert mb.payloads is None


def test_heads_and_masks_survive_truncation_mixed_packing_and_padding():
    """Explicit values: a top-k head, a mask with sampler logprobs and a plain mask reach
    the trainer cut to their slices, packed together and padded."""
    server = PayloadServer("127.0.0.1")

    def sample(key: str | None, fields: list[str], **arrays) -> TrainingSample:
        if key is not None:
            server.put(key, payload._encoder.encode(wire.BranchArrays(**arrays)))
        return TrainingSample(
            token_ids=np.array([1, 2, 3, 4], np.int32),
            mask=np.array([False, False, True, True]),
            logprobs=np.array([0.0, 0.0, -0.1, -0.2], np.float32),
            temperatures=np.ones(4, np.float32),
            advantages=np.array([0.0, 0.0, 1.0, 1.0], np.float32),
            env_name="env",
            payload=PayloadSlice(url=server.url, id=key, num_tokens=4, fields=fields) if key is not None else None,
        )

    head = wire.TopLogprobs(
        ids=np.array([3, 7, 4], np.int32).tobytes(),
        logprobs=np.array([-0.3, -1.8, -0.2], np.float32).tobytes(),
        counts=np.array([0, 0, 2, 1], np.int32).tobytes(),
    )
    replayed = wire.SamplingMask(
        ids=np.array([5, 6, 8], np.int32).tobytes(),
        counts=np.array([0, 0, 1, 2], np.int32).tobytes(),
        logprobs=np.log(np.array([1.0, 0.75, 0.25], np.float32)).tobytes(),
    )
    plain = wire.SamplingMask(ids=np.array([9], np.int32).tobytes(), counts=np.array([0, 0, 0, 1], np.int32).tobytes())
    samples = [
        sample("head", ["top_logprobs"], top_logprobs=head),
        sample("replayed", ["sampling_mask", "sampling_mask_logprobs"], sampling_mask=replayed),
        sample("plain", ["sampling_mask"], sampling_mask=plain),
        # No arrays at all: it shares the micro batch and gets empty rows.
        sample(None, []),
    ]
    with ThreadPoolExecutor(2) as pool:
        [[truncated]] = prepare_batch(samples[:1], seq_len=3, num_train_workers=1, bin_cost=sum)
        assert resolve_payloads([truncated], pool)[0] == []
        np.testing.assert_array_equal(np.frombuffer(truncated.top_logprobs.counts, np.int32), [0, 0, 2])
        np.testing.assert_array_equal(np.frombuffer(truncated.top_logprobs.ids, np.int32), [3, 7])
        [[truncated]] = prepare_batch(samples[1:2], seq_len=3, num_train_workers=1, bin_cost=sum)
        assert resolve_payloads([truncated], pool)[0] == []
        np.testing.assert_array_equal(np.frombuffer(truncated.sampling_mask.ids, np.int32), [5])
        np.testing.assert_array_equal(np.frombuffer(truncated.sampling_mask.logprobs, np.float32), [0.0])

        [[packed]] = prepare_batch(samples, seq_len=16, num_train_workers=1, bin_cost=sum, pad_to_multiple_of=16)
        packed = msgspec.msgpack.decode(msgspec.msgpack.encode(packed, enc_hook=encode_array), type=MicroBatch)
        assert resolve_payloads([packed], pool)[0] == []
    assert len(packed.top_logprobs.ids) == 3 * 4
    tensor = DataLoader._micro_batch_to_tensor(None, packed)
    offsets, ids, values = tensor["support_offsets"], tensor["support_ids"], tensor["support_logprobs"]
    assert ids.shape == values.shape
    assert offsets.shape[0] == sum(packed.sequence_lengths) + 1
    rows = {
        tuple(ids[start:end].tolist()): values[start:end].tolist()
        for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist())
        if end > start
    }
    expected = {(3, 7): [-0.3, -1.8], (4,): [-0.2], (5,): [0.0], (6, 8): np.log([0.75, 0.25]), (9,): [np.nan]}
    assert rows.keys() == expected.keys()
    for key, value in expected.items():
        np.testing.assert_allclose(rows[key], value, rtol=1e-6)
    replayed_rows = {
        tuple(ids[start:end].tolist())
        for start, end, row_replayed in zip(offsets[:-1].tolist(), offsets[1:].tolist(), tensor["support_replayed"])
        if row_replayed
    }
    assert replayed_rows == {(5,), (6, 8), (9,)}


@pytest.mark.parametrize("support", ["complete", "mismatch"])
def test_sampler_evidence_moves_masked_top_logprobs_onto_the_mask(support):
    import verifiers.v1 as vf

    from prime_rl.transports.payload import _encode_sampler_evidence

    mask = vf.SamplingMask(ids=np.array([5, 6, 8], np.int32), counts=np.array([0, 0, 1, 2], np.int32))
    head = vf.TopLogprobs(
        ids=np.array([5, 8, 6], np.int32),
        logprobs=np.array([0.0, -0.3, -1.4], np.float32),
        counts=np.array([0, 0, 1, 2] if support == "complete" else [0, 1, 0, 2], np.int32),
    )
    if support == "mismatch":
        with pytest.raises(ValueError):
            _encode_sampler_evidence(mask, head, 4)
        return
    sampling_mask, top_logprobs = _encode_sampler_evidence(mask, head, 4)
    assert top_logprobs is None
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.ids, np.int32), [5, 8, 6])
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.counts, np.int32), [0, 0, 1, 2])
    np.testing.assert_array_equal(np.frombuffer(sampling_mask.logprobs, np.float32), np.float32([0.0, -0.3, -1.4]))


def test_unfetchable_micro_batches_are_reported():
    rng = np.random.default_rng(1)
    episode = _episode(rng, 10, 20)
    PayloadHook().keep(episode, vf.SamplingConfig())
    [sample] = _samples([episode])
    sample.payload.id = "missing"
    # Truncated, so the reference the trainer sees is the packer's copy.
    [[micro_batch]] = prepare_batch([sample], seq_len=16, num_train_workers=1, bin_cost=sum)
    with ThreadPoolExecutor(1) as pool:
        assert resolve_payloads([micro_batch], pool)[0] == [0]
    # Still replayable: the model's routing shape, every top-k slot a different expert.
    routed = micro_batch.routed_experts
    assert routed.shape == [16, 3, 2]
    rows = np.frombuffer(routed.data, dtype=routed.dtype).reshape(routed.shape)
    assert (rows[..., 0] != rows[..., 1]).all()


def _two_trace_episode(rng: np.random.Generator) -> Episode:
    first, second = _episode(rng, 6, 8), _episode(rng, 6, 8)
    return Episode(env=first.env, task=first.task, ok=True, traces=[first.traces[0], second.traces[0]])


def _failing_second_encode(monkeypatch, error: Exception) -> None:
    encode = payload.encode_branch_arrays
    calls = []

    def flaky(branch, num_tokens):
        calls.append(branch)
        if len(calls) == 2:
            raise error
        return encode(branch, num_tokens)

    monkeypatch.setattr(payload, "encode_branch_arrays", flaky)


def test_keep_frees_what_it_kept_when_it_fails(monkeypatch):
    hook = PayloadHook()
    episode = _two_trace_episode(np.random.default_rng(3))
    _failing_second_encode(monkeypatch, MemoryError())
    with pytest.raises(MemoryError):
        hook.keep(episode, vf.SamplingConfig())
    assert not hook.server.blobs


def test_keep_keeps_routing_when_the_sampler_evidence_does_not_encode(monkeypatch):
    """The episode still succeeds (eval never needs the evidence) and keeps its routing
    rows; a train run that needs the mask or head stops at the train sink."""
    hook = PayloadHook()
    episode = _two_trace_episode(np.random.default_rng(4))
    _failing_second_encode(monkeypatch, ValueError("Top logprobs are not aligned to branch tokens"))
    hook.keep(episode, vf.SamplingConfig())
    first, second = episode.traces
    assert first.info[PAYLOAD_INFO_KEY]["branches"]["0"]["fields"] == ["routed_experts", "top_logprobs"]
    assert second.info[PAYLOAD_INFO_KEY]["branches"]["0"]["fields"] == ["routed_experts"]
    assert all(node.routed_experts is None for node in second.nodes)


def test_a_hook_that_keeps_nothing_starts_no_server():
    """A pool broker installs the hook too; it must not open a payload server."""
    hook = PayloadHook()
    hook.discard(_episode(np.random.default_rng(5), 4, 4))
    assert hook._server is None


def test_fetches_bypass_http_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    server = PayloadServer("127.0.0.1")
    server.put("a", payload._encoder.encode(wire.BranchArrays()))
    assert payload._fetch(f"{server.url}/a") == wire.BranchArrays()


def test_short_routing_rows_fail_the_alignment_check():
    rows = wire.RoutedExperts(data=bytes(3 * 2 * 2), shape=[4, 2, 2], dtype="uint8")
    mb = SimpleNamespace(routed_experts=rows, sampling_mask=None, top_logprobs=None)
    with pytest.raises(AssertionError, match="bytes"):
        payload._assert_aligned(mb, num_tokens=4)


def test_payload_server_binds_the_env_server_host():
    assert payload_host(None) == ("127.0.0.1", "127.0.0.1")
    assert payload_host("tcp://10.1.2.3:0") == ("10.1.2.3", "10.1.2.3")
    bind, url = payload_host("tcp://*:5000")
    assert bind == "0.0.0.0" and not url.startswith("127.")


def test_store_serves_and_deletes():
    server = PayloadServer("127.0.0.1")
    server.put("a", b"x" * 40)
    server.put("b", b"y" * 40)
    with urllib.request.urlopen(f"{server.url}/b") as response:
        assert response.read() == b"y" * 40
    delete_payloads([f"{server.url}/b", f"{server.url}/missing"])
    wait_for_deletes()
    assert server.get("b") is None and server.get("a") is not None


def test_discard_frees_what_keep_kept():
    hook = PayloadHook()
    episode = _episode(np.random.default_rng(2), 6, 8)
    hook.keep(episode, vf.SamplingConfig())
    assert hook.server.blobs
    hook.discard(episode)
    assert not hook.server.blobs
