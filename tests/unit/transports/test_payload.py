"""Training arrays kept by env servers: the trainer must assemble exactly the arrays the
inline path packs, and the store must serve and free them."""

import base64
import copy
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.episode import EnvInfo, Episode
from verifiers.v1.types import SamplingMask, TopLogprobs, TurnTokens

from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.transports import payload
from prime_rl.transports.payload import PayloadHook, PayloadServer, delete_payloads, payload_host, resolve_payloads


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


def test_fetched_batches_equal_inline_batches():
    rng = np.random.default_rng(0)
    episodes = [
        _episode(rng, 40, 300),
        _episode(rng, 5, 30, masked=True),
        _episode(rng, 20, 90),
        _episode(rng, 8, 12, masked=True),
        # Samples without routing must not share a micro batch with routed ones.
        _episode(rng, 6, 40, routed=False),
        _episode(rng, 9, 25, routed=False, masked=True),
    ]
    inline = _samples(copy.deepcopy(episodes))

    hook = PayloadHook()
    for episode in episodes:
        hook.keep(episode, vf.SamplingConfig())
        assert all(node.routed_experts is None for node in episode.traces[0].nodes)
    by_ref = _samples(episodes)
    assert all(s.payload is not None and s.routed_experts is None and s.top_logprobs is None for s in by_ref)

    for pad in (1, 64):
        kwargs = dict(seq_len=256, num_train_workers=2, bin_cost=sum, pad_to_multiple_of=pad)
        expected = prepare_batch(copy.deepcopy(inline), **kwargs)
        packed = prepare_batch(copy.deepcopy(by_ref), **kwargs)
        with ThreadPoolExecutor(4) as pool:
            for want_rank, got_rank in zip(expected, packed, strict=True):
                failed, urls = resolve_payloads(got_rank, pool)
                assert failed == [] and urls
                for want, got in zip(want_rank, got_rank, strict=True):
                    assert np.array_equal(got.input_ids, want.input_ids)
                    assert got.routed_experts == want.routed_experts
                    assert got.sampling_mask == want.sampling_mask
                    assert got.top_logprobs == want.top_logprobs
                    assert got.payloads is None
                    assert want.sampling_mask is None or want.sampling_mask.logprobs is not None


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


def test_keep_frees_what_it_kept_when_it_fails(monkeypatch):
    hook = PayloadHook()
    rng = np.random.default_rng(3)
    first, second = _episode(rng, 6, 8), _episode(rng, 6, 8)
    episode = Episode(env=first.env, task=first.task, ok=True, traces=[first.traces[0], second.traces[0]])
    encode = payload.encode_branch_arrays
    calls = []

    def flaky(branch, num_tokens):
        calls.append(branch)
        if len(calls) == 2:
            raise ValueError("Top logprobs are not aligned to branch tokens")
        return encode(branch, num_tokens)

    monkeypatch.setattr(payload, "encode_branch_arrays", flaky)
    with pytest.raises(ValueError):
        hook.keep(episode, vf.SamplingConfig())
    assert not hook.server.blobs


def test_payload_server_binds_the_env_server_host():
    assert payload_host(None) == "127.0.0.1"
    assert payload_host("tcp://10.1.2.3:0") == "10.1.2.3"


def test_store_serves_and_deletes():
    server = PayloadServer("127.0.0.1")
    server.put("a", b"x" * 40)
    server.put("b", b"y" * 40)
    with urllib.request.urlopen(f"{server.url}/b") as response:
        assert response.read() == b"y" * 40
    with ThreadPoolExecutor(1) as pool:
        delete_payloads([f"{server.url}/b", f"{server.url}/missing"], pool)
    assert server.get("b") is None and server.get("a") is not None


def test_discard_frees_what_keep_kept():
    hook = PayloadHook()
    episode = _episode(np.random.default_rng(2), 6, 8)
    hook.keep(episode, vf.SamplingConfig())
    assert hook.server.blobs
    hook.discard(episode)
    assert not hook.server.blobs
