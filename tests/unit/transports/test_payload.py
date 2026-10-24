"""Training arrays kept by env servers: the trainer must assemble exactly the arrays the
inline path packs, and the store must serve and free them."""

import base64
import copy
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.episode import EnvInfo, Episode
from verifiers.v1.types import TopLogprobs, TurnTokens

from prime_rl.orchestrator.batch import prepare_batch
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.transports.payload import PayloadHook, PayloadServer, delete_payloads, resolve_payloads


def _episode(rng: np.random.Generator, prompt_len: int, completion_len: int) -> vf.Episode:
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="x")),
    )
    routed = rng.integers(0, 128, size=(prompt_len + completion_len - 1, 3, 2)).astype(np.uint8)
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
                    "data": base64.b64encode(routed.tobytes()).decode(),
                    "shape": list(routed.shape),
                    "start": 0,
                },
                top_logprobs=head,
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
            sample.temperatures = [1.0] * len(sample.token_ids)
            sample.advantages = [1.0] * len(sample.token_ids)
            samples.append(sample)
    return samples


def test_fetched_batches_equal_inline_batches():
    rng = np.random.default_rng(0)
    episodes = [_episode(rng, prompt, completion) for prompt, completion in [(40, 300), (5, 30), (20, 90), (8, 12)]]
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
                    assert got.input_ids == want.input_ids
                    assert got.routed_experts == want.routed_experts
                    assert got.top_logprobs == want.top_logprobs
                    assert got.payloads is None


def test_unfetchable_micro_batches_are_reported():
    rng = np.random.default_rng(1)
    episode = _episode(rng, 10, 20)
    PayloadHook().keep(episode, vf.SamplingConfig())
    [sample] = _samples([episode])
    sample.payload.id = "missing"
    [[micro_batch]] = prepare_batch([sample], seq_len=64, num_train_workers=1, bin_cost=sum)
    with ThreadPoolExecutor(1) as pool:
        assert resolve_payloads([micro_batch], pool)[0] == [0]
    assert micro_batch.routed_experts.shape[0] == len(micro_batch.input_ids)


def test_store_serves_and_deletes():
    server = PayloadServer()
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
