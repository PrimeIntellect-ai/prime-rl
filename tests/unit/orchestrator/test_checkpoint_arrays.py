"""An orchestrator checkpoint keeps the training arrays its open work needs: env-server
workers write the blobs of unshipped episodes and the arrays of in-flight rollouts up to
their restore points next to it, and a resumed run reads them back from there."""

import base64
import uuid
from types import SimpleNamespace

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1 import graph
from verifiers.v1.serve import RestoreRequest
from verifiers.v1.types import SamplingMask, TopLogprobs, TurnTokens

from prime_rl.orchestrator.ckpt import PERSISTED_URL, persist_arrays, resolve_arrays
from prime_rl.orchestrator.types import GroupState
from prime_rl.transports import payload
from prime_rl.transports.batch.types import PAYLOAD_ARRAYS_KEY, PAYLOAD_INFO_KEY, PAYLOAD_WORKER_KEY, PayloadSlice
from prime_rl.transports.payload import PayloadHook, persist_payloads


def _trace(rng: np.random.Generator, prompt_len: int = 5, completion_len: int = 6) -> vf.Trace:
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
                },
                top_logprobs=head,
                sampling_mask=SamplingMask(ids=head.ids, counts=head.counts),
            ),
        )
    )
    return trace


def _arrays(trace: vf.Trace) -> list[tuple]:
    def rows(value):
        if value is None or isinstance(value, np.ndarray):
            return None if value is None else value.tolist()
        return [getattr(value, name).tolist() for name in vars(value)]

    return [tuple(rows(getattr(node, name)) for name in payload.NODE_ARRAY_FIELDS) for node in trace.nodes]


def _without_arrays(trace: vf.Trace) -> vf.Trace:
    copy = trace.model_copy(deep=True)
    for node in copy.nodes:
        node.routed_experts = node.sampling_mask = node.top_logprobs = None
    return copy


def test_a_worker_persists_an_inflight_trace_up_to_its_restore_point(tmp_path):
    hook = PayloadHook(checkpoint_dirs=[tmp_path])
    trace = _trace(np.random.default_rng(0))
    hook.watch(trace)
    assert trace.info[PAYLOAD_WORKER_KEY] == hook.server.url
    trace.checkpoints.append(vf.Checkpoint(id="ck", node=len(trace.nodes) - 1))
    assert trace.nodes[-1].routed_experts is not None

    # A checkpoint the trace no longer holds (a retry rewound past it) is not written.
    assert persist_payloads(hook.server.url, tmp_path, [], {trace.id: ("gone", 1)}) == ([], [])
    _, traces = persist_payloads(hook.server.url, tmp_path, [], {trace.id: ("ck", len(trace.nodes))})
    assert traces == [trace.id]

    restored = _without_arrays(trace)
    restored.info[PAYLOAD_ARRAYS_KEY] = str(tmp_path / trace.id)
    PayloadHook(checkpoint_dirs=[tmp_path]).restore(restored)
    assert PAYLOAD_ARRAYS_KEY not in restored.info
    assert _arrays(restored) == _arrays(trace)


def _episode(hook: PayloadHook, rng: np.random.Generator) -> vf.Episode:
    trace = _trace(rng)
    run = vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=0, end=0)))
    episode = vf.Episode(env=vf.EnvInfo(id="e"), task=trace.task, ok=True, traces=[trace], run=run)
    hook.keep(episode, vf.SamplingConfig())
    return episode


def _restore(trace: vf.Trace) -> RestoreRequest:
    checkpoint = {"id": "ck", "node": len(trace.nodes) - 1}
    return RestoreRequest(trace={"id": trace.id, "info": dict(trace.info)}, point={"checkpoint": checkpoint})


def _work(hook: PayloadHook) -> tuple[dict, vf.Trace]:
    rng = np.random.default_rng(1)
    pending, queued, result = (_episode(hook, rng) for _ in range(3))
    (reference,) = queued.traces[0].info[PAYLOAD_INFO_KEY]["branches"].values()
    sample = SimpleNamespace(payload=PayloadSlice(url=hook.server.url, id=reference["id"], num_tokens=4, fields=[]))
    inflight, gone = _trace(rng), _trace(rng)
    for trace in (inflight, gone):
        hook.watch(trace)
        trace.checkpoints.append(vf.Checkpoint(id="ck", node=len(trace.nodes) - 1))
    group = GroupState(
        kind="train",
        env_name="a",
        task=vf.Task(vf.TaskData(idx=0, prompt="x")),
        step=3,
        episodes_to_schedule=2,
        target_episodes=4,
        group_id=uuid.uuid4(),
        restores=[_restore(inflight), _restore(gone)],
    )
    # Finished on the worker since the snapshot: nothing to persist for it anymore.
    hook.traces.pop(gone.id)
    work = {
        "groups": [group],
        "sink": {
            "pending_groups": {"g": [pending]},
            "episode_by_trace": {queued.traces[0].id: queued},
            "pending_batch": {queued.traces[0].id: [sample]},
        },
        "results": [result],
    }
    return work, inflight


def test_the_checkpoint_holds_the_open_works_arrays_and_a_resume_reads_them(tmp_path):
    hook = PayloadHook(checkpoint_dirs=[tmp_path])
    work, inflight = _work(hook)
    blobs = dict(hook.server.blobs)

    persist_arrays(work, tmp_path / "step_5" / "arrays")
    episodes = [*work["sink"]["pending_groups"]["g"], *work["sink"]["episode_by_trace"].values(), *work["results"]]
    assert {episode.traces[0].info[PAYLOAD_INFO_KEY]["url"] for episode in episodes} == {PERSISTED_URL}
    ((sample,),) = work["sink"]["pending_batch"].values()
    assert sample.payload.url == PERSISTED_URL
    # The rollout whose arrays were not written starts over; its slot stays scheduled.
    (group,) = work["groups"]
    assert [restore.trace["id"] for restore in group.restores] == [inflight.id]
    assert group.episodes_to_schedule == 2

    # The env servers are gone on resume: everything reads from the checkpoint.
    hook.server.blobs.clear()
    directory = (tmp_path / "step_5" / "arrays").resolve()
    resolve_arrays(work, directory)
    assert sample.payload.url == directory.as_uri()
    for episode in episodes:
        info = episode.traces[0].info[PAYLOAD_INFO_KEY]
        for reference in info["branches"].values():
            fetched = payload._fetch(f"{info['url']}/{reference['id']}")
            assert fetched == payload._decoder.decode(blobs[reference["id"]])
    restored = _without_arrays(inflight)
    restored.info |= group.restores[0].trace["info"]
    PayloadHook(checkpoint_dirs=[tmp_path]).restore(restored)
    assert _arrays(restored) == _arrays(inflight)

    # The resumed run checkpoints again before it dispatched anything: what it still
    # holds moves to the new checkpoint, the restore with it.
    persist_arrays(work, tmp_path / "step_6" / "arrays")
    assert sample.payload.url == PERSISTED_URL
    assert (tmp_path / "step_6" / "arrays" / sample.payload.id).exists()
    assert [restore.trace["info"][PAYLOAD_ARRAYS_KEY] for restore in group.restores] == [inflight.id]
    assert (tmp_path / "step_6" / "arrays" / inflight.id).exists()


def test_a_worker_persists_and_restores_only_inside_the_runs_checkpoints(tmp_path):
    checkpoints = tmp_path / "run" / "checkpoints"
    hook = PayloadHook(checkpoint_dirs=[checkpoints])
    trace = _trace(np.random.default_rng(2))
    hook.watch(trace)
    trace.checkpoints.append(vf.Checkpoint(id="ck", node=len(trace.nodes) - 1))
    hook.server.put("blob", b"x")
    # Outside the run's checkpoints, or under a name that is not an id: nothing is written.
    assert persist_payloads(hook.server.url, tmp_path / "elsewhere", ["blob"], {}) == ([], [])
    assert not (tmp_path / "elsewhere").exists()
    hook.server.put("../escape", b"x")
    blobs, _ = persist_payloads(hook.server.url, checkpoints / "step_1", ["../escape", "blob"], {})
    assert blobs == ["blob"] and not (checkpoints / "escape").exists()

    (tmp_path / trace.id).write_bytes(payload._encode_node_arrays(trace.nodes))
    restored = _without_arrays(trace)
    restored.info[PAYLOAD_ARRAYS_KEY] = str(tmp_path / trace.id)
    with pytest.raises(ValueError, match="outside"):
        hook.restore(restored)
