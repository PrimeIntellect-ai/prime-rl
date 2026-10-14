"""Durable calls' traces kept and found by their refs: the file monitor keeps a trace
that never landed (``traces/partial/``) instead of deleting it, points each durable ref
at the trace carrying it that moved last wherever it is, and ``RunTraceLookup`` answers
``vf.TraceLookup.get`` from that pointer across a run's attempts. The deltas are the real
wire format (``verifiers.v1.serve.delta.DeltaStreamer`` over real traces)."""

import asyncio

import orjson
import verifiers.v1 as vf
from verifiers.v1.durable import call_ref
from verifiers.v1.serve.delta import DeltaStreamer, pack, unpack

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.eval import resume
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.file.traces import get_refs_dir
from prime_rl.monitors.file.traces.live import get_live_dir, get_partial_dir, list_live, read_live
from prime_rl.monitors.file.traces.lookup import RunTraceLookup
from prime_rl.monitors.file.traces.refs import read_ref

JOB = {"flow": {"job": "j1"}}


def ref(key: str) -> str:
    """The ref of durable call ``key`` in job j1."""
    return call_ref(JOB, key)


def _trace(key: str, *, start: float, checkpoints: tuple[float, ...] = ()) -> vf.Trace:
    labels = {**JOB, "call": {"key": key, "fingerprint": "f", "ref": ref(key)}}
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        info={"labels": labels},
    )
    trace.timing.boot.start = start
    trace.checkpoints = [vf.Checkpoint(id=f"ckpt-{t}", turns=i, time=t) for i, t in enumerate(checkpoints)]
    return trace


class Stream:
    """The env server's side: diffs the live traces into deltas the way it serves them,
    each through msgpack as the wire carries it, wrapped as the dispatcher queues them."""

    def __init__(self, dispatch_id: str = "d1") -> None:
        self.traces: list[vf.Trace] = []
        self.dispatch = {"id": dispatch_id, "kind": "eval", "env": "e", "started": 0.0}
        self.events: list[dict] = []
        self.streamer = DeltaStreamer(lambda: self.traces, self._send)

    async def _send(self, delta: dict) -> None:
        self.events.append({"delta": unpack(pack(delta)), "dispatch": self.dispatch})

    def flush(self) -> list[dict]:
        asyncio.run(self.streamer.flush())
        events, self.events = self.events, []
        return events


def _monitor(tmp_path) -> FileMonitor:
    monitor = FileMonitor(FileMonitorConfig(compress=False))
    asyncio.run(monitor.init(tmp_path))
    return monitor


def _land(monitor: FileMonitor, *traces: vf.Trace) -> vf.Episode:
    episode = vf.Episode(task=traces[0].task, traces=list(traces), ok=True)
    asyncio.run(monitor.log_episodes([episode], step=0, kind="eval", subset="all"))
    return episode


def _pointer(tmp_path, key: str) -> dict | None:
    return read_ref(get_refs_dir(tmp_path), ref(key))


def test_a_live_trace_is_found_by_its_ref_as_it_streams(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    stream = Stream()
    trace = _trace("agent/0", start=1.0)
    stream.traces.append(trace)
    asyncio.run(monitor.log_live(stream.flush()))
    lookup = RunTraceLookup(tmp_path)
    found = lookup.get(ref("agent/0"))
    assert found is not None and found.id == trace.id and not found.checkpoints
    assert lookup.get(ref("agent/1")) is None
    assert _pointer(tmp_path, "agent/0") == {"trace": trace.id, "state": "live", "recency": 1.0, "dispatch": "d1"}

    # a checkpoint moves it: the pointer follows, and the lookup sees the save
    trace.checkpoints.append(vf.Checkpoint(id="ckpt-1", turns=1, time=5.0))
    asyncio.run(monitor.log_live(stream.flush()))
    assert _pointer(tmp_path, "agent/0")["recency"] == 5.0
    assert [c.id for c in lookup.get(ref("agent/0")).checkpoints] == ["ckpt-1"]


def test_a_crashed_episodes_partial_trace_survives_and_is_found(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    stream = Stream()
    trace = _trace("agent/0", start=1.0, checkpoints=(2.0, 3.0))
    stream.traces.append(trace)
    asyncio.run(monitor.log_live(stream.flush()))
    # the process dies here; a relaunch streams again under a fresh monitor
    relaunched = _monitor(tmp_path)
    asyncio.run(relaunched.log_live([]))

    assert list_live(tmp_path) == []
    partial = get_partial_dir(tmp_path) / f"{trace.id}.jsonl"
    assert orjson.loads(partial.read_bytes().splitlines()[-1])["end"] == "crashed"
    assert read_live(partial)[1]["id"] == trace.id  # the ending line folds away
    assert _pointer(tmp_path, "agent/0") == {
        "trace": trace.id,
        "state": "partial",
        "recency": 3.0,
        "dispatch": "d1",
        "end": "crashed",
    }
    found = RunTraceLookup(tmp_path).get(ref("agent/0"))
    assert found.id == trace.id
    assert [c.id for c in found.checkpoints] == ["ckpt-2.0", "ckpt-3.0"]
    assert not found.is_completed


def test_a_cancelled_or_discarded_trace_is_kept_as_partial(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    stream = Stream()
    cancelled = _trace("agent/0", start=1.0)
    discarded = _trace("agent/1", start=2.0)
    stream.traces[:] = [cancelled, discarded]
    first = stream.flush()
    # an env retry abandons one; the episode is then cancelled under the other, and
    # both endings arrive in the same batch as the deltas still buffered
    stream.traces[:] = [cancelled]
    events = first + stream.flush() + [{"done": cancelled.id, "end": "cancelled"}]
    events.append({"delta": {"trace": cancelled.id, "set": {"ok": False}}, "dispatch": stream.dispatch})  # a straggler
    asyncio.run(monitor.log_live(events))

    assert sorted(path.name for path in get_partial_dir(tmp_path).iterdir()) == sorted(
        f"{trace.id}.jsonl" for trace in (cancelled, discarded)
    )
    assert not any(get_live_dir(tmp_path).glob("*.jsonl"))
    assert _pointer(tmp_path, "agent/0")["end"] == "cancelled"
    assert _pointer(tmp_path, "agent/1")["end"] == "discarded"
    lookup = RunTraceLookup(tmp_path)
    assert lookup.get(ref("agent/0")).id == cancelled.id
    assert lookup.get(ref("agent/1")).id == discarded.id


def test_a_landed_trace_is_found_in_the_stream_and_stays_final(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    stream = Stream()
    solver = _trace("solver/0", start=1.0, checkpoints=(4.0,))
    judge = _trace("judge/0", start=2.0)
    stream.traces[:] = [solver, judge]
    asyncio.run(monitor.log_live(stream.flush()))
    solver.is_completed = solver.ok = True
    _land(monitor, solver, judge)
    # the landing's live bookkeeping trails the episode: a late delta, then the ending
    late = stream.flush() + [{"done": t.id, "end": "landed"} for t in (solver, judge)]
    asyncio.run(monitor.log_live(late))

    assert not any(get_live_dir(tmp_path).glob("*.jsonl"))
    assert not get_partial_dir(tmp_path).exists()
    assert [_pointer(tmp_path, key)["state"] for key in ("solver/0", "judge/0")] == ["finished"] * 2
    lookup = RunTraceLookup(tmp_path)
    found = lookup.get(ref("solver/0"))
    assert found.id == solver.id and found.is_completed and found.ok
    assert lookup.get(ref("judge/0")).id == judge.id
    # the trace streaming again (a durable call reusing it) leaves its finished pointer
    reuse = Stream("d2")
    reuse.traces.append(solver)
    asyncio.run(monitor.log_live(reuse.flush()))
    assert _pointer(tmp_path, "solver/0")["state"] == "finished"
    assert RunTraceLookup(tmp_path).get(ref("solver/0")).is_completed


def test_a_ref_moves_only_to_a_trace_that_moved_later(tmp_path) -> None:
    """A call's later run (its continuation) takes the ref; an earlier run's late write
    (its ending, after the continuation started) does not take it back."""
    monitor = _monitor(tmp_path)
    stream = Stream()
    cut = _trace("agent/0", start=1.0, checkpoints=(2.0,))
    stream.traces.append(cut)
    asyncio.run(monitor.log_live(stream.flush()))
    continued = _trace("agent/0", start=5.0)
    stream.traces.append(continued)
    asyncio.run(monitor.log_live(stream.flush()))
    asyncio.run(monitor.log_live([{"done": cut.id, "end": "failed"}]))
    assert RunTraceLookup(tmp_path).get(ref("agent/0")).id == continued.id


def test_resume_keeps_every_attempts_traces_findable(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    landed = _trace("agent/0", start=1.0)
    cut = _trace("agent/1", start=2.0, checkpoints=(3.0,))
    _land(monitor, landed)
    stream = Stream()
    stream.traces.append(cut)
    asyncio.run(monitor.log_live(stream.flush()))
    held = RunTraceLookup(tmp_path)  # a long-lived lookup, open across the resume
    assert held.get(ref("agent/1")).id == cut.id

    # the process dies with agent/1 live; --resume archives the attempt and starts afresh
    resume.take_landed(tmp_path)
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]
    relaunched = _monitor(tmp_path)
    asyncio.run(relaunched.log_live([]))
    later = _trace("agent/2", start=9.0)
    _land(relaunched, later)
    # the cut call continues in the new attempt: its new trace takes the ref there
    again = _trace("agent/1", start=10.0)
    stream.traces[:] = [again]
    asyncio.run(relaunched.log_live(stream.flush()))

    for lookup in (held, RunTraceLookup(tmp_path)):
        assert lookup.get(ref("agent/0")).id == landed.id
        assert lookup.get(ref("agent/2")).id == later.id
        assert lookup.get(ref("agent/1")).id == again.id
    # before the continuation started, the archived attempt's cut trace was the call's
    resumed = resume.archives(tmp_path)[0]
    pointer = read_ref(resumed / "traces" / "refs", ref("agent/1"))
    assert pointer["trace"] == cut.id
