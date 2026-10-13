"""Traces kept and found by their labels: the file monitor keeps a trace that never
landed (``traces/partial/``) instead of deleting it, writes a labels index row for each
labelled trace wherever it is, and ``RunTraceLookup`` answers ``vf.TraceLookup.latest``
from that index across a run's attempts. The deltas are the real wire format
(``verifiers.v1.serve.delta.DeltaStreamer`` over real traces)."""

import asyncio

import orjson
import verifiers.v1 as vf
from verifiers.v1.serve.delta import DeltaStreamer, pack, unpack

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.eval import resume
from prime_rl.monitors.file import FileMonitor
from prime_rl.monitors.file.traces import get_labels_index_path
from prime_rl.monitors.file.traces.live import get_live_dir, get_partial_dir, list_live, read_live
from prime_rl.monitors.file.traces.lookup import RunTraceLookup

JOB = {"flow": {"job": "j1"}}


def _trace(key: str, *, start: float, checkpoints: tuple[float, ...] = (), job: str = "j1") -> vf.Trace:
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=0, prompt="hi")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        info={"labels": {"flow": {"job": job}, "call": {"key": key, "fingerprint": "f"}}},
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


def _rows(tmp_path) -> list[dict]:
    return [orjson.loads(line) for line in get_labels_index_path(tmp_path).read_bytes().splitlines()]


def test_a_live_trace_is_found_by_its_labels_as_it_streams(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    stream = Stream()
    trace = _trace("agent/0", start=1.0)
    stream.traces.append(trace)
    asyncio.run(monitor.log_live(stream.flush()))
    lookup = RunTraceLookup(tmp_path)
    found = lookup.latest({**JOB, "call": {"key": "agent/0"}})
    assert found is not None and found.id == trace.id and not found.checkpoints
    assert lookup.latest({"flow": {"job": "j2"}}) is None

    # a checkpoint moves it: a row goes out, and the lookup sees the save
    trace.checkpoints.append(vf.Checkpoint(id="ckpt-1", turns=1, time=5.0))
    asyncio.run(monitor.log_live(stream.flush()))
    assert [(row["state"], row["checkpoint"]) for row in _rows(tmp_path)] == [("live", None), ("live", 5.0)]
    assert [c.id for c in lookup.latest(JOB).checkpoints] == ["ckpt-1"]


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
    assert _rows(tmp_path)[-1] | {"labels": None} == {
        "trace": trace.id,
        "labels": None,
        "state": "partial",
        "start": 1.0,
        "checkpoint": 3.0,
        "dispatch": "d1",
        "end": "crashed",
    }
    found = RunTraceLookup(tmp_path).latest({**JOB, "call": {"key": "agent/0"}})
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
    ends = {row["trace"]: row.get("end") for row in _rows(tmp_path) if row["state"] == "partial"}
    assert ends == {cancelled.id: "cancelled", discarded.id: "discarded"}
    lookup = RunTraceLookup(tmp_path)
    assert lookup.latest({"call": {"key": "agent/0"}}).id == cancelled.id
    assert lookup.latest({"call": {"key": "agent/1"}}).id == discarded.id
    # the newest of the job's traces is the one that moved last
    assert lookup.latest(JOB).id == discarded.id


def test_a_landed_trace_is_found_in_the_stream_and_stays_final(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    stream = Stream()
    solver = _trace("solver/0", start=1.0, checkpoints=(4.0,))
    judge = _trace("judge/0", start=2.0)
    judge.info["labels"]["flow"]["seat"] = "judge"  # each trace is indexed by its own labels
    stream.traces[:] = [solver, judge]
    asyncio.run(monitor.log_live(stream.flush()))
    solver.is_completed = solver.ok = True
    _land(monitor, solver, judge)
    # the landing's live bookkeeping trails the episode: a late delta, then the ending
    late = stream.flush() + [{"done": t.id, "end": "landed"} for t in (solver, judge)]
    asyncio.run(monitor.log_live(late))

    assert not any(get_live_dir(tmp_path).glob("*.jsonl"))
    assert not get_partial_dir(tmp_path).exists()
    finished = [row for row in _rows(tmp_path) if row["state"] == "finished"]
    assert [(row["trace"], row["labels"]["flow"]) for row in finished] == [
        (solver.id, {"job": "j1"}),
        (judge.id, {"job": "j1", "seat": "judge"}),
    ]
    lookup = RunTraceLookup(tmp_path)
    found = lookup.latest({"call": {"key": "solver/0"}})
    assert found.id == solver.id and found.is_completed and found.ok
    assert lookup.latest({"flow": {"seat": "judge"}}).id == judge.id
    # the trace streaming again (a durable call reusing it) leaves its finished row in charge
    reuse = Stream("d2")
    reuse.traces.append(solver)
    asyncio.run(monitor.log_live(reuse.flush()))
    assert _rows(tmp_path)[-1]["state"] == "live"
    assert RunTraceLookup(tmp_path).latest({"call": {"key": "solver/0"}}).is_completed


def test_a_lookup_reads_only_the_rows_written_since_the_last(tmp_path, monkeypatch) -> None:
    monitor = _monitor(tmp_path)
    for n in range(20):
        _land(monitor, _trace(f"agent/{n}", start=float(n)))
    lookup = RunTraceLookup(tmp_path)
    assert lookup.latest({"call": {"key": "agent/3"}}).timing.boot.start == 3.0
    assert lookup.poll() == 0  # all 20 rows read already

    _land(monitor, _trace("agent/20", start=20.0))
    assert lookup.poll() == 1  # one new row read, not the 21
    loads = []
    real = RunTraceLookup._load_finished
    monkeypatch.setattr(
        RunTraceLookup, "_load_finished", lambda self, *args: loads.append(args[1]) or real(self, *args)
    )
    assert lookup.latest(JOB).timing.boot.start == 20.0
    assert len(loads) == 1  # and one trace loaded: the one returned
    assert lookup.poll() == 0


def test_resume_keeps_every_attempts_traces_findable(tmp_path) -> None:
    monitor = _monitor(tmp_path)
    landed = _trace("agent/0", start=1.0)
    cut = _trace("agent/1", start=2.0, checkpoints=(3.0,))
    _land(monitor, landed)
    stream = Stream()
    stream.traces.append(cut)
    asyncio.run(monitor.log_live(stream.flush()))
    held = RunTraceLookup(tmp_path)  # a long-lived lookup, open across the resume
    assert held.latest({"call": {"key": "agent/1"}}).id == cut.id

    # the process dies with agent/1 live; --resume archives the attempt and starts afresh
    resume.take_landed(tmp_path)
    assert [path.name for path in resume.archives(tmp_path)] == ["file.attempt_1"]
    relaunched = _monitor(tmp_path)
    asyncio.run(relaunched.log_live([]))
    later = _trace("agent/2", start=9.0, job="j1")
    _land(relaunched, later)

    for lookup in (held, RunTraceLookup(tmp_path)):
        assert lookup.latest({"call": {"key": "agent/0"}}).id == landed.id
        resumed = lookup.latest({"call": {"key": "agent/1"}})
        assert resumed.id == cut.id and [c.id for c in resumed.checkpoints] == ["ckpt-3.0"]
        assert lookup.latest({"call": {"key": "agent/2"}}).id == later.id
        assert lookup.latest(JOB).id == later.id
