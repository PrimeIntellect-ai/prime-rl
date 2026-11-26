"""What the file monitor writes for traces.

`monitors.file.trace_sample` writes the traces of a fraction of train groups: whole groups,
the same ones in the episode stream, the live traces and the orchestrator's annotations.
`monitors.file.live = false` writes no live traces: no per-turn appends, no placeholders."""

import asyncio

import orjson
import pytest

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.env.heads import EpisodeHead, recorded_head
from prime_rl.monitors.file.monitor import FileMonitor, in_trace_sample
from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_path
from prime_rl.monitors.file.traces.live import get_live_dir, get_pending_dir
from prime_rl.orchestrator.annotations import stamp_batch
from tests.unit.orchestrator.test_episode_records import _over_the_wire
from tests.unit.orchestrator.test_train_sink import trace, whole_episode

GROUPS = [f"group-{i}" for i in range(400)]


def episode(group: str) -> EpisodeHead:
    """A finished episode of ``group`` as the orchestrator receives it, holding its record."""
    return recorded_head(whole_episode([trace()], group=group))


def _log(monitor: FileMonitor, episodes: list[EpisodeHead], kind: str = "train") -> None:
    records = [(head, head.take_record()) for head in episodes]
    asyncio.run(monitor.log_records(records, step=1, kind=kind))


def _monitor(tmp_path, fraction: float = 1.0, live: bool = True) -> FileMonitor:
    monitor = FileMonitor(FileMonitorConfig(trace_sample=fraction, live=live))
    asyncio.run(monitor.init(tmp_path))
    return monitor


def _live_events(group: str) -> tuple[str, list[dict]]:
    """A dispatched train episode of ``group`` as the dispatcher reports it: its placeholder,
    then the deltas that stream its trace. Returns the trace id and the events."""
    dispatch = {"id": f"d-{group}", "kind": "train", "group": group}
    _, deltas = _over_the_wire(whole_episode([trace()], group=group))
    events = [{"pending": dispatch["id"], "dispatch": dispatch}]
    events.extend({"delta": delta, "dispatch": dispatch} for delta in deltas)
    return deltas[0]["trace"], events


def _stream_records(tmp_path) -> list[dict]:
    path = chunk_path(get_trace_stream(tmp_path), 0)
    return [orjson.loads(line) for line in path.read_bytes().splitlines()] if path.exists() else []


def _group_pair(fraction: float) -> tuple[str, str]:
    """A group in the sample and one out of it."""
    inside = next(g for g in GROUPS if in_trace_sample(g, fraction))
    outside = next(g for g in GROUPS if not in_trace_sample(g, fraction))
    return inside, outside


def test_default_writes_every_trace():
    assert FileMonitorConfig().trace_sample == 1.0
    assert all(in_trace_sample(g, 1.0) for g in GROUPS)


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_fraction_must_be_in_range(fraction):
    with pytest.raises(ValueError):
        FileMonitorConfig(trace_sample=fraction)


def test_writes_whole_sampled_groups(tmp_path):
    monitor = _monitor(tmp_path, 0.25)
    _log(monitor, [episode(g) for g in GROUPS for _ in range(2)])

    written = [record["group"]["id"] for record in _stream_records(tmp_path)]
    expected = {g for g in GROUPS if in_trace_sample(g, 0.25)}
    assert set(written) == expected
    assert all(written.count(g) == 2 for g in expected)  # a written group is complete
    assert 0.15 < len(expected) / len(GROUPS) < 0.35


def test_eval_episodes_are_always_written(tmp_path):
    monitor = _monitor(tmp_path, 0.01)
    _log(monitor, [episode(g) for g in GROUPS[:20]], kind="eval")
    assert len(_stream_records(tmp_path)) == 20


def test_annotations_follow_the_written_traces(tmp_path):
    monitor = _monitor(tmp_path, 0.5)
    inside, outside = _group_pair(0.5)
    episodes = [episode(inside), episode(outside)]
    _log(monitor, episodes)
    asyncio.run(monitor.log_annotations(stamp_batch(episodes, step=1)))

    stream = chunk_path(get_annotations_dir(tmp_path) / "unknown", 0)
    annotated = [orjson.loads(line)["trace_id"] for line in stream.read_bytes().splitlines()]
    assert annotated == [episodes[0].traces[0].id]


def test_a_producer_without_episodes_keeps_its_annotations(tmp_path):
    """The trainer annotates traces it never logged: its updates are not sampled."""
    monitor = _monitor(tmp_path, 0.5)
    _, outside = _group_pair(0.5)
    asyncio.run(monitor.log_annotations(stamp_batch([episode(outside)], step=1)))
    assert chunk_path(get_annotations_dir(tmp_path) / "unknown", 0).exists()


def test_live_traces_follow_the_sample(tmp_path):
    monitor = _monitor(tmp_path, 0.5)
    inside, outside = _group_pair(0.5)
    events, trace_ids = [], {}
    for group in (inside, outside):
        trace_ids[group], group_events = _live_events(group)
        events.extend(group_events)
    asyncio.run(monitor.log_live(events))

    live_dir, pending_dir = get_live_dir(tmp_path), get_pending_dir(tmp_path)
    assert {p.stem for p in live_dir.glob("*.jsonl")} == {trace_ids[inside]}
    assert {p.stem for p in pending_dir.glob("*.json")} == {f"d-{inside}"}

    # cleanup events for both groups: the sampled group's files go, the other's never existed
    cleanup = [{"dispatched": f"d-{g}"} for g in (inside, outside)] + [
        {"done": trace_ids[g]} for g in (inside, outside)
    ]
    asyncio.run(monitor.log_live(cleanup))
    assert not list(live_dir.glob("*.jsonl"))
    assert not list(pending_dir.glob("*.json"))


def test_live_off_writes_no_live_traces(tmp_path):
    monitor = _monitor(tmp_path, live=False)
    trace_id, events = _live_events("g")
    asyncio.run(monitor.log_live(events))
    asyncio.run(monitor.log_live([{"dispatched": "d-g"}, {"done": trace_id}]))
    assert not list(get_live_dir(tmp_path).rglob("*"))


def test_live_off_clears_a_previous_attempts_live_traces(tmp_path):
    stale = get_live_dir(tmp_path) / "stale.jsonl"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}\n")
    monitor = _monitor(tmp_path, live=False)
    _, events = _live_events("g")
    asyncio.run(monitor.log_live(events))
    assert not stale.exists()


def test_live_off_still_writes_finished_episodes(tmp_path):
    monitor = _monitor(tmp_path, live=False)
    _log(monitor, [episode("g")])
    assert len(_stream_records(tmp_path)) == 1
