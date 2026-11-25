"""Episodes reach the orchestrator as their heads and the records their env worker serialized,
with deltas for the live view. Every record written to disk or uploaded must carry plain JSON
lists, and the orchestrator writes the worker's bytes as they are."""

import asyncio

import orjson
import verifiers.v1 as vf
from prime_runs.sinks.base import stamp_run, to_mapping
from prime_runs.sinks.traces import names_run
from prime_traces.traces import _encode_record
from verifiers.v1.delta import DeltaStreamer

from prime_rl import monitors
from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.env.encoding import pack, unpack
from prime_rl.env.heads import EpisodeHead, recorded_head
from prime_rl.env.worker import stamp_dispatch
from prime_rl.monitors.file.monitor import FileMonitor, in_trace_sample
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_path
from prime_rl.monitors.file.traces.index import index_record, index_row
from prime_rl.monitors.file.traces.live import get_live_dir, read_live
from prime_rl.monitors.file.traces.update import fold_trace_updates
from prime_rl.monitors.prime import RecordUpload
from prime_rl.orchestrator.algo.routing import assign_advantage, assign_advantages
from prime_rl.orchestrator.annotations import log_arrival, stamp_arrival, stamp_batch
from tests.unit.orchestrator.test_train_sink import trace, whole_episode


async def _noop(_):
    return None


def _over_the_wire(source: vf.Episode) -> tuple[EpisodeHead, list[dict]]:
    """The episode as the orchestrator receives it: its head holding its record, and the
    deltas that streamed its traces."""
    traces = source.traces
    deltas = [unpack(pack(delta)) for _, delta, _ in DeltaStreamer(lambda: traces, _noop).diff()]
    return recorded_head(source, float_decimals=4), deltas


def _assert_plain_streams(node: dict) -> None:
    for name in ("token_ids", "mask", "is_content", "logprobs", "advantages", "reference_logprobs"):
        if node.get(name) is not None:
            assert isinstance(node[name], list), name


def test_received_episode_writes_plain_records(tmp_path):
    received, deltas = _over_the_wire(whole_episode([trace()]))
    record = received.take_record()
    assign_advantage(received.traces[0], 0.7)

    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_annotations(stamp_arrival(received, record, "train", step=1)))
    asyncio.run(monitor.log_records([(received, record)], step=1, kind="train"))
    asyncio.run(monitor.log_live([{"delta": delta, "dispatch": {"id": "d"}} for delta in deltas]))
    asyncio.run(monitor.log_annotations(stamp_batch([received], step=1)))

    # The trace stream: the episode record, as every reader of the run directory parses it
    stream = get_trace_stream(tmp_path)
    (line,) = chunk_path(stream, 0).read_bytes().splitlines()
    written = orjson.loads(line)
    for node in written["traces"][0]["nodes"]:
        _assert_plain_streams(node)
    assert written["traces"][0]["nodes"][2]["token_ids"] == [3, 4]
    assert written["traces"][0]["nodes"][2]["logprobs"] == [-0.1, -0.2]
    # Its index row knows when the episode arrived, which the record itself does not
    (row,) = get_index_path(stream).read_bytes().splitlines()
    assert orjson.loads(row)["arrival"] is not None

    # The live trace file folds back to the same tokens
    (live,) = get_live_dir(tmp_path).glob("*.jsonl")
    _, live_trace = read_live(live)
    for node in live_trace["nodes"]:
        _assert_plain_streams(node)
    assert [node["token_ids"] for node in live_trace["nodes"]] == [[1], [2], [3, 4]]

    # The annotation stream: the arrival, then the shipped cohort's advantage
    stream = get_annotations_dir(tmp_path) / "unknown"
    arrival, shipped = (orjson.loads(line) for line in chunk_path(stream, 0).read_bytes().splitlines())
    assert arrival["info"]["arrival"]["step"] == 1 and arrival["info"]["kind"] == "train"
    assert shipped["info"]["advantage"] == 0.7 and shipped["info"]["effective"]
    # A reader folding the updates onto the record sees the advantage on every sampled token
    fold_trace_updates(written["traces"][0], [arrival, shipped])
    assert [node.get("advantages") for node in written["traces"][0]["nodes"]] == [None, None, [0.7, 0.7]]


def test_received_episode_encodes_as_its_record_did(tmp_path):
    """The file monitor writes the worker's record bytes as they are, with the index row of
    the record; the Prime SDK uploads the same bytes for a record that names its run."""
    source = whole_episode([trace(), trace()])
    source.record_run(vf.TrainRunInfo(id="prime-run", work=vf.TrainWorkInfo(step=1)))
    received, _ = _over_the_wire(source)
    record = received.take_record()
    assert record.json == source.to_json()

    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_records([(received, record)], step=1, kind="train"))

    stream = get_trace_stream(tmp_path)
    (line,) = chunk_path(stream, 0).read_bytes().splitlines()
    assert line == record.json
    for node in orjson.loads(line)["traces"][0]["nodes"]:
        _assert_plain_streams(node)
    (row,) = get_index_path(stream).read_bytes().splitlines()
    expected = index_row(1, index_record(source), 0, 0)
    assert orjson.loads(row) == orjson.loads(orjson.dumps(expected, default=str))

    # With the run recorded on the episode and its traces, as the env worker does, the SDK
    # uploads `to_json` as is; it carries what the SDK stamped on the record before.
    upload = RecordUpload(received, record)
    assert names_run(upload, "prime-run", "train")
    assert orjson.loads(upload.to_json()) == orjson.loads(
        _encode_record(stamp_run(to_mapping(source), "prime-run", "train"), 1)
    )
    # A record that does not name the run maps to what the SDK keys to its run, unchanged.
    assert orjson.loads(orjson.dumps(to_mapping(upload))) == orjson.loads(record.json)
    assert orjson.loads(_encode_record(stamp_run(to_mapping(upload), "other", "train"), 1)) == orjson.loads(
        _encode_record(stamp_run(to_mapping(source), "other", "train"), 1)
    )


def test_a_folded_advantage_credits_only_trainable_branches():
    """A rejected compaction attempt's branch never trains, so a reader folding the cohort's
    advantage onto the record leaves its tokens uncredited."""
    nodes = [
        vf.MessageNode(message=vf.UserMessage(content="q"), token_ids=[1], mask=[False]),
        vf.MessageNode(
            parent=0,
            message=vf.AssistantMessage(content="kept"),
            sampled=True,
            token_ids=[2, 3],
            mask=[True, True],
            logprobs=[-0.1, -0.2],
        ),
        vf.MessageNode(
            parent=0,
            semantic_parents=[vf.ParentLink(node=1, type="compaction_attempt")],
            message=vf.AssistantMessage(content="rejected"),
            sampled=True,
            token_ids=[4],
            mask=[True],
            logprobs=[-0.3],
        ),
    ]
    source = whole_episode([trace()])
    source.traces[0].nodes = nodes
    received = recorded_head(source)
    record = orjson.loads(received.take_record().json)
    assign_advantage(received.traces[0], 0.5)
    fold_trace_updates(record["traces"][0], stamp_batch([received], step=1))
    assert [node.get("advantages") for node in record["traces"][0]["nodes"]] == [None, [0.5, 0.5], None]


def test_folded_token_advantages_land_on_the_tokens_they_credit():
    """Per-token credit after a fork: a reader folds each branch's advantages onto its record,
    the shared answer taking them from the first branch, as it trained."""
    nodes = [
        vf.MessageNode(message=vf.UserMessage(content="q"), token_ids=[1], mask=[False]),
        vf.MessageNode(
            parent=0, message=vf.AssistantMessage(content="A"), sampled=True, token_ids=[2, 3], mask=[True, True]
        ),
        vf.MessageNode(
            parent=1, message=vf.AssistantMessage(content="B1"), sampled=True, token_ids=[4, 5], mask=[True, True]
        ),
        vf.MessageNode(parent=1, message=vf.AssistantMessage(content="B2"), sampled=True, token_ids=[6], mask=[True]),
    ]
    source = whole_episode([trace()])
    source.traces[0].nodes = nodes
    b1, b2 = (
        next(b.index for b in source.traces[0].branches if b.nodes[-1].message.content == leaf) for leaf in ("B1", "B2")
    )
    received = recorded_head(source)
    record = orjson.loads(received.take_record().json)
    assign_advantages(received.traces[0], b1, [1.0, 2.0, 3.0, 4.0])
    assign_advantages(received.traces[0], b2, [-1.0])
    fold_trace_updates(record["traces"][0], stamp_batch([received], step=1))
    assert [node.get("advantages") for node in record["traces"][0]["nodes"]] == [None, [1.0, 2.0], [3.0, 4.0], [-1.0]]


def test_sampled_episodes_keep_their_arrival_under_a_trace_sample(tmp_path, monkeypatch):
    group = next(f"g{i}" for i in range(100) if in_trace_sample(f"g{i}", 0.5))
    monitor = FileMonitor(FileMonitorConfig(trace_sample=0.5))
    asyncio.run(monitor.init(tmp_path))
    monkeypatch.setattr(monitors, "MONITORS", [monitor])

    arrived = [recorded_head(whole_episode([trace()], group=group), float_decimals=4) for _ in range(2)]
    for episode in arrived:
        asyncio.run(log_arrival(episode, "train", step=1))

    stream = get_annotations_dir(tmp_path) / "unknown"
    updates = [orjson.loads(line) for line in chunk_path(stream, 0).read_bytes().splitlines()]
    assert [update["trace_id"] for update in updates] == [episode.traces[0].id for episode in arrived]
    assert all(update["info"]["arrival"]["step"] == 1 for update in updates)


def test_a_timed_out_train_rollout_is_recorded_as_failed():
    timed_out = trace()
    timed_out.record_timeout("rollout")
    source = whole_episode([timed_out])
    stamp_dispatch(source, vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1)), "env", "g")

    head = recorded_head(source, float_decimals=4)
    record = head.take_record()
    assert not head.ok and head.traces[0].last_error.type == "Timeout"
    (written,) = orjson.loads(record.json)["traces"]
    assert written["ok"] is False and written["errors"][-1]["type"] == "Timeout"
    assert record.index["ok"] is False
