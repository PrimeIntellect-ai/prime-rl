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

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.env.encoding import pack, unpack
from prime_rl.env.heads import EpisodeHead, recorded_head
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path, get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_path
from prime_rl.monitors.file.traces.index import index_record, index_row
from prime_rl.monitors.file.traces.live import get_live_dir, read_live
from prime_rl.monitors.prime import RecordUpload
from prime_rl.orchestrator.algo.routing import assign_advantage
from prime_rl.orchestrator.annotations import stamp_arrival, stamp_batch
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
