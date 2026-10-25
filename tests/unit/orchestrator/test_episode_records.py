"""Episodes decoded from the env-server wire hold per-token streams as numpy arrays (read-only
views into the received bytes). Every record written to disk or uploaded must still carry
plain JSON lists."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import orjson
import verifiers.v1 as vf
from prime_runs.sinks.base import to_mapping
from verifiers.v1.serve.delta import DeltaStreamer, EpisodeAssembly, TraceSummary, dump, pack, unpack

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_annotations_dir, get_trace_stream
from prime_rl.monitors.file.traces.chunks import chunk_path
from prime_rl.monitors.file.traces.live import get_live_dir, read_live
from prime_rl.monitors.prime import log_episode_records
from prime_rl.orchestrator.algo.routing import assign_advantages
from prime_rl.orchestrator.annotations import stamp_batch
from tests.unit.orchestrator.test_train_sink import episode, trace


async def _noop(_):
    return None


def _over_the_wire(source: vf.Episode) -> tuple[vf.WireEpisode, list[dict]]:
    """The episode as the orchestrator receives it, with the deltas that carried it."""
    traces = source.traces
    assembly = EpisodeAssembly()
    deltas = []
    for _, delta, _ in DeltaStreamer(lambda: traces, _noop).diff():
        delta = unpack(pack(delta))
        assembly.apply(delta)
        deltas.append(delta)
    summaries = [TraceSummary(id=t.id, nodes=len(t.nodes), calls=len(t.calls)) for t in traces]
    head = dump(source, exclude={"traces"})
    return vf.WireEpisode.model_validate(assembly.finish(head, summaries)), deltas


def _assert_plain_streams(node: dict) -> None:
    for name in ("token_ids", "mask", "is_content", "logprobs", "advantages", "reference_logprobs"):
        if node.get(name) is not None:
            assert isinstance(node[name], list), name


def test_received_episode_writes_plain_records(tmp_path):
    received, deltas = _over_the_wire(episode([trace()]))
    nodes = received.traces[0].nodes
    assert isinstance(nodes[2].token_ids, np.ndarray) and not nodes[2].token_ids.flags.writeable
    assign_advantages(received.traces[0], 0.7)

    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_episodes([received], step=1, kind="train", subset="all"))
    asyncio.run(monitor.log_live([{"delta": delta, "dispatch": {"id": "d"}} for delta in deltas]))
    asyncio.run(monitor.log_annotations(stamp_batch([received], step=1)))

    # The trace stream: the episode record, as every reader of the run directory parses it
    (line,) = chunk_path(get_trace_stream(tmp_path), 0).read_bytes().splitlines()
    record = orjson.loads(line)
    for node in record["traces"][0]["nodes"]:
        _assert_plain_streams(node)
    assert record["traces"][0]["nodes"][2]["token_ids"] == [3, 4]
    assert record["traces"][0]["nodes"][2]["logprobs"] == [-0.1, -0.2]

    # The live trace file folds back to the same tokens
    (live,) = get_live_dir(tmp_path).glob("*.jsonl")
    _, live_trace = read_live(live)
    for node in live_trace["nodes"]:
        _assert_plain_streams(node)
    assert [node["token_ids"] for node in live_trace["nodes"]] == [[1], [2], [3, 4]]

    # The annotation stream carries the branch advantages as numbers
    annotations = chunk_path(get_annotations_dir(tmp_path) / "unknown", 0)
    updates = [orjson.loads(line) for line in annotations.read_bytes().splitlines()]
    branch_advantages = [branch["advantages"] for update in updates for branch in update["branches"]]
    assert branch_advantages == [[0.0, 0.0, 0.7, 0.7]]


def test_received_episode_uploads_plain_json():
    received, _ = _over_the_wire(episode([trace()]))
    uploaded = []
    log_episode_records(SimpleNamespace(log_episodes=uploaded.extend), [received])

    # Prime Traces: the episode record, encoded with the SDK's strict JSON encoder
    record = json.loads(json.dumps(to_mapping(uploaded[0]), allow_nan=False))
    for node in record["traces"][0]["nodes"]:
        _assert_plain_streams(node)
    assert record["traces"][0]["nodes"][2]["token_ids"] == [3, 4]
