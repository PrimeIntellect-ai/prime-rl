"""The v2 annotation format: per-token streams as raw float32 bytes in chunked binary
sidecars beside the JSONL records, written by the file monitor and read back the way the
dashboard does."""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.dashboard.server import trace_updates
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces import get_annotations_dir, get_index_path
from prime_rl.monitors.file.traces.chunks import ChunkedJsonl
from prime_rl.monitors.file.traces.update import fold_trace_updates, make_update, update_index_row


def _stream_values(update: dict, field: str) -> np.ndarray:
    """A record's decoded stream as float32, NaN where the reader saw no value."""
    values = [np.nan if value is None else value for value in update["branches"][0][field]]
    return np.asarray(values, dtype=np.float32)


def _write_records(stream: Path, records: list[dict[str, Any]]) -> None:
    """Append hand-built records and their index rows to a stream the monitor wrote."""
    jsonl = ChunkedJsonl(stream, 1 << 30, False)
    with open(get_index_path(stream), "ab") as index:
        for record in records:
            chunk, offset = jsonl.append(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            index.write(orjson.dumps(update_index_row(record, chunk, offset), option=orjson.OPT_APPEND_NEWLINE))
        jsonl.flush()


def _folded_advantages(update: dict) -> np.ndarray:
    """The advantages one update folds onto a four-token, two-sampled node, as float32."""
    trace = {"info": {}, "nodes": [{"token_ids": [9, 9, 3, 4], "mask": [False, False, True, True]}]}
    fold_trace_updates(trace, [update])
    return np.asarray(trace["nodes"][0]["advantages"], dtype=np.float32)


def test_written_streams_round_trip_bit_exact(tmp_path):
    logprobs = np.array([0.5, np.nan, -1.25, np.nan], dtype=np.float32)
    entropies = np.array([np.nan, 0.0, 2.5, 8.0], dtype=np.float32)
    plain = [0.1, 0.2, 0.3, 0.4]
    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(
        monitor.log_annotations(
            [
                make_update("masked", branches={0: {"trainer_logprobs": logprobs, "entropies": entropies}}),
                make_update("plain", branches={0: {"trainer_logprobs": plain}}),
            ]
        )
    )

    (masked,) = trace_updates(tmp_path, "masked")
    assert masked["branches"][0]["trainer_logprobs"] == [0.5, None, -1.25, None]
    assert masked["branches"][0]["entropies"] == [None, 0.0, 2.5, 8.0]
    assert _stream_values(masked, "trainer_logprobs").tobytes() == logprobs.tobytes()
    assert _stream_values(masked, "entropies").tobytes() == entropies.tobytes()

    (plain_update,) = trace_updates(tmp_path, "plain")
    assert _stream_values(plain_update, "trainer_logprobs").tobytes() == np.asarray(plain, dtype=np.float32).tobytes()


def test_written_refs_resolve_before_finalize(tmp_path):
    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_annotations([make_update("live", branches={0: {"trainer_logprobs": [1.0, 2.0]}})]))

    (update,) = trace_updates(tmp_path, "live")
    assert update["branches"][0]["trainer_logprobs"] == [1.0, 2.0]


def test_streams_larger_than_a_chunk_never_span_chunks(tmp_path):
    big = np.arange(100, dtype=np.float32)
    medium = np.arange(40, dtype=np.float32)
    small = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    monitor = FileMonitor(FileMonitorConfig(chunk_bytes=200, compress=False))
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(
        monitor.log_annotations(
            [
                make_update("a", branches={0: {"trainer_logprobs": big, "entropies": small}}),
                make_update("b", branches={0: {"trainer_logprobs": medium, "entropies": small}}),
                make_update("c", branches={0: {"trainer_logprobs": small}}),
            ]
        )
    )
    stream = get_annotations_dir(tmp_path) / "unknown"

    sizes = {int(path.stem): path.stat().st_size for path in stream.glob("*.bin")}
    assert sizes == {0: 400, 1: 192, 2: 16}
    refs: dict[tuple[str, str], dict] = {}
    for path in sorted(stream.glob("*.jsonl")):
        for line in path.read_bytes().splitlines():
            update = orjson.loads(line)
            for field, ref in update["branches"][0].items():
                if isinstance(ref, dict):
                    refs[(update["trace_id"], field)] = ref

    assert refs[("a", "trainer_logprobs")] == {"bin": 0, "offset": 0, "count": 100}
    assert sum(ref["bin"] == 0 for ref in refs.values()) == 1
    assert {ref["bin"] for ref in refs.values()} == {0, 1, 2}
    assert all(ref["offset"] + ref["count"] * 4 <= sizes[ref["bin"]] for ref in refs.values())

    written = {
        "a": {"trainer_logprobs": big, "entropies": small},
        "b": {"trainer_logprobs": medium, "entropies": small},
        "c": {"trainer_logprobs": small},
    }
    for trace_id, values in written.items():
        (update,) = trace_updates(tmp_path, trace_id)
        for field, expected in values.items():
            assert _stream_values(update, field).tobytes() == np.asarray(expected, dtype=np.float32).tobytes()


def test_relaunch_after_sealing_reads_both_runs(tmp_path):
    first = np.float32([0.5, np.nan, -1.25])
    second = np.float32([np.nan, 2.0])
    for trace_id, values in (("first", first), ("second", second)):
        monitor = FileMonitor(FileMonitorConfig())
        asyncio.run(monitor.init(tmp_path))
        asyncio.run(monitor.log_annotations([make_update(trace_id, branches={0: {"trainer_logprobs": values}})]))
        asyncio.run(monitor.finalize())
    monitor = FileMonitor(FileMonitorConfig())
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_annotations([make_update("info-only", info={"origin": 3})]))
    asyncio.run(monitor.finalize())

    stream = get_annotations_dir(tmp_path) / "unknown"
    assert sorted(path.name for path in stream.iterdir()) == [
        "00000.bin.zst",
        "00000.jsonl.zst",
        "00001.bin.zst",
        "00001.jsonl.zst",
        "00002.jsonl.zst",
    ]
    for trace_id, values in (("first", first), ("second", second)):
        (update,) = trace_updates(tmp_path, trace_id)
        assert _stream_values(update, "trainer_logprobs").tobytes() == values.tobytes()


def test_v1_record_folds_alongside_v2_records(tmp_path):
    monitor = FileMonitor(FileMonitorConfig(compress=False))
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(
        monitor.log_annotations(
            [make_update("v2", info={"origin": 2}, branches={0: {"advantages": np.float32([0.0, 0.0, 0.7, 0.7])}})]
        )
    )
    stream = get_annotations_dir(tmp_path) / "unknown"
    v1 = {
        "version": 1,
        "trace_id": "v1",
        "info": {"origin": 1},
        "branches": [{"index": 0, "advantages": [0.0, 0.0, 0.7, 0.7]}],
    }
    _write_records(stream, [v1])

    (v1_update,) = trace_updates(tmp_path, "v1")
    assert v1_update == v1
    (v2_update,) = trace_updates(tmp_path, "v2")

    expected = np.float32([0.7, 0.7])
    assert _folded_advantages(v1_update).tobytes() == expected.tobytes()
    assert _folded_advantages(v2_update).tobytes() == expected.tobytes()


def test_torn_ref_reads_as_an_absent_stream(tmp_path):
    monitor = FileMonitor(FileMonitorConfig(compress=False))
    asyncio.run(monitor.init(tmp_path))
    asyncio.run(monitor.log_annotations([make_update("good", branches={0: {"trainer_logprobs": np.float32([1.5])}})]))
    stream = get_annotations_dir(tmp_path) / "unknown"
    torn = {
        "version": 2,
        "trace_id": "torn",
        "info": {},
        "branches": [{"index": 0, "trainer_logprobs": {"bin": 0, "offset": 1 << 20, "count": 4}}],
    }
    _write_records(stream, [torn])

    (update,) = trace_updates(tmp_path, "torn")
    assert update["branches"][0]["trainer_logprobs"] is None

    trace = {"info": {}, "nodes": [{"token_ids": [1, 2, 3, 4], "mask": [True, True, True, True]}]}
    fold_trace_updates(trace, [update])
    assert "trainer_logprobs" not in trace["nodes"][0]
