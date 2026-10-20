import asyncio

import orjson

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces.live import get_live_dir


def test_live_traces_drop_trainer_only_node_fields(tmp_path):
    monitor = FileMonitor(FileMonitorConfig())
    node = {
        "token_ids": [1, 2, 3],
        "logprobs": [-0.1, -0.2],
        "mask": [False, True, True],
        "routed_experts": {"data": "AAAA", "shape": [3, 1, 2], "dtype": "uint8"},
        "sampling_mask": {"data": "AA"},
        "top_logprobs": {"counts": "AAAA", "ids": "AAAA", "values": "AAAA"},
    }
    event = {
        "delta": {"trace": "t1", "open": {"id": "t1"}, "nodes": [node], "routing_repairs": [1]},
        "dispatch": {"kind": "train", "group": "g1"},
    }

    async def run():
        await monitor.init(tmp_path)
        await monitor.log_live([event])
        await monitor.finalize()

    asyncio.run(run())

    (line,) = [line for line in (get_live_dir(tmp_path) / "t1.jsonl").read_bytes().splitlines() if line.strip()]
    written = orjson.loads(line)
    assert written["dispatch"] == event["dispatch"]
    assert "routing_repairs" not in written
    assert written["nodes"] == [{"token_ids": [1, 2, 3], "logprobs": [-0.1, -0.2], "mask": [False, True, True]}]
