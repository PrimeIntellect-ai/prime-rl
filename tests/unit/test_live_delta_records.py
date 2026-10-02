import asyncio
from copy import deepcopy

import numpy as np
import orjson
import pytest
from verifiers.v1.graph import _decode_ndarray, _encode_ndarray
from verifiers.v1.serve import EpisodeAssembly

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces.live import get_live_dir, read_live


@pytest.mark.parametrize("cleanup", ["done", "discard"])
def test_live_log_roundtrip_preserves_content_and_training_routing(tmp_path, cleanup):
    routing = _encode_ndarray(np.zeros((3, 2, 4), dtype=np.uint8))
    replacement = _encode_ndarray(np.ones((1, 2, 4), dtype=np.uint8))
    first = {
        "trace": "t",
        "open": {"id": "t"},
        "nodes": [
            {
                "message": {"role": "assistant", "content": "answer"},
                "semantic_parents": [],
                "token_ids": [1, 2, 3],
                "routed_experts": routing,
                "sampling_mask": {"ids": routing, "counts": replacement},
            }
        ],
        "calls": [{"usage": {"output_tokens": 3}}],
    }
    second = {
        "trace": "t",
        "routing_repairs": {0: replacement},
        "links": {0: [{"node": 1, "kind": "subagent_return"}]},
        "set": {"ok": True, "rewards": {"correct": 1.0}},
    }
    before = deepcopy([first, second])
    dispatch = {"id": "episode", "env": "test"}

    async def run():
        monitor = FileMonitor(FileMonitorConfig())
        await monitor.init(tmp_path)
        try:
            await monitor.log_live([{"delta": first, "dispatch": dispatch}])
            await monitor.log_live([{"delta": second, "dispatch": dispatch}])
            path = get_live_dir(tmp_path) / "t.jsonl"
            for line in path.read_bytes().splitlines():
                if line.strip():
                    record = orjson.loads(line)
                    assert "routing_repairs" not in record
                    for node in record.get("nodes", []):
                        assert "routed_experts" not in node
                        assert "sampling_mask" not in node
            saved_dispatch, trace = read_live(path)
            assert saved_dispatch == dispatch
            assert trace["nodes"][0]["message"]["content"] == "answer"
            assert trace["nodes"][0]["token_ids"] == [1, 2, 3]
            assert trace["nodes"][0]["semantic_parents"] == second["links"][0]
            assert trace["calls"] == first["calls"]
            assert trace["rewards"] == {"correct": 1.0}
            event = {"done": "t"} if cleanup == "done" else {"delta": {"trace": "t", "discard": True}}
            await monitor.log_live([event])
            assert not path.exists()
        finally:
            await monitor.finalize()

    asyncio.run(run())
    assert [first, second] == before
    assembly = EpisodeAssembly()
    assembly.apply(first)
    assembly.apply(second)
    actual = _decode_ndarray(assembly.traces["t"]["nodes"][0]["routed_experts"])
    np.testing.assert_array_equal(actual, np.concatenate([np.zeros((2, 2, 4)), np.ones((1, 2, 4))]))
