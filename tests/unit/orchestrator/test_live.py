import asyncio

import numpy as np
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.serve.delta import DeltaStreamer, pack, unpack
from verifiers.v1.trace import TraceTask
from verifiers.v1.types import AssistantMessage, UserMessage

from prime_rl.configs.monitors import FileMonitorConfig
from prime_rl.monitors.file.monitor import FileMonitor
from prime_rl.monitors.file.traces.live import LiveFolds, live_path
from prime_rl.orchestrator import live


def test_live_file_drops_routed_experts_and_folds(tmp_path):
    trace = vf.Trace(
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=TraceTask(type="TaskData", data=vf.TaskData(idx=0)),
    )
    trace.nodes.append(MessageNode(parent=None, message=UserMessage(content="q")))
    trace.nodes.append(
        MessageNode(
            parent=0,
            message=AssistantMessage(content="a"),
            sampled=True,
            token_ids=[1, 2, 3],
            mask=[True, True, True],
            logprobs=[-0.1, -0.2, -0.3],
            routed_experts=np.arange(6, dtype=np.uint8).reshape(3, 2, 1),
        )
    )
    deltas = []

    async def send(delta: dict) -> None:
        deltas.append(unpack(pack(delta)))

    async def stream_and_write() -> None:
        streamer = DeltaStreamer(lambda: [trace], send)
        await streamer.flush()
        # the next prefill repairs the last routing row of an already sent node
        trace.nodes[1].routed_experts = np.array([[[0], [1]], [[2], [3]], [[9], [9]]], dtype=np.uint8)
        await streamer.flush()
        monitor = FileMonitor(FileMonitorConfig())
        monitor.output_dir = tmp_path
        monitor._live_cleared = False
        await monitor.log_live([{"delta": live.without_payloads(d), "dispatch": {"id": "x"}} for d in deltas])

    asyncio.run(stream_and_write())
    assert "routing_repairs" in deltas[1]

    path = live_path(tmp_path, trace.id)
    assert b"routed_experts" not in path.read_bytes()
    dispatch, folded = LiveFolds().read(path)
    assert dispatch == {"id": "x"}
    assert folded["nodes"][1]["token_ids"] == [1, 2, 3]
