import io
import json
from unittest.mock import Mock

import pyarrow.parquet as pq
import verifiers.v1 as vf

from prime_rl.orchestrator.types import AgentGraph, TrainingTrace
from prime_rl.utils.monitor.prime import PrimeMonitor


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor._closed = True
    return monitor


def _build_graph(*, example_id: int, reward: float, task: str) -> AgentGraph:
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content=f"prompt-{example_id}"),
            token_ids=[1, 2, 3],
            mask=[False, False, False],
            logprobs=[0.0, 0.0, 0.0],
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content=f"completion-{example_id}"),
            token_ids=[4, 5],
            mask=[True, True],
            logprobs=[-0.1, -0.2],
            sampled=True,
        ),
    ]
    trace = TrainingTrace(
        task=vf.TraceTask(type="Task", data=vf.WireTaskData(idx=example_id, prompt=f"prompt-{example_id}")),
        nodes=nodes,
        rewards={"reward": reward},
    )
    # Per-token advantage stream (full-length-N): 0.0 on the 3 prompt tokens,
    # reward/2 on the 2 completion (mask-True) tokens.
    trace.advantages = [0.0, 0.0, 0.0, reward / 2, reward / 2]
    return AgentGraph(task=trace.task, traces=[trace], topology="single-agent", env_name=task)


def test_graphs_to_parquet_bytes_preserves_all_graphs_and_ids():
    monitor = _new_monitor()
    monitor.run_id = "run-123"

    parquet_bytes = monitor._graphs_to_parquet_bytes(
        [
            _build_graph(example_id=101, reward=1.0, task="task-a"),
            _build_graph(example_id=202, reward=0.0, task="task-b"),
        ],
        step=7,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 2
    assert [row["problem_id"] for row in rows] == [101, 202]
    assert [row["sample_id"] for row in rows] == [0, 1]
    assert all(row["run_id"] == "run-123" for row in rows)
    assert all(row["step"] == 7 for row in rows)
    # `completion` is the last branch's messages; the prompt user message lives in `trajectory`.
    assert json.loads(rows[1]["completion"])[0]["content"] == "completion-202"
    trajectory = json.loads(rows[0]["trajectory"])
    assert trajectory[0]["messages"][0]["content"] == "prompt-101"
    info = json.loads(rows[0]["info"])
    assert info["graph_id"]
    assert info["topology"] == "single-agent"


def test_graphs_to_parquet_bytes_skips_graphs_without_trajectory():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    graph_with_branches = _build_graph(example_id=1, reward=1.0, task="task-a")
    empty_trace = TrainingTrace(
        task=vf.TraceTask(type="Task", data=vf.WireTaskData(idx=2, prompt="missing-trajectory"))
    )
    graph_without_branches = AgentGraph(task=empty_trace.task, traces=[empty_trace])
    assert graph_without_branches.training_trace.branches == []

    parquet_bytes = monitor._graphs_to_parquet_bytes(
        [graph_with_branches, graph_without_branches],
        step=3,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 1
    assert rows[0]["sample_id"] == 0


def test_graphs_to_parquet_bytes_emits_each_trainable_trace():
    monitor = _new_monitor()
    monitor.run_id = "run-multi"
    graph = _build_graph(example_id=1, reward=1.0, task="task-a")
    graph.traces[0].agent = "proposer"
    solver = graph.traces[0].model_copy(deep=True)
    solver.id = "solver-trace"
    solver.agent = "solver"
    solver.parents = [graph.traces[0].id]
    graph.traces.append(solver)

    parquet_bytes = monitor._graphs_to_parquet_bytes([graph], step=1)

    assert parquet_bytes is not None
    rows = pq.read_table(io.BytesIO(parquet_bytes)).to_pylist()
    assert [json.loads(row["info"])["agent"] for row in rows] == ["proposer", "solver"]


def test_sanitize_json_payload_drops_non_finite_values_and_logs_paths():
    monitor = _new_monitor()
    monitor.logger = Mock()

    payload = {
        "metrics": {"finite": 1.0, "nan": float("nan")},
        "distributions": [0.5, float("inf")],
    }

    sanitized = monitor._sanitize_json_payload("metrics", payload)

    assert sanitized == {"metrics": {"finite": 1.0}, "distributions": [0.5]}
    monitor.logger.warning.assert_called_once_with(
        "Dropping 2 non-finite value(s) from Prime monitor metrics payload: metrics.nan, distributions[1]"
    )
