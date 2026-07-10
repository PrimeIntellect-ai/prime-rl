import io
import json
from unittest.mock import Mock

import pyarrow.parquet as pq
import verifiers.v1 as vf

from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.monitor.prime import PrimeMonitor


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor._closed = True
    return monitor


def _build_rollout(*, example_id: int, reward: float, task: str) -> Rollout:
    """Build a v1 ``Rollout`` (message-graph trace). The user node carries the prompt and the
    assistant node the completion; ``_rollouts_to_parquet_bytes`` reads the conversation off the
    branches (its ``completion`` column is the last branch's messages, ``trajectory`` is one
    message list per branch)."""
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
    rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=example_id, prompt=f"prompt-{example_id}")),
        nodes=nodes,
        rewards={"reward": reward},
    )
    rollout.env_name = task
    # Per-token advantage stream (full-length-N): 0.0 on the 3 prompt tokens,
    # reward/2 on the 2 completion (mask-True) tokens.
    rollout.advantages = [0.0, 0.0, 0.0, reward / 2, reward / 2]
    return rollout


def test_rollouts_to_parquet_bytes_preserves_all_rollouts_and_ids():
    monitor = _new_monitor()
    monitor.run_id = "run-123"

    parquet_bytes = monitor._rollouts_to_parquet_bytes(
        [
            _build_rollout(example_id=101, reward=1.0, task="task-a"),
            _build_rollout(example_id=202, reward=0.0, task="task-b"),
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


def test_rollouts_to_parquet_bytes_skips_rollouts_without_trajectory():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    rollout_with_branches = _build_rollout(example_id=1, reward=1.0, task="task-a")
    rollout_without_branches = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=2, prompt="missing-trajectory"))
    )
    assert rollout_without_branches.branches == []

    parquet_bytes = monitor._rollouts_to_parquet_bytes(
        [rollout_with_branches, rollout_without_branches],
        step=3,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 1
    assert rows[0]["sample_id"] == 0


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
