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


def _build_rollout(*, example_id: int, reward: float, task: str, with_nodes: bool = True) -> Rollout:
    nodes = []
    if with_nodes:
        nodes = [
            vf.MessageNode(
                parent=None,
                message=vf.UserMessage(content=f"prompt-{example_id}"),
                token_ids=[1, 2, 3],
                mask=[False, False, False],
                logprobs=[],
            ),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content=f"completion-{example_id}"),
                sampled=True,
                token_ids=[4, 5],
                mask=[True, True],
                logprobs=[-0.1, -0.2],
            ),
        ]
    rollout = Rollout[vf.Task](
        task=vf.Task(idx=example_id, prompt=task),
        nodes=nodes,
        rewards={"reward": reward},
        metrics={"accuracy": reward},
    )
    rollout.env_name = "test-env"
    rollout.advantages = [reward / 2, reward / 2]
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
    assert rows[0]["prompt"] == ""
    assert json.loads(rows[1]["completion"])[-1]["content"] == "completion-202"


def test_rollouts_to_parquet_bytes_skips_rollouts_without_trajectory():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    parquet_bytes = monitor._rollouts_to_parquet_bytes(
        [
            _build_rollout(example_id=1, reward=1.0, task="task-a"),
            _build_rollout(example_id=2, reward=0.0, task="task-b", with_nodes=False),
        ],
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
