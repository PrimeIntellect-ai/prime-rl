import io
import json

import pyarrow.parquet as pq

from prime_rl.utils.monitor.prime import PrimeMonitor


def _build_rollout(*, example_id: int, reward: float, task: str) -> dict:
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": f"prompt-{example_id}"}],
        "completion": [{"role": "assistant", "content": f"completion-{example_id}"}],
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": f"prompt-{example_id}"}],
                "completion": [{"role": "assistant", "content": f"completion-{example_id}"}],
                "reward": reward,
                "advantage": reward / 2,
                "extras": {"source": "test"},
                "tokens": {
                    "prompt_ids": [1, 2, 3],
                    "completion_ids": [4, 5],
                },
            }
        ],
        "answer": f"answer-{example_id}",
        "task": task,
        "info": {"difficulty": "easy"},
        "reward": reward,
        "advantage": reward / 2,
        "metrics": {"accuracy": reward},
        "timing": {"generation_ms": 12.5},
    }


def test_rollouts_to_parquet_bytes_preserves_all_rollouts_and_ids():
    monitor = PrimeMonitor.__new__(PrimeMonitor)
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
    assert json.loads(rows[0]["prompt"])[0]["content"] == "prompt-101"
    assert json.loads(rows[1]["completion"])[0]["content"] == "completion-202"


def test_rollouts_to_parquet_bytes_skips_rollouts_without_trajectory():
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor.run_id = "run-456"

    parquet_bytes = monitor._rollouts_to_parquet_bytes(
        [
            _build_rollout(example_id=1, reward=1.0, task="task-a"),
            {
                "example_id": 2,
                "prompt": [{"role": "user", "content": "missing-trajectory"}],
                "completion": [{"role": "assistant", "content": "ignored"}],
                "trajectory": [],
            },
        ],
        step=3,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 1
    assert rows[0]["sample_id"] == 0
