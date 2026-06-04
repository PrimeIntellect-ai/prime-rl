import base64
import io
import json
from pathlib import Path
from unittest.mock import Mock

import pyarrow.parquet as pq

from prime_rl.utils.monitor.prime import PrimeMonitor


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor._closed = True
    return monitor


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
        "timing": {"generation": {"start": 0.0, "end": 12.5, "duration": 12.5}},
    }


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
    assert json.loads(rows[0]["prompt"])[0]["content"] == "prompt-101"
    assert json.loads(rows[1]["completion"])[0]["content"] == "completion-202"


def test_rollouts_to_parquet_bytes_skips_rollouts_without_trajectory():
    monitor = _new_monitor()
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


def test_rollouts_to_parquet_bytes_inlines_local_image_urls_without_mutating(tmp_path: Path):
    monitor = _new_monitor()
    monitor.run_id = "run-images"
    image_path = tmp_path / "sample.jpg"
    image_bytes = b"jpeg-bytes"
    image_path.write_bytes(image_bytes)
    file_url = image_path.as_uri()
    rollout = _build_rollout(example_id=1, reward=1.0, task="image-task")
    image_part = {"type": "image_url", "image_url": {"url": file_url}}
    rollout["prompt"] = [{"role": "user", "content": [image_part]}]
    rollout["completion"] = [{"role": "assistant", "content": [image_part]}]
    rollout["trajectory"][0]["prompt"] = [{"role": "user", "content": [image_part]}]

    parquet_bytes = monitor._rollouts_to_parquet_bytes([rollout], step=9)

    assert parquet_bytes is not None
    row = pq.read_table(io.BytesIO(parquet_bytes)).to_pylist()[0]
    expected_url = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode("ascii")
    assert json.loads(row["prompt"])[0]["content"][0]["image_url"]["url"] == expected_url
    assert json.loads(row["completion"])[0]["content"][0]["image_url"]["url"] == expected_url
    assert json.loads(row["trajectory"])[0]["prompt"][0]["content"][0]["image_url"]["url"] == expected_url
    assert rollout["prompt"][0]["content"][0]["image_url"]["url"] == file_url


def test_rollouts_to_parquet_bytes_leaves_large_local_image_urls(tmp_path: Path):
    monitor = _new_monitor()
    monitor.run_id = "run-large-image"
    image_path = tmp_path / "large.png"
    image_path.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    file_url = image_path.as_uri()
    rollout = _build_rollout(example_id=1, reward=1.0, task="image-task")
    rollout["prompt"] = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": file_url}}]}]

    parquet_bytes = monitor._rollouts_to_parquet_bytes([rollout], step=9)

    assert parquet_bytes is not None
    row = pq.read_table(io.BytesIO(parquet_bytes)).to_pylist()[0]
    assert json.loads(row["prompt"])[0]["content"][0]["image_url"]["url"] == file_url


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
