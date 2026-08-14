import io
import json
from unittest.mock import Mock

import pyarrow.parquet as pq
import verifiers.v1 as vf

from prime_rl.monitors.prime import PrimeMonitor, group_episodes
from prime_rl.orchestrator.types import Rollout


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor.logger = Mock()
    return monitor


def _build_rollout(*, example_id: int, reward: float, env_name: str, episode_id: str = "") -> Rollout:
    """Build a v1 ``Rollout`` (message-graph trace). The user node carries the prompt and the
    assistant node the completion; the parquet ``completion`` column is the last branch's
    messages, ``trajectory`` is one message list per branch."""
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
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
    )
    rollout.env_name = env_name
    rollout.episode_id = episode_id
    rollout.ok = True
    # Per-token advantage stream (full-length-N): 0.0 on the 3 prompt tokens,
    # reward/2 on the 2 completion (mask-True) tokens.
    rollout.advantages = [0.0, 0.0, 0.0, reward / 2, reward / 2]
    return rollout


def test_group_episodes_links_rollouts_by_episode_id():
    rollouts = [
        _build_rollout(example_id=1, reward=1.0, env_name="task-a", episode_id="ep-1"),
        _build_rollout(example_id=1, reward=0.5, env_name="task-a", episode_id="ep-1"),
        _build_rollout(example_id=2, reward=0.0, env_name="task-b", episode_id="ep-2"),
        _build_rollout(example_id=3, reward=0.0, env_name="task-b"),  # no episode_id: own episode
    ]

    episodes = group_episodes(rollouts)

    assert [len(e.traces) for e in episodes] == [2, 1, 1]
    assert episodes[0].id == "ep-1"
    assert episodes[0].env.id == "task-a"
    assert all(e.ok for e in episodes)


def test_episodes_to_parquet_bytes_one_row_per_episode():
    monitor = _new_monitor()
    monitor.run_id = "run-123"

    episodes = group_episodes(
        [
            _build_rollout(example_id=101, reward=1.0, env_name="task-a", episode_id="ep-1"),
            _build_rollout(example_id=202, reward=0.0, env_name="task-b", episode_id="ep-2"),
        ]
    )
    parquet_bytes = monitor._episodes_to_parquet_bytes(episodes, step=7)

    assert parquet_bytes is not None
    rows = pq.read_table(io.BytesIO(parquet_bytes)).to_pylist()

    assert len(rows) == 2
    assert [row["problem_id"] for row in rows] == [101, 202]
    assert [row["sample_id"] for row in rows] == [0, 1]
    assert [row["env_name"] for row in rows] == ["task-a", "task-b"]
    assert all(row["run_id"] == "run-123" for row in rows)
    assert all(row["step"] == 7 for row in rows)
    # `completion` is the last branch's messages; the prompt user message lives in `trajectory`.
    assert json.loads(rows[1]["completion"])[0]["content"] == "completion-202"
    trajectory = json.loads(rows[0]["trajectory"])
    assert trajectory[0]["messages"][0]["content"] == "prompt-101"
    # The full native episode travels in info, the episode advantage on each branch.
    assert json.loads(rows[0]["info"])["native_wrapper"]["id"] == "ep-1"
    assert trajectory[0]["advantage"] == 0.5


def test_episodes_to_parquet_bytes_skips_episodes_without_trajectory():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    empty_rollout = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=2, prompt="missing-trajectory")),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
    )
    empty_rollout.env_name = "task-a"
    assert empty_rollout.branches == []

    episodes = group_episodes(
        [_build_rollout(example_id=1, reward=1.0, env_name="task-a", episode_id="ep-1"), empty_rollout]
    )
    parquet_bytes = monitor._episodes_to_parquet_bytes(episodes, step=3)

    assert parquet_bytes is not None
    rows = pq.read_table(io.BytesIO(parquet_bytes)).to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 1
    assert rows[0]["sample_id"] == 0


def test_sanitize_drops_non_finite_values_and_logs_paths():
    monitor = _new_monitor()

    payload = {
        "metrics": {"finite": 1.0, "nan": float("nan")},
        "values": [0.5, float("inf")],
    }

    sanitized = monitor._sanitize("metrics", payload)

    assert sanitized == {"metrics": {"finite": 1.0}, "values": [0.5]}
    monitor.logger.warning.assert_called_once_with(
        "Dropping 2 non-finite value(s) from Prime monitor metrics payload: metrics.nan, values[1]"
    )
