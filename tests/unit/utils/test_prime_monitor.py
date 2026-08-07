import io
import json
from unittest.mock import Mock

import pyarrow.parquet as pq
import verifiers.v1 as vf
from verifiers.v1.configs.agent import WireAgentConfig

from prime_rl.orchestrator.types import Episode, Rollout
from prime_rl.utils.monitor.prime import PrimeMonitor


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor._closed = True
    return monitor


def _build_rollout(
    *, example_id: int, reward: float, task: str, agent_name: str = "agent", trainable: bool = True
) -> Rollout:
    """Build a v1 ``Rollout`` (message-graph trace). The user node carries the prompt and the
    assistant node the completion; ``_episodes_to_parquet_bytes`` reads the conversation off the
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
        agent=vf.AgentInfo(config=WireAgentConfig(), name=agent_name, trainable=trainable),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
    )
    rollout.env_name = task
    rollout.assign_advantages(reward / 2)  # over the 2 completion (mask-True) tokens
    return rollout


def _episode(*rollouts: Rollout, episode_id: str, env_name: str = "task-a") -> Episode:
    """The unit the monitor uploads: one episode, represented by one native row."""
    return Episode.model_construct(
        id=episode_id,
        traces=list(rollouts),
        env=vf.EnvInfo(id=f"{env_name}-v1", name=env_name),
        run=vf.TrainRunInfo(id="training-run", metadata=vf.TrainMetadata(step=7)),
        ok=True,
    )


def test_episodes_to_parquet_bytes_preserves_episode_rows_and_ids():
    monitor = _new_monitor()
    monitor.run_id = "run-123"

    parquet_bytes = monitor._episodes_to_parquet_bytes(
        [
            _episode(
                _build_rollout(example_id=101, reward=1.0, task="task-a"),
                episode_id="episode-101",
            ),
            _episode(
                _build_rollout(example_id=202, reward=0.0, task="task-b"),
                episode_id="episode-202",
                env_name="task-b",
            ),
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
    infos = [json.loads(row["info"]) for row in rows]
    assert [info["native_wrapper"]["id"] for info in infos] == ["episode-101", "episode-202"]
    assert all(info["native_trace_index"] == 0 for info in infos)
    assert all(len(info["native_wrapper"]["traces"]) == 1 for info in infos)
    assert infos[0]["native_wrapper"]["run"]["id"] == "training-run"


def test_episodes_to_parquet_bytes_uses_trainable_summary_and_preserves_all_traces():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    fixed_trace_without_branches = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=2, prompt="missing-trajectory")),
        agent=vf.AgentInfo(config=WireAgentConfig(), name="judge", trainable=False),
    )
    trainable_trace = _build_rollout(
        example_id=3,
        reward=1.0,
        task="task-a",
        agent_name="solver",
    )
    assert fixed_trace_without_branches.branches == []

    parquet_bytes = monitor._episodes_to_parquet_bytes(
        [
            _episode(
                fixed_trace_without_branches,
                trainable_trace,
                episode_id="multi-trace-episode",
            )
        ],
        step=3,
    )

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 3
    assert rows[0]["sample_id"] == 0
    assert rows[0]["reward"] == 1.0
    assert rows[0]["advantage"] == 0.5
    assert json.loads(rows[0]["completion"])[0]["content"] == "completion-3"
    info = json.loads(rows[0]["info"])
    assert info["native_trace_index"] == 1
    assert [trace["agent"]["name"] for trace in info["native_wrapper"]["traces"]] == [
        "judge",
        "solver",
    ]
    assert info["native_wrapper"]["traces"][0]["nodes"] == []


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
