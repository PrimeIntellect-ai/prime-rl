import io
import json
from unittest.mock import Mock

import pyarrow.parquet as pq
import verifiers.v1 as vf
from verifiers.v1.episode import EnvInfo

from prime_rl.orchestrator.types import Rollout
from prime_rl.utils.monitor.prime import PrimeMonitor


def _new_monitor() -> PrimeMonitor:
    monitor = PrimeMonitor.__new__(PrimeMonitor)
    monitor._closed = True
    return monitor


def _build_rollout(
    *, example_id: int, reward: float, task: str, agent_name: str = "agent", trainable: bool = True
) -> Rollout:
    """Build a v1 ``Rollout`` with one user/assistant message branch."""
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
        agent=vf.AgentInfo(config=vf.AgentConfig(), name=agent_name, trainable=trainable),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
    )
    rollout.env_name = task
    # Per-token advantage stream (full-length-N): 0.0 on the 3 prompt tokens,
    # reward/2 on the 2 completion (mask-True) tokens.
    rollout.advantages = [0.0, 0.0, 0.0, reward / 2, reward / 2]
    return rollout


def _attach_episode(
    *rollouts: Rollout,
    episode_id: str,
    env_id: str = "task-a-v1",
    ok: bool = True,
    errors: list[vf.Error] | None = None,
) -> vf.WireEpisode:
    episode = vf.WireEpisode.model_construct(
        id=episode_id,
        env=EnvInfo(id=env_id),
        ok=ok,
        errors=errors or [],
        traces=list(rollouts),
    )
    for rollout in rollouts:
        rollout.episode_id = episode_id
        rollout.native_episode = episode
    return episode


def test_episodes_to_parquet_bytes_preserves_episode_rows_and_ids():
    monitor = _new_monitor()
    monitor.run_id = "run-123"

    first = _build_rollout(example_id=101, reward=1.0, task="task-a")
    second = _build_rollout(example_id=202, reward=0.0, task="task-b")
    _attach_episode(first, episode_id="episode-101")
    _attach_episode(second, episode_id="episode-202", env_id="task-b-v1")

    episodes = monitor._rollouts_to_episodes([first, second])
    parquet_bytes = monitor._episodes_to_parquet_bytes(episodes, step=7)

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


def test_effective_rollout_retains_fixed_and_errored_sibling_traces():
    monitor = _new_monitor()
    monitor.run_id = "run-456"

    fixed_trace = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=2, prompt="missing-trajectory")),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="judge", trainable=False),
    )
    trainable_trace = _build_rollout(
        example_id=3,
        reward=1.0,
        task="task-a",
        agent_name="solver",
    )
    error = vf.Error(type="JudgeError", message="judge failed after scoring")
    failed_trace = Rollout[vf.TaskData](
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=3, prompt="prompt-3")),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="critic", trainable=False),
        errors=[error],
    )
    _attach_episode(
        fixed_trace,
        trainable_trace,
        failed_trace,
        episode_id="multi-trace-episode",
        ok=False,
        errors=[error],
    )

    # The orchestrator still passes only effective rollouts. Its retained native envelope
    # carries the fixed and errored siblings without changing the rollout-based pipeline.
    episodes = monitor._rollouts_to_episodes([trainable_trace])
    parquet_bytes = monitor._episodes_to_parquet_bytes(episodes, step=3)

    assert parquet_bytes is not None

    table = pq.read_table(io.BytesIO(parquet_bytes))
    rows = table.to_pylist()

    assert len(rows) == 1
    assert rows[0]["problem_id"] == 3
    assert rows[0]["sample_id"] == 0
    assert rows[0]["reward"] == 1.0
    assert rows[0]["advantage"] == 0.5
    info = json.loads(rows[0]["info"])
    assert info["native_trace_index"] == 1
    assert info["native_wrapper"]["ok"] is False
    assert info["native_wrapper"]["errors"] == [{"type": "JudgeError", "message": "judge failed after scoring"}]
    assert [trace["agent"]["name"] for trace in info["native_wrapper"]["traces"]] == [
        "judge",
        "solver",
        "critic",
    ]


def test_rollout_without_native_envelope_uses_single_trace_fallback():
    monitor = _new_monitor()
    monitor.run_id = "run-legacy"
    rollout = _build_rollout(example_id=4, reward=0.25, task="legacy-env")

    episodes = monitor._rollouts_to_episodes([rollout])

    assert len(episodes) == 1
    assert episodes[0].id == rollout.id
    assert episodes[0].env.id == "legacy-env"
    assert episodes[0].traces == [rollout]
    assert "native_episode" not in rollout.model_dump()


def test_legacy_rollouts_with_same_episode_id_are_not_deduplicated():
    first = _build_rollout(example_id=4, reward=0.25, task="legacy-env")
    second = _build_rollout(example_id=4, reward=0.5, task="legacy-env")
    first.episode_id = second.episode_id = "legacy-group"

    episodes = PrimeMonitor._rollouts_to_episodes([first, second])

    assert len(episodes) == 2
    assert [episode.traces for episode in episodes] == [[first], [second]]


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


def test_log_samples_warns_and_skips_upload_when_serialization_fails():
    monitor = _new_monitor()
    monitor.is_master = True
    monitor.enabled = True
    monitor.config = Mock()
    monitor.config.log_extras.samples = True
    monitor.config.log_extras.interval = 1
    monitor.config.log_extras.sample_ratio = 1.0
    monitor.last_log_samples_step = -1
    monitor._pending_sample_steps = set()
    monitor.logger = Mock()
    monitor._episodes_to_parquet_bytes = Mock(side_effect=ValueError("invalid episode"))
    monitor._upload_samples_via_presigned_url = Mock()
    rollout = _build_rollout(example_id=5, reward=1.0, task="task-a")

    monitor.log_samples([rollout], step=1)

    monitor.logger.warning.assert_called_once_with(
        "Failed to build Prime monitor samples at step 1: ValueError: invalid episode"
    )
    monitor._upload_samples_via_presigned_url.assert_not_called()
