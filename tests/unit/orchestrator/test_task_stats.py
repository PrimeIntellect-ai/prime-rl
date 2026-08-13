import verifiers.v1 as vf

from prime_rl.orchestrator.task_stats import DECAY, PRIOR, TaskStats, task_key
from prime_rl.orchestrator.types import Rollout


def make_rollout(
    *, reward: float | None = None, env_name: str = "env", idx: int = 0, ok: bool = True, trainable: bool = True
) -> Rollout:
    rollout = Rollout(
        task=vf.TraceTask(type="Task", data=vf.TaskData(idx=idx, prompt=f"task {idx}")),
        agent=vf.AgentInfo(config=vf.AgentConfig(), trainable=trainable),
        ok=ok,
    )
    if not ok:
        rollout.errors = [vf.Error(type="TestError", message="boom")]
    if reward is not None:
        rollout.rewards = {"main": vf.Reward(score=reward, weight=1.0)}
    rollout.env_name = env_name
    return rollout


def test_task_key_is_canonical_and_content_sensitive():
    assert task_key({"a": 1, "b": 2}) == task_key({"b": 2, "a": 1})
    assert task_key({"a": 1}) != task_key({"a": 2})


def test_observe_accumulates_discounted_evidence_per_role():
    stats = TaskStats()
    group = [make_rollout(reward=1.0), make_rollout(reward=0.0)]
    stats.observe(group)

    key = task_key(group[0].task.data.model_dump(mode="json"))
    stat = stats.stats["env"][key]["agent"]
    assert stat.s == 1.0 and stat.f == 1.0 and stat.visits == 1
    alpha, beta = PRIOR
    assert stat.p_hat == (alpha + 1.0) / (alpha + beta + 2.0)

    # A second all-success group: prior counts decay, new evidence lands whole.
    stats.observe([make_rollout(reward=1.0), make_rollout(reward=1.0)])
    assert stat.s == DECAY * 1.0 + 2.0 and stat.f == DECAY * 1.0
    assert stat.visits == 2


def test_errored_rollouts_update_tick_counters_but_not_evidence():
    stats = TaskStats()
    stats.observe([make_rollout(ok=False), make_rollout(ok=False)])
    assert stats.stats == {}
    metrics = stats.metrics({"env": 10})
    assert metrics["sampler/env/groups_observed"] == 1.0
    assert metrics["sampler/env/realized_signal_rate"] == 0.0
    assert metrics["sampler/env/pool/unseen"] == 10.0
    # Tick counters drain on read; snapshot metrics persist.
    assert "sampler/env/groups_observed" not in stats.metrics({"env": 10})


def test_signal_rate_reads_nonzero_advantages():
    stats = TaskStats()
    signal = [make_rollout(reward=1.0), make_rollout(reward=0.0)]
    signal[0].advantages = [0.5]
    signal[1].advantages = [-0.5]
    degenerate = [make_rollout(reward=1.0, idx=1), make_rollout(reward=1.0, idx=1)]
    for rollout in degenerate:
        rollout.advantages = [0.0]
    stats.observe(signal)
    stats.observe(degenerate)
    metrics = stats.metrics({"env": None})
    assert metrics["sampler/env/realized_signal_rate"] == 0.5
    assert "sampler/env/coverage" not in metrics  # infinite taskset


def test_state_dict_roundtrip():
    stats = TaskStats()
    stats.observe([make_rollout(reward=1.0), make_rollout(reward=0.0)])
    restored = TaskStats()
    restored.load_state_dict(stats.state_dict())
    key = task_key(make_rollout().task.data.model_dump(mode="json"))
    assert restored.stats["env"][key]["agent"] == stats.stats["env"][key]["agent"]
