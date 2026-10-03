import uuid
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import DifficultyPoolConfig, DifficultyPoolSamplerConfig
from prime_rl.orchestrator.samplers import DifficultyPoolSampler, StandardSampler
from prime_rl.orchestrator.train_source import TrainSource


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def make_rollout(task: vf.Task, *, env_name: str = "test", reward: float = 0.0) -> list[vf.Episode]:
    trace = vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[],
        rewards={"reward": vf.Reward(score=reward)},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id=str(uuid.uuid4())),
        traces=[trace],
    )
    return [episode]


def test_standard_sampler_resumes_finite_and_infinite_tasksets() -> None:
    tasks = [make_task(i) for i in range(5)]
    finite = StandardSampler(tasks)
    for _ in range(3):
        next(finite)
    state = finite.state_dict()
    expected = [next(finite).key for _ in range(4)]

    restored = StandardSampler(tasks)
    restored.load_state_dict(state)
    assert [next(restored).key for _ in range(4)] == expected

    def task_stream() -> Iterator[vf.Task]:
        yield from (make_task(i) for i in range(10))

    infinite = StandardSampler(task_stream())
    next(infinite)
    next(infinite)
    restored_infinite = StandardSampler(task_stream())
    restored_infinite.load_state_dict(infinite.state_dict())
    assert next(restored_infinite).key == make_task(2).key


def test_sampler_requires_unique_finite_task_keys() -> None:
    task = make_task(0)
    with pytest.raises(ValueError, match="Task keys must be unique"):
        StandardSampler([task, task])


def test_train_source_observes_sampler_with_state_and_metrics() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {"all": DifficultyPoolConfig(threshold=1.0, weight=1.0)}
    config = SimpleNamespace(ratio=1.0, sampler=DifficultyPoolSamplerConfig(pools=pools))
    env = SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)
    source = TrainSource([env])

    sampled = source.next_task(step=1).task
    source.observe(make_rollout(sampled, reward=0.25))
    assert source.metrics() == {"sampler/test/pool/unseen": 2.0, "sampler/test/pool/all": 1.0}

    state = source.state_dict()
    state["envs"]["test"]["gates"] = {}
    restored = TrainSource([SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)])
    restored.load_state_dict(state)
    assert restored.samplers["test"].task_rewards == {sampled.key: 0.25}


def test_difficulty_pools_resume_sampling() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {
        "hard": DifficultyPoolConfig(threshold=0.25, weight=0.0),
        "normal": DifficultyPoolConfig(threshold=0.75, weight=1.0),
        "easy": DifficultyPoolConfig(threshold=1.0, weight=0.0),
    }
    config = DifficultyPoolSamplerConfig(pools=pools, seed=7)
    sampler = DifficultyPoolSampler(config, tasks)
    rewards = {0: 0.1, 1: 0.5, 2: 0.9}
    for task in tasks:
        sampler.observe(make_rollout(task, reward=rewards[task.data.idx]))

    assert sampler.metrics() == {
        "pool/unseen": 0.0,
        "pool/hard": 1.0,
        "pool/normal": 1.0,
        "pool/easy": 1.0,
    }
    state = sampler.state_dict()
    expected = [next(sampler).key for _ in range(10)]
    assert set(expected) == {tasks[1].key}
    restored = DifficultyPoolSampler(config, tasks)
    restored.load_state_dict(state)
    assert [next(restored).key for _ in range(10)] == expected
