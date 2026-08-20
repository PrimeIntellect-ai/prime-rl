import uuid
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import (
    AdvRangeGateConfig,
    CurriculumConfig,
    DifficultyPoolConfig,
    DifficultyPoolSamplerConfig,
)
from prime_rl.orchestrator.curriculum import (
    AdvRangeGate,
    Curriculum,
    StandardSampler,
)
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import PreparedGroup
from prime_rl.transports.rollouts import TrainingSample


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def make_rollout(
    task: vf.Task,
    *,
    env_name: str = "test",
    reward: float = 0.0,
    advantages: list[float] | None = None,
) -> tuple[list[vf.Episode], PreparedGroup]:
    samples = []
    if advantages is not None:
        samples = [
            TrainingSample(
                token_ids=list(range(len(advantages))),
                mask=[True] * len(advantages),
                logprobs=[0.0] * len(advantages),
                temperatures=[1.0] * len(advantages),
                advantages=advantages,
                env_name=env_name,
            )
        ]
    trace = vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        rewards={"reward": vf.Reward(score=reward)},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task_key=task.key,
        task_hash=task.hash,
        group_id=str(uuid.uuid4()),
        policy_version=0,
        traces=[trace],
    )
    prepared = {trace.id: samples} if advantages is not None else {}
    return [episode], prepared


def test_default_curriculum_resumes_finite_and_infinite_tasksets() -> None:
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


def test_curriculum_requires_unique_finite_task_keys() -> None:
    task = make_task(0)
    with pytest.raises(ValueError, match="Task keys must be unique"):
        StandardSampler([task, task])


def test_train_source_composes_sampler_and_all_gates_with_state_and_metrics() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {"all": DifficultyPoolConfig(threshold=1.0, weight=1.0)}
    config = SimpleNamespace(
        ratio=1.0,
        curriculum=CurriculumConfig(
            sampler=DifficultyPoolSamplerConfig(pools=pools),
            gates={
                "reject": AdvRangeGateConfig(),
                "admit": AdvRangeGateConfig(reject_min=0.5, reject_max=0.5),
            },
        ),
    )
    env = SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)
    source = TrainSource([env])

    sampled = source.next_example()["task"]
    assert source.on_result(*make_rollout(sampled, reward=0.25, advantages=[0.0])) is False
    assert source.metrics() == {
        "curriculum/test/admission_rate": 0.0,
        "curriculum/test/sampler/pool/unseen": 2.0,
        "curriculum/test/sampler/pool/all": 1.0,
    }

    state = source.state_dict()
    restored = TrainSource([SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)])
    restored.load_state_dict(state)
    assert restored.curricula["test"].state_dict()["sampler"]["task_rewards"] == {sampled.key: 0.25}
    assert restored.curricula["test"].state_dict()["gates"] == {
        "reject": {},
        "admit": {},
    }


def test_difficulty_pools_stack_with_advantage_gate_and_resume_sampling() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {
        "hard": DifficultyPoolConfig(threshold=0.25, weight=0.0),
        "normal": DifficultyPoolConfig(threshold=0.75, weight=1.0),
        "easy": DifficultyPoolConfig(threshold=1.0, weight=0.0),
    }
    sampler_config = DifficultyPoolSamplerConfig(pools=pools, seed=7)
    gate_config = AdvRangeGateConfig()
    config = CurriculumConfig(
        sampler=sampler_config,
        gates={"zero_advantage": gate_config},
    )
    curriculum = Curriculum(config, tasks)
    rewards = {0: 0.1, 1: 0.5, 2: 0.9}
    decisions = []
    for index, task in enumerate(tasks):
        group, prepared = make_rollout(task, reward=rewards[task.data.idx], advantages=[float(index > 0)])
        decisions.append(curriculum.on_result(group, prepared))

    assert decisions == [False, True, True]
    assert curriculum.metrics() == {
        "sampler/pool/unseen": 0.0,
        "sampler/pool/hard": 1.0,
        "sampler/pool/normal": 1.0,
        "sampler/pool/easy": 1.0,
    }
    state = curriculum.state_dict()
    expected = [next(curriculum.sampler).key for _ in range(10)]
    assert set(expected) == {tasks[1].key}
    restored = Curriculum(config, tasks)
    restored.load_state_dict(state)
    assert [next(restored.sampler).key for _ in range(10)] == expected


def test_advantage_range_gate_generalizes_zero_advantage_rejection() -> None:
    task = make_task(0)
    zero_gate = AdvRangeGate(AdvRangeGateConfig())
    assert zero_gate.admit(*make_rollout(task, advantages=[0.0, 0.0])) is False
    assert zero_gate.admit(*make_rollout(task, advantages=[0.0, 0.2])) is True
    assert zero_gate.admit(*make_rollout(task)) is True

    tolerance_gate = AdvRangeGate(AdvRangeGateConfig(reject_min=-0.1, reject_max=0.1))
    assert tolerance_gate.admit(*make_rollout(task, advantages=[-0.05, 0.0, 0.05])) is False

    masked, prepared = make_rollout(task, advantages=[0.0, 0.5])
    next(iter(prepared.values()))[0].mask = [False, True]
    positive_gate = AdvRangeGate(AdvRangeGateConfig(reject_min=0.5, reject_max=0.5))
    assert positive_gate.admit(masked, prepared) is False
