from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import CurriculumComponentConfig, CurriculumConfig
from prime_rl.orchestrator.curriculum import (
    AdmissionGate,
    Curriculum,
    CurriculumResult,
    DifficultyPools,
    OnlineDifficultyFiltering,
    StandardSampler,
)
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Rollout
from prime_rl.transport import TrainingSample


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def make_rollout(
    task: vf.Task,
    *,
    env_name: str = "test",
    reward: float = 0.0,
    advantages: list[float] | None = None,
) -> Rollout:
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
    return Rollout(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        env_name=env_name,
        rewards={"reward": vf.Reward(score=reward)},
        advantages=advantages,
        samples=samples,
        ok=True,
    )


class CountingSampler(StandardSampler):
    def __init__(self, tasks):
        super().__init__(tasks)
        self.seen = 0

    def observe(self, result: CurriculumResult) -> None:
        self.seen += 1

    def state_dict(self) -> dict:
        return super().state_dict() | {"seen": self.seen}

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.seen = state_dict["seen"]

    def metrics(self) -> dict[str, float]:
        return {"seen": float(self.seen)}


class CountingGate(AdmissionGate):
    def __init__(self, *, decision: bool):
        self.decision = decision
        self.seen = 0

    def admit(self, result: CurriculumResult) -> bool:
        self.seen += 1
        return self.decision

    def state_dict(self) -> dict:
        return {"seen": self.seen}

    def load_state_dict(self, state_dict: dict) -> None:
        self.seen = state_dict["seen"]

    def metrics(self) -> dict[str, float]:
        return {"seen": float(self.seen)}


def test_default_curriculum_resumes_finite_and_infinite_tasksets() -> None:
    tasks = [make_task(i) for i in range(5)]
    finite = StandardSampler(tasks)
    for _ in range(3):
        finite.sample()
    state = finite.state_dict()
    expected = [finite.sample().key for _ in range(4)]

    restored = StandardSampler(tasks)
    restored.load_state_dict(state)
    assert [restored.sample().key for _ in range(4)] == expected

    def task_stream() -> Iterator[vf.Task]:
        yield from (make_task(i) for i in range(10))

    infinite = StandardSampler(task_stream())
    infinite.sample()
    infinite.sample()
    restored_infinite = StandardSampler(task_stream())
    restored_infinite.load_state_dict(infinite.state_dict())
    assert restored_infinite.sample().key == make_task(2).key


def test_curriculum_requires_unique_finite_task_keys() -> None:
    task = make_task(0)
    with pytest.raises(ValueError, match="Task keys must be unique"):
        StandardSampler([task, task])


def test_train_source_composes_sampler_and_all_gates_with_state_and_metrics() -> None:
    tasks = [make_task(i) for i in range(3)]
    config = SimpleNamespace(
        ratio=1.0,
        curriculum=CurriculumConfig(
            sampler=CurriculumComponentConfig(import_path=f"{__name__}.CountingSampler"),
            gates={
                "reject": CurriculumComponentConfig(import_path=f"{__name__}.CountingGate", kwargs={"decision": False}),
                "observe": CurriculumComponentConfig(import_path=f"{__name__}.CountingGate", kwargs={"decision": True}),
            },
        ),
    )
    env = SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)
    source = TrainSource([env])

    sampled = source.next_example()["task"]
    assert source.on_result([make_rollout(sampled)]) is False
    assert source.metrics() == {
        "curriculum/test/admission_rate": 0.0,
        "curriculum/test/sampler/seen": 1.0,
        "curriculum/test/gate/reject/seen": 1.0,
        "curriculum/test/gate/observe/seen": 1.0,
    }

    state = source.state_dict()
    restored = TrainSource([SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)])
    restored.load_state_dict(state)
    assert restored.curricula["test"].state_dict()["sampler"]["seen"] == 1
    assert restored.curricula["test"].state_dict()["gates"] == {
        "reject": {"seen": 1},
        "observe": {"seen": 1},
    }


def test_difficulty_pools_stack_with_online_difficulty_filtering_and_resume_sampling() -> None:
    tasks = [make_task(i) for i in range(3)]
    curriculum = Curriculum(
        DifficultyPools(tasks, seed=7),
        {"online_difficulty": OnlineDifficultyFiltering()},
    )
    rewards = {0: 0.1, 1: 0.5, 2: 0.9}
    decisions = []
    for index in range(len(tasks)):
        task = curriculum.sample()
        rollout = make_rollout(task, reward=rewards[task.data.idx], advantages=[float(index > 0)])
        decisions.append(curriculum.on_result(CurriculumResult.from_rollouts([rollout])))

    assert decisions == [False, True, True]
    assert curriculum.metrics() == {
        "sampler/pool/unseen": 0.0,
        "sampler/pool/hard": 1.0,
        "sampler/pool/medium": 1.0,
        "sampler/pool/easy": 1.0,
    }
    state = curriculum.state_dict()
    expected = [curriculum.sample().key for _ in range(10)]
    restored = Curriculum(
        DifficultyPools(tasks, seed=7),
        {"online_difficulty": OnlineDifficultyFiltering()},
    )
    restored.load_state_dict(state)
    assert [restored.sample().key for _ in range(10)] == expected


def test_online_difficulty_filtering_generalizes_zero_advantage_rejection() -> None:
    task = make_task(0)
    zero_gate = OnlineDifficultyFiltering()
    assert zero_gate.admit(CurriculumResult.from_rollouts([make_rollout(task, advantages=[0.0, 0.0])])) is False
    assert zero_gate.admit(CurriculumResult.from_rollouts([make_rollout(task, advantages=[0.0, 0.2])])) is True
    assert zero_gate.admit(CurriculumResult.from_rollouts([make_rollout(task)])) is True

    tolerance_gate = OnlineDifficultyFiltering(reject_min=-0.1, reject_max=0.1)
    assert (
        tolerance_gate.admit(CurriculumResult.from_rollouts([make_rollout(task, advantages=[-0.05, 0.0, 0.05])]))
        is False
    )

    masked = make_rollout(task, advantages=[0.0, 0.5])
    masked.samples[0].mask = [False, True]
    positive_gate = OnlineDifficultyFiltering(reject_min=0.5, reject_max=0.5)
    assert positive_gate.admit(CurriculumResult.from_rollouts([masked])) is False
