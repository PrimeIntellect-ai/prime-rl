from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import CurriculumConfig
from prime_rl.orchestrator.curricula import AdvantageRangeGate, DifficultyPools
from prime_rl.orchestrator.curriculum import Curriculum, CurriculumResult
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


class RejectingCurriculum(Curriculum):
    def __init__(self, tasks, *, reason: str):
        super().__init__(tasks)
        self.reason = reason
        self.seen = 0

    def on_result(self, result: CurriculumResult) -> bool:
        self.seen += 1
        return False

    def state_dict(self) -> dict:
        return super().state_dict() | {"seen": self.seen}

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.seen = state_dict["seen"]

    def metrics(self) -> dict[str, float]:
        return {"seen": float(self.seen)}


def test_default_curriculum_resumes_finite_and_infinite_tasksets() -> None:
    tasks = [make_task(i) for i in range(5)]
    finite = Curriculum(tasks)
    for _ in range(3):
        finite.sample()
    state = finite.state_dict()
    expected = [finite.sample().key for _ in range(4)]

    restored = Curriculum(tasks)
    restored.load_state_dict(state)
    assert [restored.sample().key for _ in range(4)] == expected

    def task_stream() -> Iterator[vf.Task]:
        yield from (make_task(i) for i in range(10))

    infinite = Curriculum(task_stream())
    infinite.sample()
    infinite.sample()
    restored_infinite = Curriculum(task_stream())
    restored_infinite.load_state_dict(infinite.state_dict())
    assert restored_infinite.sample().key == make_task(2).key


def test_curriculum_requires_unique_finite_task_keys() -> None:
    task = make_task(0)
    with pytest.raises(ValueError, match="Task keys must be unique"):
        Curriculum([task, task])


def test_train_source_hosts_user_curriculum_admission_state_and_metrics() -> None:
    tasks = [make_task(i) for i in range(3)]
    config = SimpleNamespace(
        ratio=1.0,
        curriculum=CurriculumConfig(
            import_path=f"{__name__}.RejectingCurriculum",
            kwargs={"reason": "test"},
        ),
    )
    env = SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)
    source = TrainSource([env])

    sampled = source.next_example()["task"]
    assert source.on_result([make_rollout(sampled)]) is False
    assert source.metrics() == {
        "curriculum/test/admission_rate": 0.0,
        "curriculum/test/seen": 1.0,
    }

    state = source.state_dict()
    restored = TrainSource([SimpleNamespace(name="test", tasks=iter(tasks), num_tasks=len(tasks), config=config)])
    restored.load_state_dict(state)
    assert restored.curricula["test"].state_dict()["seen"] == 1


def test_difficulty_pools_track_group_reward_and_resume_sampling() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = DifficultyPools(tasks, seed=7)
    rewards = {0: 0.1, 1: 0.5, 2: 0.9}
    for _ in tasks:
        task = pools.sample()
        pools.on_result(CurriculumResult.from_rollouts([make_rollout(task, reward=rewards[task.data.idx])]))

    assert pools.metrics() == {
        "pool/unseen": 0.0,
        "pool/hard": 1.0,
        "pool/medium": 1.0,
        "pool/easy": 1.0,
    }
    state = pools.state_dict()
    expected = [pools.sample().key for _ in range(10)]
    restored = DifficultyPools(tasks, seed=7)
    restored.load_state_dict(state)
    assert [restored.sample().key for _ in range(10)] == expected


def test_advantage_range_gate_generalizes_zero_advantage_rejection() -> None:
    task = make_task(0)
    zero_gate = AdvantageRangeGate([task])
    assert zero_gate.on_result(CurriculumResult.from_rollouts([make_rollout(task, advantages=[0.0, 0.0])])) is False
    assert zero_gate.on_result(CurriculumResult.from_rollouts([make_rollout(task, advantages=[0.0, 0.2])])) is True
    assert zero_gate.on_result(CurriculumResult.from_rollouts([make_rollout(task)])) is True

    tolerance_gate = AdvantageRangeGate([task], reject_min=-0.1, reject_max=0.1)
    assert (
        tolerance_gate.on_result(CurriculumResult.from_rollouts([make_rollout(task, advantages=[-0.05, 0.0, 0.05])]))
        is False
    )

    masked = make_rollout(task, advantages=[0.0, 0.5])
    masked.samples[0].mask = [False, True]
    positive_gate = AdvantageRangeGate([task], reject_min=0.5, reject_max=0.5)
    assert positive_gate.on_result(CurriculumResult.from_rollouts([masked])) is False
