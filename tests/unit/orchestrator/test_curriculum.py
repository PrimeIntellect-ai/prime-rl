from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from prime_rl.configs.orchestrator import CurriculumConfig
from prime_rl.orchestrator.curriculum import Curriculum, CurriculumResult
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Rollout


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def make_rollout(task: vf.Task, *, env_name: str = "test") -> Rollout:
    return Rollout(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        env_name=env_name,
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
