import itertools
import random
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


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def reload_tasks(indices: list[int]) -> list[vf.Task]:
    class Taskset(vf.Taskset[vf.Task]):
        def load(self):
            return [make_task(i) for i in indices]

    return list(Taskset(vf.TasksetConfig()))


def make_rollout(
    task: vf.Task,
    *,
    env_name: str = "test",
    reward: float = 0.0,
    advantages: list[float] | None = None,
) -> list[vf.Episode]:
    nodes = (
        [
            vf.MessageNode(
                message=vf.AssistantMessage(content="answer"),
                token_ids=list(range(len(advantages))),
                mask=[True] * len(advantages),
                logprobs=[0.0] * len(advantages),
                advantages=advantages,
                sampled=True,
            )
        ]
        if advantages is not None
        else [vf.MessageNode(message=vf.AssistantMessage(content="answer"), sampled=True)]
    )
    trace = vf.Trace(
        task=vf.TraceTask(
            type=type(task).__name__,
            data=task.data,
            key=task.key,
            hash=task.hash,
        ),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=nodes,
        rewards={"reward": vf.Reward(score=reward)},
        ok=True,
    )
    episode = vf.Episode(
        env=vf.EnvInfo(id=env_name, name=env_name),
        task=trace.task,
        group=vf.GroupInfo(id=str(uuid.uuid4())),
        traces=[trace],
        ok=True,
    )
    return [episode]


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
        group_size=1,
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

    sampled = source.next_task(step=1).task
    empty = make_rollout(sampled, reward=1.0)
    empty[0].traces[0].nodes[0].sampled = False
    group = make_rollout(sampled, reward=0.25, advantages=[0.0])
    failed = make_rollout(sampled, reward=1.0, advantages=[1.0])
    failed[0].ok = False
    assert source.on_result(group + failed + empty) is False
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

    assert source.on_result(empty) is False
    assert source.metrics()["curriculum/test/admission_rate"] == 0.0
    assert source.curricula["test"].sampler.task_rewards == {sampled.key: 0.25}

    # A scored rollout without token metadata still informs task difficulty.
    assert source.on_result(make_rollout(sampled, reward=0.5)) is False
    assert source.metrics()["curriculum/test/admission_rate"] == 0.0
    assert source.curricula["test"].sampler.task_rewards == {sampled.key: 0.5}

    assert source.on_result(make_rollout(sampled, reward=0.75, advantages=[1.0])) is True
    assert source.metrics()["curriculum/test/admission_rate"] == 1.0
    assert source.curricula["test"].sampler.task_rewards == {sampled.key: 0.75}


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
        group = make_rollout(task, reward=rewards[task.data.idx], advantages=[float(index > 0)])
        decisions.append(curriculum.on_result(group))

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
    assert zero_gate.admit(make_rollout(task, advantages=[0.0, 0.0])) is False
    assert zero_gate.admit(make_rollout(task, advantages=[0.0, 0.2])) is True
    assert zero_gate.admit(make_rollout(task)) is True

    tolerance_gate = AdvRangeGate(AdvRangeGateConfig(reject_min=-0.1, reject_max=0.1))
    assert tolerance_gate.admit(make_rollout(task, advantages=[-0.05, 0.0, 0.05])) is False

    positive_gate = AdvRangeGate(AdvRangeGateConfig(reject_min=0.5, reject_max=0.5))
    assert positive_gate.admit(make_rollout(task, advantages=[0.5])) is False


def keys(tasks: list[vf.Task]) -> list[str]:
    return [task.key for task in tasks]


def sample_keys(sampler: StandardSampler, count: int) -> list[str]:
    return [next(sampler).key for _ in range(count)]


def test_standard_sampler_matches_cycle_order() -> None:
    tasks = [make_task(i) for i in range(5)]
    sampler = StandardSampler(tasks)
    assert sample_keys(sampler, 23) == keys(list(itertools.islice(itertools.cycle(tasks), 23)))
    assert sampler.state_dict() == {"counts": {task.key: 5 if i < 3 else 4 for i, task in enumerate(tasks)}}


def test_standard_sampler_new_tasks_join_the_current_pass() -> None:
    tasks = reload_tasks(list(range(5)))
    sampler = StandardSampler(tasks)
    sample_keys(sampler, 7)
    reloaded = reload_tasks([10, *range(5), 11])
    first, last = reloaded[0], reloaded[-1]
    restored = StandardSampler(reloaded)
    restored.load_state_dict(sampler.state_dict())

    current_pass = [first, *tasks[2:], last]
    assert sample_keys(restored, 12) == keys(current_pass + [first, *tasks, last])


def test_standard_sampler_drops_removed_tasks() -> None:
    tasks = reload_tasks(list(range(5)))
    sampler = StandardSampler(tasks)
    sample_keys(sampler, 7)
    kept = reload_tasks([1, 3, 4])
    restored = StandardSampler(kept)
    restored.load_state_dict(sampler.state_dict())

    assert restored.state_dict() == {"counts": {tasks[1].key: 2, tasks[3].key: 1, tasks[4].key: 1}}
    assert sample_keys(restored, 5) == keys([tasks[3], tasks[4], tasks[1], tasks[3], tasks[4]])


def test_standard_sampler_restarts_counts_when_every_task_is_new() -> None:
    sampler = StandardSampler([make_task(i) for i in range(3)])
    sample_keys(sampler, 4)
    tasks = [make_task(i) for i in range(10, 12)]
    restored = StandardSampler(tasks)
    restored.load_state_dict(sampler.state_dict())

    assert restored.state_dict() == {"counts": dict.fromkeys(keys(tasks), 0)}


def test_standard_sampler_converts_a_legacy_cursor() -> None:
    tasks = [make_task(i) for i in range(5)]
    restored = StandardSampler(tasks)
    restored.load_state_dict({"cursor": 7})

    assert restored.state_dict() == {"counts": {task.key: 2 if i < 2 else 1 for i, task in enumerate(tasks)}}
    assert sample_keys(restored, 6) == keys(list(itertools.islice(itertools.cycle(tasks), 7, 13)))


def test_curriculum_resumes_a_legacy_checkpoint() -> None:
    tasks = [make_task(i) for i in range(3)]
    curriculum = Curriculum(None, tasks)
    curriculum.load_state_dict({"sampler": {"cursor": 4}, "gates": {}})

    assert next(curriculum.sampler).key == tasks[1].key


def test_curriculum_starts_a_changed_sampler_type_fresh() -> None:
    tasks = [make_task(i) for i in range(3)]
    pools = {"all": DifficultyPoolConfig(threshold=1.0, weight=1.0)}
    pool_config = CurriculumConfig(sampler=DifficultyPoolSamplerConfig(pools=pools))
    standard = Curriculum(None, tasks)
    next(standard.sampler)

    pool = Curriculum(pool_config, tasks)
    pool.load_state_dict(standard.state_dict())
    assert pool.state_dict() == Curriculum(pool_config, tasks).state_dict()

    pool.on_result(make_rollout(tasks[0], reward=0.5))
    restored = Curriculum(None, tasks)
    restored.load_state_dict(pool.state_dict())
    assert restored.state_dict() == Curriculum(None, tasks).state_dict()
    # A checkpoint written before the sampler type was recorded.
    legacy = {key: value for key, value in pool.state_dict().items() if key != "sampler_type"}
    restored.load_state_dict(legacy)
    assert restored.state_dict() == Curriculum(None, tasks).state_dict()


def test_curriculum_matches_gates_by_name() -> None:
    tasks = [make_task(0)]
    loaded: list[str] = []

    class RecordingGate(AdvRangeGate):
        def __init__(self, name: str) -> None:
            super().__init__(AdvRangeGateConfig())
            self.name = name

        def load_state_dict(self, state_dict: dict) -> None:
            loaded.append(self.name)

    saved = Curriculum(CurriculumConfig(gates={"kept": AdvRangeGateConfig(), "removed": AdvRangeGateConfig()}), tasks)
    curriculum = Curriculum(
        CurriculumConfig(gates={"kept": AdvRangeGateConfig(), "added": AdvRangeGateConfig()}), tasks
    )
    curriculum.gates = {name: RecordingGate(name) for name in curriculum.gates}
    curriculum.load_state_dict(saved.state_dict())

    assert loaded == ["kept"]


def test_difficulty_pool_sampler_drops_rewards_of_removed_tasks() -> None:
    tasks = reload_tasks(list(range(3)))
    config = CurriculumConfig(sampler=DifficultyPoolSamplerConfig())
    curriculum = Curriculum(config, tasks)
    for task in tasks:
        curriculum.on_result(make_rollout(task, reward=0.5))
    reloaded = reload_tasks([2, 0, 10])
    restored = Curriculum(config, reloaded)
    restored.load_state_dict(curriculum.state_dict())

    assert restored.sampler.task_rewards == {tasks[0].key: 0.5, tasks[2].key: 0.5}
    assert restored.metrics()["sampler/pool/unseen"] == 1.0


def make_env(name: str, tasks: list[vf.Task]) -> SimpleNamespace:
    config = SimpleNamespace(ratio=1.0, group_size=1, curriculum=None)
    return SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=len(tasks), config=config)


def test_train_source_records_group_size_at_draw_time() -> None:
    env = make_env("a", reload_tasks(list(range(4))))
    env.config.group_size = 2
    source = TrainSource([env])
    old = source.next_task(step=1)
    env.config = SimpleNamespace(ratio=1.0, group_size=4, curriculum=None)
    new = source.next_task(step=2)

    assert (old.rollouts, new.rollouts) == (2, 4)
    assert (source.group_size(old.group_id), source.group_size(new.group_id)) == (2, 4)
    assert source.state_dict()["open_groups"] == [("a", old.task.key, 2), ("a", new.task.key, 4)]
    source.close_groups([old.group_id])
    assert source.group_size(old.group_id) is None


def test_train_source_matches_envs_by_name() -> None:
    tasks = {name: [make_task(i) for i in range(offset, offset + 3)] for name, offset in [("a", 0), ("b", 10)]}
    source = TrainSource([make_env(name, env_tasks) for name, env_tasks in tasks.items()])
    for name in tasks:
        next(source.curricula[name].sampler)
    source.rng.random()  # the restored rng must differ from a fresh one

    added = [make_task(i) for i in range(20, 23)]
    restored = TrainSource([make_env("b", tasks["b"]), make_env("c", added)])
    restored.load_state_dict(source.state_dict())

    assert set(restored.curricula) == {"b", "c"}
    assert next(restored.curricula["b"].sampler).key == tasks["b"][1].key
    assert next(restored.curricula["c"].sampler).key == added[0].key
    assert restored.rng.getstate() == source.rng.getstate() != random.Random(42).getstate()


def test_train_source_reopens_open_groups_in_order_without_recounting() -> None:
    tasks = reload_tasks(list(range(4)))
    env = make_env("a", tasks)
    source = TrainSource([env])
    requests = []
    for size in (2, 4, 8):
        env.config.group_size = size
        requests.append(source.next_task(step=1))
    source.close_groups([requests[1].group_id])
    state = source.state_dict()
    assert state["open_groups"] == [("a", requests[0].task.key, 2), ("a", requests[2].task.key, 8)]

    resume_env = make_env("a", reload_tasks([3, 2, 0, 1]))
    resume_env.config.group_size = 16
    restored = TrainSource([resume_env])
    restored.load_state_dict(state)
    counts, rng = dict(restored.curricula["a"].sampler.counts), restored.rng.getstate()
    first = restored.next_task(step=2)
    assert first.task.key == requests[0].task.key
    assert first.rollouts == restored.group_size(first.group_id) == 2
    assert first.group_id not in {request.group_id for request in requests}
    # A checkpoint between re-runs keeps the waiting groups after the re-run ones.
    assert restored.state_dict()["open_groups"] == state["open_groups"]
    resumed_again = TrainSource([make_env("a", tasks)])
    resumed_again.load_state_dict(restored.state_dict())
    assert [resumed_again.next_task(step=3).rollouts for _ in range(2)] == [2, 8]
    second = restored.next_task(step=2)
    assert second.task.key == requests[2].task.key
    assert second.rollouts == restored.group_size(second.group_id) == 8
    assert restored.curricula["a"].sampler.counts == counts and restored.rng.getstate() == rng
    assert len(restored.open_groups) == 2 and next(iter(restored.open_groups)) == first.group_id
    fresh = restored.next_task(step=2)
    assert fresh.task.key == tasks[3].key
    assert fresh.rollouts == 16


def test_train_source_drops_open_groups_it_cannot_reopen() -> None:
    tasks = [make_task(i) for i in range(3)]
    state = {
        "rng": TrainSource([make_env("a", tasks)]).rng.getstate(),
        "envs": {},
        "open_groups": [("a", tasks[0].key), ("removed", tasks[0].key), ("a", "gone"), ("stream", tasks[1].key)],
    }
    stream = SimpleNamespace(
        name="stream", tasks=iter(tasks), num_tasks=None, config=SimpleNamespace(ratio=1.0, curriculum=None)
    )
    restored = TrainSource([make_env("a", tasks), stream])
    restored.load_state_dict(state)
    assert [(env_name, task.key, size) for env_name, task, size in restored.reopened] == [("a", tasks[0].key, 1)]
    assert restored.next_task(step=2).rollouts == 1


def test_train_source_loads_a_checkpoint_without_open_groups() -> None:
    tasks = [make_task(i) for i in range(3)]
    source = TrainSource([make_env("a", tasks)])
    source.next_task(step=1)
    state = source.state_dict()
    del state["open_groups"]

    restored = TrainSource([make_env("a", tasks)])
    restored.load_state_dict(state)
    assert not restored.reopened
    assert restored.next_task(step=2).task.key == tasks[1].key
