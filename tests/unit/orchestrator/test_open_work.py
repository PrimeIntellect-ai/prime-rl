"""A resumed run picks the train work up where the checkpoint left it: open groups with
their in-flight rollouts' restore points, the sink's buffers, the env mix."""

import asyncio
import uuid
from types import SimpleNamespace

import verifiers.v1 as vf
from verifiers.v1.serve import EpisodeAssembly, RestoreRequest

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import CurriculumConfig, OrchestratorConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.dispatcher import Dispatcher, inflight_restore
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import GroupState, InflightEpisode, Progress


def make_task(idx: int) -> vf.Task:
    return vf.Task(vf.TaskData(idx=idx, prompt=f"task {idx}"))


def train_env(name: str, num_tasks: int = 4) -> SimpleNamespace:
    tasks = [make_task(i) for i in range(num_tasks)]
    config = SimpleNamespace(ratio=1.0, curriculum=CurriculumConfig())
    return SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=num_tasks, config=config)


def test_train_source_resumes_a_changed_mix():
    source = TrainSource([train_env("a"), train_env("b")])
    for _ in range(5):
        source.next_task(step=1)
    state = source.state_dict()

    resumed = TrainSource([train_env("a"), train_env("c")])
    resumed.load_state_dict(state)
    assert resumed.curricula["a"].state_dict() == source.curricula["a"].state_dict()
    assert resumed.curricula["c"].state_dict() == TrainSource([train_env("c")]).curricula["c"].state_dict()
    assert "b" not in resumed.curricula


def streamed(trace_ids: list[str], restore: bool = True) -> EpisodeAssembly:
    assembly = EpisodeAssembly()
    for trace_id in trace_ids:
        assembly.apply({"trace": trace_id, "open": {"id": trace_id}})
        if restore:
            assembly.apply({"trace": trace_id, "restore": {"checkpoint": {"id": f"ck-{trace_id}"}}})
    return assembly


def inflight(env: str, group_id: uuid.UUID, assembly: EpisodeAssembly) -> InflightEpisode:
    return InflightEpisode(
        kind="train",
        env_name=env,
        group_id=group_id,
        task=make_task(0),
        policy_version=0,
        step=1,
        assembly=assembly,
    )


def test_an_inflight_rollout_continues_from_its_newest_restore_point():
    group_id = uuid.uuid4()
    restore = inflight_restore(inflight("a", group_id, streamed(["t1"])))
    assert isinstance(restore, RestoreRequest)
    assert restore.point == {"checkpoint": {"id": "ck-t1"}}
    assert "pending" not in restore.trace
    # Nothing to continue from yet, or several traces (a multi-agent episode).
    assert inflight_restore(inflight("a", group_id, streamed(["t1"], restore=False))) is None
    assert inflight_restore(inflight("a", group_id, streamed(["t1", "t2"]))) is None


def group(env: str, *, to_schedule: int, emitted: int) -> GroupState:
    return GroupState(
        kind="train",
        env_name=env,
        task=make_task(0),
        step=3,
        episodes_to_schedule=to_schedule,
        target_episodes=4,
        emitted=emitted,
        group_id=uuid.uuid4(),
    )


class SnapshotEnv:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def discard_checkpoints(self, ids: list[str]) -> list[str]:
        self.deleted += ids
        return ids


class TrainEnvs:
    def __init__(self, names: list[str]) -> None:
        self.envs = {name: SnapshotEnv() for name in names}
        self.names = names

    def get(self, name: str) -> SnapshotEnv:
        return self.envs[name]

    def __iter__(self):
        return iter(self.envs.values())


def dispatcher(names: list[str]) -> Dispatcher:
    d = Dispatcher.__new__(Dispatcher)
    d.groups = {}
    d.inflight = {}
    d.train_envs = TrainEnvs(names)
    d.snapshots = {}
    d.held_snapshots = set()
    d.sweep = None
    d.sweep_again = False
    return d


def test_open_work_holds_each_open_groups_inflight_and_unscheduled_members():
    d = dispatcher(["a", "b"])
    a, b = group("a", to_schedule=1, emitted=1), group("b", to_schedule=0, emitted=3)
    d.groups = {a.group_id: a, b.group_id: b}
    d.inflight = {
        object(): inflight("a", a.group_id, streamed(["t1"])),
        object(): inflight("a", a.group_id, streamed(["t2"], restore=False)),
        object(): inflight("b", b.group_id, streamed(["t3"])),
    }
    work = {g.env_name: g for g in d.open_work()}
    # Two in flight plus one never dispatched; only one has a restore point.
    assert work["a"].episodes_to_schedule == 3
    assert [r.point["checkpoint"]["id"] for r in work["a"].restores] == ["ck-t1"]
    assert work["b"].episodes_to_schedule == 1 and len(work["b"].restores) == 1
    assert work["a"].emitted + work["a"].episodes_to_schedule == work["a"].target_episodes
    # The dispatcher's own groups are untouched.
    assert a.episodes_to_schedule == 1 and a.restores == []

    resumed = dispatcher(["a", "c"])
    resumed.resume_open_work(list(work.values()), {})
    assert [g.env_name for g in resumed.groups.values()] == ["a"]


def sink(names: list[str], group_size: int = 2) -> TrainSink:
    config = OrchestratorConfig(batch_size=100, max_staleness=8)
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=group_size, sampling=SimpleNamespace(temperature=1.0)),
        requires_sampling_masks=False,
        requires_top_logprobs=False,
        algorithm=Algorithm(GRPOAlgoConfig(), None),
    )
    return TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env, names=names),
        progress=Progress(),
        batch_size=100,
        on_result=lambda group: True,
    )


def member(env: str, group_id: str) -> vf.Episode:
    task = make_task(0)
    trace = vf.Trace(
        task=vf.TraceTask(type="Task", data=task.data, key=task.key, hash=task.hash),
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        nodes=[vf.MessageNode(message=vf.AssistantMessage(content="a"), sampled=True)],
        ok=True,
    )
    return vf.Episode(
        env=vf.EnvInfo(id=env, name=env),
        task=trace.task,
        group=vf.GroupInfo(id=group_id),
        traces=[trace],
        ok=True,
        run=vf.TrainRunInfo(id="run", work=vf.TrainWorkInfo(step=1, policy=vf.PolicySpan(start=0, end=0))),
    )


def test_the_sink_resumes_its_partial_groups_of_kept_envs():
    s = sink(["a", "b"])
    asyncio.run(s.add(member("a", "ga")))
    asyncio.run(s.add(member("b", "gb")))
    state = s.state_dict()
    assert set(state["pending_groups"]) == {"ga", "gb"}

    resumed = sink(["a"])
    resumed.load_state_dict(state)
    assert set(resumed.pending_groups) == {"ga"}
    # The group completes with its last member after the resume.
    asyncio.run(resumed.add(member("a", "ga")))
    assert "ga" not in resumed.pending_groups


def test_a_sweep_keeps_the_newest_and_held_snapshots_only():
    async def run():
        d = dispatcher(["a"])
        a = group("a", to_schedule=0, emitted=3)
        a.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "held", "box": True}})]
        d.inflight = {object(): inflight("a", a.group_id, streamed(["t1"]))}  # newest: ck-t1
        d.snapshots = {"held": "a", "ck-t1": "a", "stale": "a"}
        d.hold_snapshots([a])
        await d.sweep
        assert d.train_envs.get("a").deleted == ["stale"]
        assert set(d.snapshots) == {"held", "ck-t1"}
        # A newer checkpoint no longer resumes from it: the old held snapshot goes.
        d.hold_snapshots([])
        await d.sweep
        assert set(d.snapshots) == {"ck-t1"}

    asyncio.run(run())


def wire(episode: vf.Episode) -> vf.WireEpisode:
    """The episode as the env client returns it: a parametrized generic pickle cannot find."""
    from verifiers.v1.serve.delta import dump

    return vf.WireEpisode.model_validate(dump(episode))


def test_open_work_survives_the_checkpoint_file(tmp_path):
    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import CheckpointManager, build_state

    d = dispatcher(["a"])
    a = group("a", to_schedule=1, emitted=1)
    d.groups = {a.group_id: a}
    d.inflight = {object(): inflight("a", a.group_id, streamed(["t1"]))}
    s = sink(["a"])
    asyncio.run(s.add(wire(member("a", "ga"))))
    work = {"groups": d.open_work(), "sink": s.state_dict(), "results": [wire(member("a", "gb"))], "snapshots": {}}
    manager = CheckpointManager(tmp_path, CheckpointConfig())
    manager.save_state(build_state(Progress(step=6), TrainSource([train_env("a")]), work), step=6)

    loaded = manager.load(Progress(), TrainSource([train_env("a")]), step=6)
    (resumed,) = loaded["groups"]
    assert resumed.task.key == a.task.key and resumed.episodes_to_schedule == 2
    # The rollout without persisted arrays (no env server here) starts over.
    assert resumed.restores == []
    (pending,) = loaded["sink"]["pending_groups"]["ga"]
    assert isinstance(pending, vf.WireEpisode) and pending.traces[0].nodes
    assert loaded["results"][0].group.id == "gb"


def test_a_resumed_group_finalizes_at_the_size_it_opened_with():
    s = sink(["a"], group_size=2)
    asyncio.run(s.add(member("a", "ga")))
    state = s.state_dict()
    assert state["group_targets"] == {"ga": 2}

    resumed = sink(["a"], group_size=3)
    resumed.load_state_dict(state)
    asyncio.run(resumed.add(member("a", "ga")))
    assert "ga" not in resumed.pending_groups
    # A group opened after the resume takes the new size.
    asyncio.run(resumed.add(member("a", "gb")))
    asyncio.run(resumed.add(member("a", "gb")))
    assert len(resumed.pending_groups["gb"]) == 2


def test_a_group_waiting_in_the_queue_keeps_its_size():
    from prime_rl.orchestrator.orchestrator import Orchestrator

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.train_sink = sink(["a"], group_size=2)
    orchestrator.dispatcher = SimpleNamespace(
        open_work=lambda: [], queued_results=lambda: [member("a", "gq")], snapshots={}
    )
    work = orchestrator.open_work()
    assert work["sink"]["group_targets"] == {"gq": 2}


def test_a_checkpoint_being_written_holds_its_snapshots_and_the_last_ones():
    async def run():
        d = dispatcher(["a"])
        last, new = group("a", to_schedule=0, emitted=3), group("a", to_schedule=0, emitted=3)
        last.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "last", "box": True}})]
        new.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "new", "box": True}})]
        d.snapshots = {"last": "a", "new": "a"}
        d.hold_snapshots([last])
        d.hold_snapshots([new], release=False)
        await d.sweep
        assert set(d.snapshots) == {"last", "new"}
        d.hold_snapshots([new])
        await d.sweep
        assert set(d.snapshots) == {"new"}

    asyncio.run(run())
