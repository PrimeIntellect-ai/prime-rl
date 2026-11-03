"""A resumed run picks the train work up where the checkpoint left it: open groups with
their in-flight rollouts' restore points, the sink's buffers, the env mix."""

import asyncio
import dataclasses
import shutil
import time
import uuid
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import CurriculumConfig, OrchestratorConfig
from prime_rl.env import EpisodeAssembly, RestoreRequest
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.dispatcher import Dispatcher, _interrupted, inflight_restore
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
        self.on_interrupt = lambda: (0, set())
        self.on_save = lambda: (0, set())

    async def interrupt(self, timeout: float) -> tuple[int, set[str]]:
        return self.on_interrupt()

    async def save(self, timeout: float) -> tuple[int, set[str]]:
        return self.on_save()

    async def discard_checkpoints(self, snapshots: list[dict]) -> list[str]:
        ids = [snapshot["id"] for snapshot in snapshots]
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
    d.held_snapshots = {}
    d.sweep = None
    d.sweep_again = False
    d.eval_envs = None
    d.interrupting = False
    d.held = set()
    return d


def test_open_work_holds_each_open_groups_inflight_and_unscheduled_members():
    d = dispatcher(["a", "b"])
    a, b = group("a", to_schedule=1, emitted=1), group("b", to_schedule=0, emitted=3)
    # A fresh group has no id of its own; the dispatcher keys it.
    a_id, a.group_id = a.group_id, None
    d.groups = {a_id: a, b.group_id: b}
    d.inflight = {
        object(): inflight("a", a_id, streamed(["t1"])),
        object(): inflight("a", a_id, streamed(["t2"], restore=False)),
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
    resumed.resume_open_work(list(work.values()), {}, step=3)
    assert list(resumed.groups) == [a_id]


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
        a.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "held", "snapshot": {"id": "held"}}})]
        d.inflight = {object(): inflight("a", a.group_id, streamed(["t1"]))}  # newest: ck-t1
        d.snapshots = {"held": ("a", {"id": "held"}), "ck-t1": ("a", {"id": "ck-t1"}), "stale": ("a", {"id": "stale"})}
        d.hold_snapshots(2, [a])
        await d.sweep
        assert d.train_envs.get("a").deleted == ["stale"]
        assert set(d.snapshots) == {"held", "ck-t1"}
        # The checkpoint that resumes from it is gone: the held snapshot goes.
        d.release_snapshots([2])
        await d.sweep
        assert set(d.snapshots) == {"ck-t1"}

    asyncio.run(run())


def wire(episode: vf.Episode) -> vf.WireEpisode:
    """The episode as the env client returns it: a parametrized generic pickle cannot find."""
    from prime_rl.env.delta import dump

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
    manager.save_state(build_state(Progress(step=6), TrainSource([train_env("a")]), {}, work), step=6)

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
    orchestrator.eval_source = orchestrator.eval_sink = None
    orchestrator.eval_triggered_steps = set()
    orchestrator.train_sink = sink(["a"], group_size=2)
    orchestrator.dispatcher = SimpleNamespace(
        open_work=lambda: [], queued_results=lambda: [member("a", "gq")], snapshots={}
    )
    work = orchestrator.open_work()
    assert work["sink"]["group_targets"] == {"gq": 2}


def test_each_kept_checkpoint_holds_its_snapshots(tmp_path):
    """Every checkpoint still on disk resumes from its own snapshots; once the trainer
    deletes a checkpoint's step directory, the next save frees them."""
    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import CheckpointManager
    from prime_rl.orchestrator.orchestrator import Orchestrator

    async def run():
        o = Orchestrator.__new__(Orchestrator)
        o.ckpt_manager = CheckpointManager(tmp_path, CheckpointConfig())
        o.dispatcher = d = dispatcher(["a"])
        old, new = group("a", to_schedule=0, emitted=3), group("a", to_schedule=0, emitted=3)
        old.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "old", "snapshot": {"id": "old"}}})]
        new.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "new", "snapshot": {"id": "new"}}})]
        d.snapshots = {"old": ("a", {"id": "old"}), "new": ("a", {"id": "new"})}

        async def save_forked(state: dict, step: int) -> None:
            o.ckpt_manager.get_ckpt_path(step).mkdir(parents=True)

        o.ckpt_manager.save_forked = save_forked
        await o.save_state({"open_work": {"groups": [old]}}, step=2)
        await o.save_state({"open_work": {"groups": [new]}}, step=4)
        await d.sweep
        assert set(d.snapshots) == {"old", "new"}
        shutil.rmtree(o.ckpt_manager.get_ckpt_path(2).parent)
        await o.save_state({"open_work": {"groups": [new]}}, step=6)
        await d.sweep
        assert set(d.snapshots) == {"new"} and set(d.held_snapshots) == {4, 6}

    asyncio.run(run())


def test_a_fork_leaves_the_parents_snapshots_to_it():
    async def run():
        d = dispatcher(["a"])
        a = group("a", to_schedule=0, emitted=3)
        a.restores = [RestoreRequest(trace={}, point={"checkpoint": {"id": "ck", "snapshot": {"id": "ck"}}})]
        d.resume_open_work([a], {"ck": ("a", {"id": "ck"})}, step=2, owned=False)
        assert d.snapshots == {} and d.held_snapshots == {}
        assert list(d.groups) == [a.group_id]

    asyncio.run(run())


def test_a_checkpoint_holds_open_work_only_when_it_saves_rollouts():
    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.orchestrator import Orchestrator

    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.eval_source = orchestrator.eval_sink = None
    orchestrator.eval_triggered_steps = set()
    orchestrator.progress = Progress(step=6)
    orchestrator.train_source = TrainSource([train_env("a")])
    orchestrator.config_state = lambda: {}
    orchestrator.train_sink = sink(["a"], group_size=2)
    orchestrator.dispatcher = SimpleNamespace(open_work=lambda: [], queued_results=lambda: [], snapshots={})
    for rollouts in (False, True):
        orchestrator.config = SimpleNamespace(ckpt=CheckpointConfig(interval=5, rollouts=rollouts))
        assert ("open_work" in orchestrator.checkpoint_state()) is rollouts


def test_a_forked_checkpoint_links_the_rollouts_it_still_holds(tmp_path):
    """The forked writer saves the run as of the fork, and the next checkpoint links
    the in-queue rollouts the last one wrote."""
    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import ROLLOUTS_DIR, CheckpointManager, build_state

    s = sink(["a"])
    queued = wire(member("a", "gb"))
    manager = CheckpointManager(tmp_path, CheckpointConfig())

    def save(step: int) -> None:
        work = {"groups": [], "sink": s.state_dict(), "results": [queued], "snapshots": {}}
        asyncio.run(
            manager.save_forked(build_state(Progress(step=step), TrainSource([train_env("a")]), {}, work), step)
        )

    save(6)
    asyncio.run(s.add(wire(member("a", "ga"))))
    save(7)

    first, second = (manager.get_ckpt_path(step) / ROLLOUTS_DIR for step in (6, 7))
    name = f"{queued.id}.queued"
    assert (first / name).stat().st_ino == (second / name).stat().st_ino
    assert manager.last_rollouts == second
    loaded = manager.load(Progress(), TrainSource([train_env("a")]), step=7)
    assert loaded["results"][0].id == queued.id
    (pending,) = loaded["sink"]["pending_groups"]["ga"]
    assert isinstance(pending, vf.WireEpisode)


def test_a_failed_checkpoint_writer_raises(tmp_path):
    import pytest

    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import CheckpointManager

    manager = CheckpointManager(tmp_path, CheckpointConfig())

    def fail(state, step):
        raise OSError("disk full")

    manager.save_state = fail
    with pytest.raises(RuntimeError, match="exited with 1"):
        asyncio.run(manager.save_forked({}, 6))


def test_a_rollout_whose_arrays_were_persisted_since_is_written_again(tmp_path):
    """A rollout file names whether its arrays were persisted: the one of a checkpoint
    that could not persist them still points at the env server, and must not be
    linked by one that did."""
    from prime_rl.orchestrator.ckpt import PERSISTED_URL, write_rollouts
    from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY

    queued = wire(member("a", "gb"))
    queued.traces[0].info[PAYLOAD_INFO_KEY] = {"url": "http://worker", "branches": {}}
    work = {"groups": [], "sink": sink(["a"]).state_dict(), "results": [queued], "snapshots": {}}

    first = write_rollouts(work, tmp_path / "6", None)["results"][0]
    queued.traces[0].info[PAYLOAD_INFO_KEY]["url"] = PERSISTED_URL
    second = write_rollouts(work, tmp_path / "7", tmp_path / "6")["results"][0]

    assert (first, second) == (f"{queued.id}.queued.unpersisted", f"{queued.id}.queued")
    assert (tmp_path / "7" / second).stat().st_ino != (tmp_path / "6" / first).stat().st_ino


def eval_member(env: str, step: int = 4) -> vf.Episode:
    episode = member(env, "unused")
    episode.run = vf.TrainRunInfo(id="run", work=vf.EvalWorkInfo(step=step))
    return episode


def test_a_training_runs_eval_episode_is_eval_work():
    from prime_rl.orchestrator.utils import result_kind

    assert result_kind(wire(member("a", "g"))) == "train"
    assert result_kind(wire(eval_member("e"))) == "eval"


def test_an_eval_env_owns_the_snapshots_it_keeps():
    from prime_rl.configs.orchestrator import EvalSourceConfig
    from prime_rl.orchestrator.envs import EvalEnv

    config = EvalSourceConfig.model_validate(
        {"env": {"taskset": {"id": "reverse-text"}, "agent": {"checkpoint": {"after": "keep"}}}}
    )
    assert EvalEnv(config, None, []).owns_snapshots


def test_eval_epochs_survive_the_checkpoint(tmp_path):
    """A resumed run gets its eval epochs back: the open eval groups, the examples not yet
    dispatched, the rollouts already in, and the steps whose evals fired."""
    from prime_rl.configs.orchestrator import CheckpointConfig
    from prime_rl.orchestrator.ckpt import ROLLOUTS_DIR, CheckpointManager, build_state
    from prime_rl.orchestrator.dispatcher import DispatcherMode
    from prime_rl.orchestrator.eval_sink import EvalSink
    from prime_rl.orchestrator.eval_source import EvalSource
    from prime_rl.orchestrator.orchestrator import Orchestrator
    from prime_rl.orchestrator.types import TaskRequest
    from prime_rl.transports.batch.types import PAYLOAD_INFO_KEY

    def orchestrator() -> Orchestrator:
        o = Orchestrator.__new__(Orchestrator)
        eval_env = SimpleNamespace(
            name="e", examples=[make_task(0), make_task(1)], config=SimpleNamespace(group_size=2)
        )
        o.eval_envs = SimpleNamespace(names=["e"], get=lambda name: eval_env)
        o.train_envs = TrainEnvs(["a"])
        o.train_sink = sink(["a"])
        o.train_source = TrainSource([train_env("a")])
        o.eval_source = EvalSource([eval_env], intervals={"e": 1}, is_resumed=True)
        o.eval_sink = EvalSink(eval_envs=o.eval_envs)
        o.eval_triggered_steps, o.eval_triggered_at = set(), {}
        o.dispatcher = dispatcher(["a"])
        o.dispatcher.eval_envs, o.dispatcher.eval_source = o.eval_envs, o.eval_source
        o.dispatcher.mode = DispatcherMode.PREFER_TRAIN
        o.dispatcher.out_q = asyncio.Queue()
        o.config = SimpleNamespace(resume=None)
        o.resume_step = 6
        return o

    saved = orchestrator()
    open_group = dataclasses.replace(group("e", to_schedule=1, emitted=1), kind="eval", step=4)
    saved.dispatcher.groups = {open_group.group_id: open_group}
    saved.eval_source.queue.append(TaskRequest(env_name="e", task=make_task(1), step=4))
    landed = wire(eval_member("e"))
    landed.traces[0].info[PAYLOAD_INFO_KEY] = {"url": "http://worker", "branches": {}}
    saved.eval_sink.pending_batches[("e", 4)].append(landed)
    saved.eval_triggered_steps.add(4)
    work = saved.open_work()
    manager = CheckpointManager(tmp_path, CheckpointConfig())
    manager.save_state(build_state(Progress(step=6), TrainSource([train_env("a")]), {}, work), step=6)

    (name,) = [f for f in (manager.get_ckpt_path(6) / ROLLOUTS_DIR).iterdir()]
    assert name.name == f"{landed.id}.eval"
    resumed = orchestrator()
    resumed.resume_open_work(manager.load(Progress(), TrainSource([train_env("a")]), step=6))
    (reopened,) = resumed.dispatcher.groups.values()
    assert (reopened.kind, reopened.env_name, reopened.step) == ("eval", "e", 4)
    assert [request.task.key for request in resumed.eval_source.queue] == [make_task(1).key]
    ((episode,),) = resumed.eval_sink.pending_batches.values()
    assert episode.id == landed.id and PAYLOAD_INFO_KEY not in episode.traces[0].info
    assert resumed.eval_triggered_steps == {4} and ("e", 4) in resumed.eval_triggered_at
    assert resumed.dispatcher.mode == DispatcherMode.PREFER_EVAL

    # The open work continues the checkpoint's train groups: the train source re-runs
    # none of them whole, and tracks them as open.
    saved = orchestrator()
    train_group = group("a", to_schedule=1, emitted=1)
    saved.dispatcher.groups = {train_group.group_id: train_group}
    source = TrainSource([train_env("a")])
    source.open_groups[str(train_group.group_id)] = ("a", train_group.task.key)
    manager.save_state(build_state(Progress(step=7), source, {}, saved.open_work()), step=7)
    resumed = orchestrator()
    resumed.resume_open_work(manager.load(Progress(), resumed.train_source, step=7))
    assert not resumed.train_source.reopened
    assert resumed.train_source.open_groups == {str(train_group.group_id): ("a", train_group.task.key)}


def test_an_interrupt_saves_the_restore_points_it_took():
    """A pause interrupts the in-flight rollouts and saves the restore points their
    interrupts took; an interrupted run that replies stays in flight for the save."""

    async def run():
        d = dispatcher(["a"])
        a = group("a", to_schedule=0, emitted=3)
        d.groups = {a.group_id: a}
        meta = inflight("a", a.group_id, streamed(["t1"], restore=False))
        task = asyncio.get_running_loop().create_future()
        d.inflight = {task: meta}

        def interrupt() -> int:
            # The rollout checkpoints its box and streams the restore point; its run
            # then replies.
            meta.assembly.apply({"trace": "t1", "restore": {"checkpoint": {"id": "ck", "captured": time.time() + 1}}})
            task.set_result(
                SimpleNamespace(traces=[SimpleNamespace(last_error=SimpleNamespace(type="RolloutInterrupted"))])
            )
            return 1, {"t1"}

        d.train_envs.get("a").on_interrupt = interrupt
        await d.interrupt_inflight(timeout=1)
        await d.handle_completed_request(task)
        assert d.inflight == {task: meta} and d.held == {task}
        (saved,) = d.open_work()
        assert [r.point["checkpoint"]["id"] for r in saved.restores] == ["ck"]

    asyncio.run(run())


def test_a_save_keeps_the_rollouts_running_with_their_restore_points():
    """An interval checkpoint saves the in-flight rollouts: it waits for the restore
    point each one streams, and the rollouts stay in flight and keep dispatching."""

    async def run():
        d = dispatcher(["a"])
        a = group("a", to_schedule=0, emitted=3)
        d.groups = {a.group_id: a}
        meta = inflight("a", a.group_id, streamed(["t1"], restore=False))
        task = asyncio.get_running_loop().create_future()
        d.inflight = {task: meta}

        def save() -> tuple[int, set[str]]:
            # The checkpoint becomes restorable after the save returns.
            async def stream() -> None:
                await asyncio.sleep(0.1)
                meta.assembly.apply({"trace": "t1", "restore": {"checkpoint": {"id": "ck", "captured": time.time()}}})

            asyncio.get_running_loop().create_task(stream())
            return 1, {"t1"}

        d.train_envs.get("a").on_save = save
        await d.save_inflight(timeout=1)
        assert not task.done() and not d.interrupting and not d.held
        (saved,) = d.open_work()
        assert [r.point["checkpoint"]["id"] for r in saved.restores] == ["ck"]

    asyncio.run(run())


def test_only_interrupted_runs_stay_in_flight_during_an_interrupt():
    """A rollout that finished on its own while the pause interrupted the others is an
    in-queue result; an interrupted, failed or cancelled run stays in flight."""

    async def run():
        loop = asyncio.get_running_loop()

        def done(error_type: str | None) -> asyncio.Future:
            task = loop.create_future()
            error = SimpleNamespace(type=error_type) if error_type else None
            task.set_result(SimpleNamespace(traces=[SimpleNamespace(last_error=error)]))
            return task

        failed = loop.create_future()
        failed.set_exception(RuntimeError("env server gone"))
        cancelled = loop.create_future()
        cancelled.cancel()
        assert not _interrupted(done(None))
        assert not _interrupted(done("ProviderError"))
        assert _interrupted(done("RolloutInterrupted"))
        assert _interrupted(failed) and _interrupted(cancelled)

    asyncio.run(run())
