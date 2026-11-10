"""A rollout checkpoint saves the train pipeline (the unshipped episodes, the open groups
and the episodes in flight), and a resumed run continues it."""

import asyncio
import shutil
import uuid
from collections import deque
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import CheckpointConfig, CurriculumConfig, OrchestratorConfig
from prime_rl.env.types import RestoreRef
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.orchestrator.ckpt import (
    ARRAYS_DIR,
    RESUMED_DIR,
    CheckpointManager,
    SnapshotLedger,
    build_state,
    count_open_work,
    owned_snapshots,
    persist_open_work,
    read_open_work,
    wait_for_writer,
    write_open_work,
    write_state,
)
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.orchestrator import Orchestrator
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import (
    DispatchFailure,
    GroupCancellation,
    GroupState,
    InflightEpisode,
    OpenEpisode,
    Policy,
    Progress,
    TrainBatch,
)
from prime_rl.transports import payload
from prime_rl.transports.payload import PayloadHook, delete_payloads, wait_for_deletes
from prime_rl.utils import pathing
from tests.unit.orchestrator.test_train_sink import episode, trace


@pytest.fixture(autouse=True)
def run_directory(tmp_path, monkeypatch):
    """The env workers of a test keep checkpoint files only under its `tmp_path`."""
    monkeypatch.setattr(pathing, "_save_root", str(tmp_path.resolve()))


# Samples hold numpy streams; encode them as lists to compare them byte for byte.
ENCODER = msgspec.msgpack.Encoder(enc_hook=lambda value: value.tolist())


def sink(*, batch_size=100, names=("test",), group_size=None) -> TrainSink:
    config = OrchestratorConfig(batch_size=batch_size, max_staleness=8)
    env = SimpleNamespace(
        config=SimpleNamespace(group_size=1),
        sampling_at=lambda step: vf.SamplingConfig(temperature=1.0),
        requires_sampling_masks=False,
        requires_sampler_logprobs=False,
        algorithm=Algorithm(GRPOAlgoConfig(), None),
    )
    return TrainSink(
        config,
        tokenizer=None,
        train_envs=SimpleNamespace(get=lambda name: env, names=list(names)),
        progress=Progress(),
        batch_size=batch_size,
        on_result=lambda group: True,
        group_size=group_size,
    )


def open_work(**fields) -> dict:
    """An open work with nothing in it but ``fields``."""
    empty = sink().state_dict()
    return {
        "sink": empty,
        "results": [],
        "batch": None,
        "groups": [],
        "episodes": [],
        "open_groups": {},
        "reopened": [],
        "max_inflight": 8,
    } | fields


def kept(hook: PayloadHook) -> vf.Trace:
    """A trace whose arrays an env worker kept."""
    t = trace()
    for node in t.nodes:
        node.routed_experts = np.ones((len(node.token_ids), 2, 2), dtype=np.uint8)
    hook.keep(episode([t]), vf.SamplingConfig())
    return t


def restore_ref(trace_id: str, directory="/ckpt") -> RestoreRef:
    return RestoreRef(trace_id=trace_id, path=f"{directory}/{trace_id}.restore", snapshot={"id": f"s-{trace_id}"})


def test_the_sink_state_round_trips_through_the_checkpoint_file(tmp_path):
    async def run():
        sizes = {"open": 2}
        s = sink(group_size=sizes.get)
        await s.add(episode([trace()], group="queued"))
        await s.add(episode([trace()], group="open"))
        assert list(s.pending_groups["open"]) and len(s.pending_batch) == 1

        write_open_work(open_work(sink=s.state_dict()), tmp_path)
        loaded = read_open_work(tmp_path, "file:///arrays")

        resumed = sink(group_size=sizes.get)
        resumed.load_state_dict(loaded["sink"])
        assert list(resumed.pending_batch) == list(s.pending_batch)
        for trace_id, samples in s.pending_batch.items():
            assert ENCODER.encode(resumed.pending_batch[trace_id]) == ENCODER.encode(samples)
            assert isinstance(resumed.episode_by_trace[trace_id], vf.WireEpisode)
            assert resumed.episode_by_trace[trace_id].id == s.episode_by_trace[trace_id].id
        assert [e.id for e in resumed.pending_groups["open"]] == [e.id for e in s.pending_groups["open"]]
        # The open group finalizes with its last episode after the resume.
        await resumed.add(episode([trace()], group="open"))
        assert "open" not in resumed.pending_groups
        assert len(resumed.pending_batch) == 3

    asyncio.run(run())


def test_the_sink_drops_the_state_of_envs_it_no_longer_trains_on():
    async def run():
        s = sink(group_size={"open": 2}.get)
        await s.add(episode([trace()], group="queued"))
        await s.add(episode([trace()], group="open"))
        resumed = sink(names=["other"])
        resumed.load_state_dict(s.state_dict())
        assert not resumed.pending_groups and not resumed.pending_batch and not resumed.episode_by_trace

    asyncio.run(run())


def test_open_work_round_trips_through_the_checkpoint_file(tmp_path):
    group_id = uuid.uuid4()
    task = vf.Task(vf.TaskData(idx=3, prompt="task 3"))
    failure = DispatchFailure(
        kind="train",
        env_name="test",
        group_id="gf",
        step=2,
        policy_version=1,
        task_type="Task",
        task_key=task.key,
        task_hash=task.hash,
        error=vf.Error(type="Boom", message="failed"),
    )
    cancellation = GroupCancellation(kind="train", env_name="test", group_id="gc", step=2, count=2, reason="overload")
    queued = episode([trace()], group="gq")
    work = open_work(
        results=[queued, failure, cancellation],
        groups=[
            GroupState(
                kind="train",
                env_name="test",
                task=task,
                step=2,
                episodes_to_schedule=1,
                target_episodes=4,
                emitted=1,
                policy_version_at_start=1,
                group_id=group_id,
            )
        ],
        episodes=[OpenEpisode(group_id, restore_ref("t1")), OpenEpisode(group_id)],
        open_groups={str(group_id): ("test", task.key, 4)},
        reopened=[("test", task.key)],
    )
    assert count_open_work(work) == {"in_queue": 1, "restorable": 1, "restart": 1}
    write_open_work(work, tmp_path)
    loaded = read_open_work(tmp_path, "file:///arrays")

    first, second, third = loaded["results"]
    assert isinstance(first, vf.WireEpisode) and first.id == queued.id and first.group.id == "gq"
    assert first.run.work.type == "train"
    assert second == failure and third == cancellation
    (group,) = loaded["groups"]
    assert group.group_id == group_id and group.episodes_to_schedule == 1 and group.emitted == 1
    assert (group.task.key, group.task.hash) == (task.key, task.hash)
    assert loaded["episodes"] == work["episodes"]
    assert loaded["open_groups"] == work["open_groups"] and loaded["reopened"] == work["reopened"]
    assert loaded["max_inflight"] == 8 and loaded["batch"] is None


def test_an_unshipped_batch_keeps_its_arrays_in_the_checkpoint(tmp_path):
    async def run():
        hook = PayloadHook()
        s = sink(batch_size=1)
        batch = await s.add(episode([kept(hook)], group="shipping"))
        assert isinstance(batch, TrainBatch)
        (sample,) = batch.samples
        worker_url = sample.payload.url
        assert worker_url == hook.server.url
        work = open_work(batch=batch)

        assert await persist_open_work(work, tmp_path) == (1, 1)
        arrays = tmp_path / ARRAYS_DIR
        assert (arrays / sample.payload.id).is_file()
        write_open_work(work, tmp_path)
        # The run goes on reading from the env worker.
        assert sample.payload.url == worker_url
        assert batch.cohort.episodes[0].traces[0].info[payload.PAYLOAD_INFO_KEY]["url"] == worker_url

        loaded = read_open_work(tmp_path, arrays.as_uri())["batch"]
        (resumed,) = loaded.samples
        assert resumed.payload.url == arrays.as_uri() and resumed.payload.id == sample.payload.id
        assert ENCODER.encode(msgspec.structs.replace(resumed, payload=None)) == ENCODER.encode(
            msgspec.structs.replace(sample, payload=None)
        )
        assert loaded.cohort.sampled_trace_ids == batch.cohort.sampled_trace_ids
        assert [e.id for e in loaded.cohort.episodes] == [e.id for e in batch.cohort.episodes]
        assert loaded.cohort.episodes[0].traces[0].info[payload.PAYLOAD_INFO_KEY]["url"] == arrays.as_uri()

        # The trainer reads the file as it reads the env worker, and its DELETE leaves the file.
        url = f"{resumed.payload.url}/{resumed.payload.id}"
        fetched = payload._fetch(url)
        assert fetched is not None and fetched == payload._fetch(f"{worker_url}/{sample.payload.id}")
        delete_payloads([url])
        wait_for_deletes()
        assert (arrays / sample.payload.id).is_file()
        assert payload._fetch(f"{arrays.as_uri()}/missing") is None

        # The next checkpoint of the resumed run links the arrays from the files.
        assert await persist_open_work(open_work(batch=loaded), tmp_path / "next") == (1, 1)
        assert (tmp_path / "next" / ARRAYS_DIR / sample.payload.id).samefile(arrays / sample.payload.id)

    asyncio.run(run())


def test_arrays_a_worker_no_longer_holds_are_reported_missing(tmp_path):
    async def run():
        hook = PayloadHook()
        s = sink(batch_size=1)
        batch = await s.add(episode([kept(hook)], group="shipping"))
        hook.server.delete(batch.samples[0].payload.id)
        assert await persist_open_work(open_work(batch=batch), tmp_path) == (1, 0)

    asyncio.run(run())


def test_a_carried_restore_file_moves_to_the_new_checkpoint(tmp_path):
    async def run():
        old, new = tmp_path / "old", tmp_path / "new"
        old.mkdir()
        new.mkdir()
        (old / "t1.restore").write_bytes(b"point")
        group_id = uuid.uuid4()
        carried = OpenEpisode(group_id, restore_ref("t1", old))
        lost = OpenEpisode(group_id, restore_ref("t2", old))
        await persist_open_work(open_work(episodes=[carried, lost]), new)
        assert carried.restore.path == str(new / "t1.restore") and (new / "t1.restore").read_bytes() == b"point"
        # Its file is gone: the episode starts over.
        assert lost.restore is None

    asyncio.run(run())


class FakeEnv:
    uses_live_policy = False

    def __init__(self) -> None:
        self.runs: list[dict] = []

        async def finish_sessions(session_ids: list[str]) -> None:
            pass

        self.clients = SimpleNamespace(train_client=None, model_name="m", finish_sessions=finish_sessions)

    def sampling_at(self, step: int) -> vf.SamplingConfig:
        return vf.SamplingConfig()

    async def run(self, **kwargs) -> None:
        self.runs.append(kwargs)
        await asyncio.Event().wait()


def dispatcher(env: FakeEnv) -> Dispatcher:
    d = Dispatcher.__new__(Dispatcher)
    d.train_envs = SimpleNamespace(get=lambda name: env)
    d.eval_envs = None
    d.train_scheduling_disabled = False
    d.policy = Policy(version=3, model_name="m")
    d.policy_weights_change = True
    d.cache_salt_prefix = "salt"
    d.rate_limiter = None
    d.current_inflight = 0
    d.admissions_in_window = 0
    d.inflight = {}
    d.groups = {}
    d.resumed = deque()
    d.emitting = []
    d.live_events = []
    d.out_q = asyncio.Queue()
    return d


def group(*, to_schedule: int, emitted: int = 0, kind="train") -> GroupState:
    return GroupState(
        kind=kind,
        env_name="test",
        task=vf.Task(vf.TaskData(idx=0, prompt="task")),
        step=3,
        episodes_to_schedule=to_schedule,
        target_episodes=4,
        emitted=emitted,
        policy_version_at_start=2,
        group_id=uuid.uuid4(),
    )


def test_a_resumed_dispatcher_continues_the_inflight_episodes_first():
    async def run():
        env = FakeEnv()
        d = dispatcher(env)
        resumed, fresh = group(to_schedule=1, emitted=1), group(to_schedule=4)
        removed = uuid.uuid4()
        d.resume_open_work(
            [fresh, resumed],
            [
                OpenEpisode(resumed.group_id, restore_ref("t1")),
                OpenEpisode(removed, restore_ref("t9")),
                OpenEpisode(resumed.group_id),
            ],
        )
        # One never dispatched, two that were in flight.
        assert resumed.episodes_to_schedule == 3
        assert [e.group_id for e in d.resumed] == [resumed.group_id, resumed.group_id]

        for _ in range(3):
            assert await d.try_schedule("train")
        await asyncio.sleep(0)
        assert [run["restore"] for run in env.runs] == ["/ckpt/t1.restore", None, None]
        # Then the open groups continue in their order, as without a resume.
        assert [meta.group_id for meta in d.inflight.values()] == [resumed.group_id, resumed.group_id, fresh.group_id]
        assert [run["request_id"] for run in env.runs] == [meta.dispatch_id for meta in d.inflight.values()]
        assert not d.resumed and resumed.episodes_to_schedule == 1 and fresh.episodes_to_schedule == 3

        tasks = list(d.inflight)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(run())


def test_open_work_holds_the_open_groups_and_the_inflight_episodes():
    env = FakeEnv()
    d = dispatcher(env)
    a, b, evaluated = group(to_schedule=2), group(to_schedule=0, emitted=3), group(to_schedule=1, kind="eval")
    # A fresh group has no id of its own; the dispatcher keys it.
    a_id, a.group_id = a.group_id, None
    d.groups = {a_id: a, b.group_id: b, evaluated.group_id: evaluated}

    def inflight(group_id: uuid.UUID, kind="train") -> InflightEpisode:
        return InflightEpisode(kind=kind, env_name="test", group_id=group_id, task=a.task, policy_version=2, step=3)

    saved, restarted, last, eval_run = inflight(a_id), inflight(a_id), inflight(b.group_id), inflight(a_id, "eval")
    d.inflight = {object(): saved, object(): restarted, object(): last, object(): eval_run}
    # An episode this run resumed and has not dispatched yet: counted as still to schedule.
    waiting = OpenEpisode(a_id, restore_ref("t0"))
    d.resumed.append(waiting)
    restores = {
        saved.dispatch_id: restore_ref("t1"),
        last.dispatch_id: restore_ref("t3"),
        "finished-meanwhile": restore_ref("t4"),
        eval_run.dispatch_id: restore_ref("t5"),
    }

    groups, episodes = d.open_work(restores)
    by_id = {g.group_id: g for g in groups}
    assert set(by_id) == {a_id, b.group_id}
    assert by_id[a_id].episodes_to_schedule == 1 and by_id[b.group_id].episodes_to_schedule == 0
    assert [(e.group_id, e.restore and e.restore.trace_id) for e in episodes] == [
        (a_id, "t0"),
        (a_id, "t1"),
        (a_id, None),
        (b.group_id, "t3"),
    ]
    # What continues no train episode in flight is left for the caller to delete.
    assert {ref.trace_id for ref in restores.values()} == {"t4", "t5"}
    # The dispatcher's own groups are untouched.
    assert a.episodes_to_schedule == 2 and a.group_id is None

    # Every episode the group still owes is dispatched again after a resume.
    resumed = dispatcher(env)
    resumed.resume_open_work(groups, episodes)
    for g in resumed.groups.values():
        assert g.emitted + g.episodes_to_schedule == g.target_episodes


def test_results_on_their_way_into_a_full_queue_are_queued_results():
    async def run():
        d = dispatcher(FakeEnv())
        d.out_q = asyncio.Queue(maxsize=1)
        await d.emit("first")
        blocked = asyncio.create_task(d.emit("second"))
        await asyncio.sleep(0)
        assert d.queued_results() == ["first", "second"]
        assert d.out_q.get_nowait() == "first"
        await blocked
        assert d.queued_results() == ["second"] and not d.emitting

    asyncio.run(run())


def orchestrator() -> Orchestrator:
    o = Orchestrator.__new__(Orchestrator)
    o.stopped = asyncio.Event()
    o.draining = False
    o.component_tasks = []
    o.sink_lock = asyncio.Lock()
    o.sink_locked = False
    o.staged = deque()
    o.resumed_batch = None
    o.shipping = None
    o.resume_step = 4
    return o


def test_a_resumed_step_loop_ships_the_batch_then_the_saved_results_then_new_ones():
    async def run():
        o = orchestrator()
        out_q = asyncio.Queue()
        o.dispatcher = SimpleNamespace(out_q=out_q, get=out_q.get, is_idle=False)
        out_q.put_nowait("new")
        o.staged.extend(["saved 1", "saved 2"])
        o.resumed_batch = "batch"
        seen = []

        async def handle_train_result(batch) -> None:
            assert o.sink_lock.locked()
            seen.append(batch)

        async def ingest(item) -> None:
            assert o.sink_lock.locked()
            seen.append(item)
            if item == "new":
                o.stopped.set()

        o.handle_train_result = handle_train_result
        o.ingest = ingest
        await o.main_loop()
        assert seen == ["batch", "saved 1", "saved 2", "new"]
        assert not o.sink_lock.locked() and o.resumed_batch is None

    asyncio.run(run())


def test_the_orchestrator_saves_and_resumes_its_unshipped_work(tmp_path):
    async def run():
        s = sink(batch_size=1)
        batch = await s.add(episode([trace()], group="shipping"))
        staged, queued = episode([trace()], group="staged"), episode([trace()], group="queued")
        evaluated = episode([trace()], group="eval")
        evaluated.run = vf.TrainRunInfo(id="run", work=vf.EvalWorkInfo(step=1, policy=vf.PolicySpan(start=0, end=0)))
        failure = GroupCancellation(kind="eval", env_name="test", group_id="ge", step=1, count=1, reason="superseded")

        o = orchestrator()
        o.train_sink = s
        o.staged.append(staged)
        o.shipping = batch
        o.dispatcher = SimpleNamespace(
            open_work=lambda restores: ([], []), queued_results=lambda: [evaluated, queued, failure]
        )
        o.train_source = SimpleNamespace(open_groups={"shipping": ("test", "k", 1)}, reopened=[])
        o.concurrency = SimpleNamespace(max_inflight=16.0)
        work = o.open_work({})
        # Eval work is not checkpointed.
        assert [result.group.id for result in work["results"]] == ["staged", "queued"]
        assert work["batch"] is batch and work["max_inflight"] == 16
        write_open_work(work, tmp_path)

        resumed = orchestrator()
        resumed.train_sink = sink(batch_size=1)
        resumed.train_envs = SimpleNamespace(names=["test"])
        adopted, dispatched = [], []
        resumed.train_source = SimpleNamespace(adopt_open_groups=lambda *args: adopted.append(args))
        resumed.dispatcher = SimpleNamespace(resume_open_work=lambda *args: dispatched.append(args))
        resumed.resume_open_work(read_open_work(tmp_path, "file:///arrays"))
        assert [result.group.id for result in resumed.staged] == ["staged", "queued"]
        assert ENCODER.encode(resumed.resumed_batch.samples) == ENCODER.encode(batch.samples)
        assert [e.id for e in resumed.resumed_batch.cohort.episodes] == [e.id for e in batch.cohort.episodes]
        assert resumed.resumed_batch.cohort.sampled_trace_ids == batch.cohort.sampled_trace_ids
        assert adopted == [({"shipping": ("test", "k", 1)}, [])] and dispatched == [([], [])]
        # A checkpoint written before the step loop took the batch still holds it.
        resumed.dispatcher.open_work = lambda restores: ([], [])
        resumed.dispatcher.queued_results = lambda: []
        resumed.train_source = o.train_source
        resumed.concurrency = o.concurrency
        assert resumed.open_work({})["batch"] is resumed.resumed_batch

    asyncio.run(run())


def train_env(name: str, num_tasks: int = 4) -> SimpleNamespace:
    tasks = [vf.Task(vf.TaskData(idx=i, prompt=f"task {i}")) for i in range(num_tasks)]
    config = SimpleNamespace(ratio=1.0, curriculum=CurriculumConfig(), group_size=2)
    return SimpleNamespace(name=name, tasks=iter(tasks), num_tasks=num_tasks, config=config)


def test_a_checkpoint_with_rollouts_continues_its_open_groups_instead_of_rerunning_them(tmp_path):
    source = TrainSource([train_env("a")])
    request = source.next_task(step=1)
    episodes = [OpenEpisode(uuid.UUID(request.group_id), restore_ref("t1", "/elsewhere"))]
    work = open_work(episodes=episodes, open_groups=dict(source.open_groups))

    def checkpoint(output_dir, rollouts: bool) -> CheckpointManager:
        manager = CheckpointManager(output_dir, CheckpointConfig(rollouts=rollouts))
        ckpt_path = manager.get_ckpt_path(6)
        (ckpt_path / "rollouts" / ARRAYS_DIR).mkdir(parents=True)
        (ckpt_path / "rollouts" / ARRAYS_DIR / "blob").write_bytes(b"arrays")
        (ckpt_path / "rollouts" / "t1.restore").write_bytes(b"point")
        write_open_work(work, ckpt_path / "rollouts")
        write_state({**build_state(Progress(step=7), source, {}), "rollouts": True}, ckpt_path)
        return manager

    manager = checkpoint(tmp_path / "run", rollouts=True)
    resumed = TrainSource([train_env("a")])
    progress = Progress()
    loaded = manager.load(progress, resumed, step=6)
    assert progress.step == 7 and not resumed.reopened
    resumed.adopt_open_groups(loaded["open_groups"], loaded["reopened"])
    assert resumed.open_groups == source.open_groups
    # The restore file is found where the checkpoint is now, and read in this run.
    (restored,) = loaded["episodes"]
    assert restored.restore.path == str((manager.ckpt_dir / RESUMED_DIR / "t1.restore").resolve())
    assert owned_snapshots(loaded) == [{"id": "s-t1"}]
    # The arrays and the restore file outlive the checkpoint directory.
    shutil.rmtree(manager.get_ckpt_path(6))
    assert (manager.ckpt_dir / RESUMED_DIR / "blob").read_bytes() == b"arrays"
    assert (manager.ckpt_dir / RESUMED_DIR / "t1.restore").read_bytes() == b"point"

    # A fork of the checkpoint leaves the snapshots to the run that took them.
    fork = CheckpointManager(tmp_path / "fork", CheckpointConfig(rollouts=True))
    source_path = checkpoint(tmp_path / "other", rollouts=True).get_ckpt_path(6)
    forked = fork.load(Progress(), TrainSource([train_env("a")]), step=6, path=source_path)
    assert owned_snapshots(forked) == []
    assert forked["episodes"][0].restore.path == str((fork.ckpt_dir / RESUMED_DIR / "t1.restore").resolve())

    # Without ckpt.rollouts the open groups re-run whole, as from a checkpoint without rollouts.
    manager = checkpoint(tmp_path / "off", rollouts=False)
    rerun = TrainSource([train_env("a")])
    assert manager.load(Progress(), rerun, step=6) is None
    assert [(env_name, task.key) for env_name, task, _ in rerun.reopened] == [("a", request.task.key)]


def test_a_forked_writer_completes_the_checkpoint(tmp_path):
    async def run():
        s = sink(group_size={"open": 2}.get)
        await s.add(episode([trace()], group="open"))
        source = TrainSource([train_env("a")])
        manager = CheckpointManager(tmp_path, CheckpointConfig(rollouts=True))
        work = open_work(sink=s.state_dict())
        state = build_state(Progress(step=7), source, {})
        pid = manager.fork_writer(state, work, step=6, freeze=False)
        # The writer holds the work as of the fork.
        s.pending_groups["open"].clear()
        assert await wait_for_writer(pid) >= 0.0

        loaded = manager.load(Progress(), TrainSource([train_env("a")]), step=6)
        assert [e.id for e in loaded["sink"]["pending_groups"]["open"]] == [
            e.id for e in work["sink"]["pending_groups"]["open"]
        ]
        # An earlier attempt's checkpoint of the step is cleared before it is written again.
        manager.clear(6)
        assert not (manager.get_ckpt_path(6) / "progress.pt").exists()
        assert not (manager.get_ckpt_path(6) / "rollouts").exists()

    asyncio.run(run())


def test_the_ledger_deletes_the_snapshots_of_pruned_steps(tmp_path):
    async def run():
        ckpt_dir = tmp_path / "checkpoints"
        refused: set[str] = set()
        deleted: list[str] = []

        async def discard(snapshots: list[dict]) -> list[str]:
            ids = [snapshot["id"] for snapshot in snapshots if snapshot["id"] not in refused]
            deleted.extend(ids)
            return ids

        def snapshot(name: str) -> dict:
            return {"id": name}

        ledger = SnapshotLedger(ckpt_dir / "rollout_snapshots.json")
        for step in (2, 4):
            (ckpt_dir / f"step_{step}").mkdir(parents=True)
        ledger.record(2, [snapshot("a"), snapshot("shared")])
        ledger.record(4, [snapshot("b"), snapshot("shared")])
        # Every checkpoint on disk keeps its snapshots.
        assert await ledger.sweep(ckpt_dir, discard) == 0 and deleted == []

        # The trainer's cleanup removed step 2: its snapshots go, except the one step 4 holds too.
        shutil.rmtree(ckpt_dir / "step_2")
        refused.add("a")
        assert await ledger.sweep(ckpt_dir, discard) == 0 and deleted == []
        # A refused snapshot stays in the ledger, also for a resumed run, until a sweep deletes it.
        ledger = SnapshotLedger(ckpt_dir / "rollout_snapshots.json")
        assert ledger.orphans == [snapshot("a")] and set(ledger.steps) == {4}
        refused.clear()
        assert await ledger.sweep(ckpt_dir, discard) == 1 and deleted == ["a"]
        assert ledger.orphans == []

        # A snapshot that restores nothing, and those of a step written again, go right away.
        ledger.orphan([snapshot("unused")])
        ledger.record(4, [snapshot("c"), snapshot("shared")])
        assert await ledger.sweep(ckpt_dir, discard) == 2 and deleted == ["a", "unused", "b"]
        assert SnapshotLedger(ckpt_dir / "rollout_snapshots.json").steps == {4: [snapshot("c"), snapshot("shared")]}

    asyncio.run(run())
