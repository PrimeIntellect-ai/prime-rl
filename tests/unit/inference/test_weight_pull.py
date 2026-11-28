"""Engines that pull their weights from ModelExpress: the puller's switch protocol, and the worker's staging."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from modelexpress_rl import WeightVersionState

from prime_rl.inference.vllm import weight_pull
from prime_rl.inference.vllm.weight_pull import PullConfig, WeightPuller, mesh_key

CONFIG = PullConfig(
    host="mx", port=1, model_name="model", staging_buffer_bytes=None, staging_buffers_count=1, timeout=60
)


def version(step: int, created: int, mesh: str = "mesh-a", state=WeightVersionState.READY):
    return SimpleNamespace(
        version_id=f"uid-{step}-{created}",
        version_number=step,
        created_at_unix_ms=created,
        state=state,
        trainer_mesh_id=mesh,
        trainer_mesh_generation=1,
    )


class FakeEngines:
    """Records the protocol; ``live`` workers are those that installed a version of the mesh before."""

    def __init__(self, fail_stage: int = 0, fail_apply: int = 0):
        self.calls = []
        self.meshes: set[str] = set()
        self.paused = False
        self.label = None
        self.fail_stage = fail_stage
        self.fail_apply = fail_apply

    async def ready(self):
        self.calls.append("ready")

    async def init(self, payload):
        self.calls.append("init")

    async def can_stage_live(self, mesh):
        return mesh in self.meshes

    async def stage(self, version_uid, live):
        assert live != self.paused, "a live stage runs while serving, any other while paused"
        self.calls.append(("stage", version_uid, live))
        if self.fail_stage:
            self.fail_stage -= 1
            raise RuntimeError("stage failed")

    async def discard(self):
        self.calls.append("discard")

    async def pause(self):
        self.paused = True
        self.calls.append("pause")

    async def apply(self, version_uid, mesh, step):
        assert self.paused, "installs only while paused"
        self.calls.append(("apply", version_uid))
        if self.fail_apply:
            self.fail_apply -= 1
            raise RuntimeError("install failed")
        self.meshes.add(mesh)
        self.label = step

    async def resume(self):
        self.paused = False
        self.calls.append("resume")


def puller(engines, versions):
    puller = WeightPuller(engines, CONFIG)
    puller._control = SimpleNamespace(list_weight_versions=lambda model_name: list(versions))
    return puller


def run(coroutine):
    return asyncio.run(coroutine)


def test_the_first_version_of_a_trainer_stages_paused_and_later_ones_while_serving():
    engines = FakeEngines()
    versions = [version(0, created=10)]
    pull = puller(engines, versions)
    run(pull.switch(run(pull.newest(after=None))))
    assert engines.calls == ["pause", ("stage", "uid-0-10", False), ("apply", "uid-0-10"), "resume"]
    assert engines.label == 0 and not engines.paused

    engines.calls.clear()
    versions.append(version(1, created=20))
    run(pull.switch(run(pull.newest(after=pull.served))))
    assert engines.calls == [("stage", "uid-1-20", True), "pause", ("apply", "uid-1-20"), "resume"]
    assert engines.label == 1


def test_the_puller_takes_the_newest_ready_version_and_skips_the_ones_between():
    versions = [
        version(3, created=30),
        version(5, created=50),
        version(4, created=40),
        version(6, created=60, state=WeightVersionState.STAGING),
        version(2, created=20, state=WeightVersionState.RELEASING),
    ]
    pull = puller(FakeEngines(), versions)
    assert run(pull.newest(after=None)).version_number == 5
    assert run(pull.newest(after=version(5, created=50))) is None


def test_the_puller_follows_a_restarted_trainer_back_to_a_lower_step():
    """Publication order decides: a trainer restarted from a checkpoint publishes lower steps."""
    pull = puller(FakeEngines(), [version(7, created=70, mesh="old"), version(4, created=80, mesh="new")])
    newest = run(pull.newest(after=version(7, created=70, mesh="old")))
    assert newest.version_number == 4 and newest.trainer_mesh_id == "new"


def test_a_failed_live_stage_discards_and_never_pauses():
    engines = FakeEngines()
    engines.meshes.add(mesh_key(version(1, created=20)))
    engines.fail_stage = 1
    pull = puller(engines, [version(1, created=20)])
    with pytest.raises(RuntimeError, match="stage failed"):
        run(pull.switch(version(1, created=20)))
    assert engines.calls == [("stage", "uid-1-20", True), "discard"]
    assert pull.served is None and engines.label is None


def test_a_failed_install_stays_paused_and_installs_the_newest_version_before_resuming(monkeypatch):
    async def no_sleep(seconds):
        pass

    monkeypatch.setattr(weight_pull.asyncio, "sleep", no_sleep)
    engines = FakeEngines(fail_apply=1)
    engines.meshes.add(mesh_key(version(1, created=20)))
    versions = [version(1, created=20)]
    pull = puller(engines, versions)

    async def publish_while_installing(version_uid, mesh, step, apply=engines.apply):
        versions.append(version(2, created=30))
        await apply(version_uid, mesh, step)

    monkeypatch.setattr(engines, "apply", publish_while_installing)
    run(pull.switch(version(1, created=20)))
    assert engines.calls == [
        ("stage", "uid-1-20", True),
        "pause",
        ("apply", "uid-1-20"),
        "discard",
        # the weights may be partly written: stage again while paused, the newest version
        ("stage", "uid-2-30", False),
        ("apply", "uid-2-30"),
        "resume",
    ]
    assert pull.served.version_number == 2 and engines.label == 2


def test_a_pause_that_fails_resumes_without_installing():
    engines = FakeEngines()

    async def fail_pause():
        raise RuntimeError("pause failed")

    engines.pause = fail_pause
    pull = puller(engines, [version(0, created=10)])
    with pytest.raises(RuntimeError, match="pause failed"):
        run(pull.switch(version(0, created=10)))
    assert engines.calls == ["discard", "resume"]


class FakeGenerator:
    """A generator client whose stage blocks until released by the test."""

    def __init__(self, fail: bool = False):
        self.go = threading.Event()
        self.fail = fail
        self.applied = []
        self.released = []

    def stage_weight(self, *, version):
        self.go.wait(timeout=10)
        if self.fail:
            raise RuntimeError("no source")
        return SimpleNamespace(version=version.version_id, release=lambda: self.released.append(version.version_id))

    def apply_weight(self, handle):
        self.applied.append(handle.version)


def worker(generator):
    from prime_rl.inference.vllm.worker.modelexpress import ModelExpressWeightUpdateWorker

    worker = ModelExpressWeightUpdateWorker()
    worker._generator = generator
    worker._mx_stream = SimpleNamespace(synchronize=lambda: None)
    worker.device = "cuda:0"
    return worker


@pytest.fixture
def no_cuda(monkeypatch):
    import contextlib

    import torch

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())


def test_a_worker_stages_on_a_background_thread_and_installs_only_what_it_staged(no_cuda):
    generator = FakeGenerator()
    w = worker(generator)
    assert not w.modelexpress_can_stage_live("mesh-a")
    w.modelexpress_stage("uid-1", live=True)
    assert w.modelexpress_stage_status() == "staging"
    with pytest.raises(RuntimeError, match="not staged"):
        w.modelexpress_apply("uid-1", "mesh-a")
    generator.go.set()
    w._mx_staged.done.wait(timeout=10)
    assert w.modelexpress_stage_status() == "staged"
    with pytest.raises(RuntimeError, match="not staged"):
        w.modelexpress_apply("uid-other", "mesh-a")
    assert w.modelexpress_served() is None
    w.modelexpress_apply("uid-1", "mesh-a")
    assert w.modelexpress_served() == "uid-1"
    assert generator.applied == ["uid-1"] and generator.released == ["uid-1"]
    assert w.modelexpress_stage_status() == "idle"
    assert w.modelexpress_can_stage_live("mesh-a") and not w.modelexpress_can_stage_live("mesh-b")


def test_a_failed_background_stage_reports_its_error_and_a_discard_releases_a_stage(no_cuda):
    generator = FakeGenerator(fail=True)
    generator.go.set()
    w = worker(generator)
    w.modelexpress_stage("uid-1", live=True)
    w._mx_staged.done.wait(timeout=10)
    assert w.modelexpress_stage_status().startswith("failed: RuntimeError: no source")

    generator.fail = False
    w.modelexpress_stage("uid-2", live=False)
    assert w.modelexpress_stage_status() == "staged"
    w.modelexpress_discard()
    assert generator.released == ["uid-2"] and w.modelexpress_stage_status() == "idle"


def test_a_failed_install_makes_the_next_stage_run_paused(no_cuda):
    generator = FakeGenerator()
    generator.go.set()
    w = worker(generator)
    w.modelexpress_stage("uid-1", live=False)
    w.modelexpress_apply("uid-1", "mesh-a")
    assert w.modelexpress_can_stage_live("mesh-a")

    def fail(handle):
        raise RuntimeError("install failed")

    generator.apply_weight = fail
    w.modelexpress_stage("uid-2", live=True)
    w._mx_staged.done.wait(timeout=10)
    with pytest.raises(RuntimeError, match="install failed"):
        w.modelexpress_apply("uid-2", "mesh-a")
    assert w.modelexpress_served() is None
    assert generator.released == ["uid-1", "uid-2"]
    assert not w.modelexpress_can_stage_live("mesh-a")
