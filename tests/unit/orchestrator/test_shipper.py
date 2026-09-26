import pytest

from prime_rl.configs.orchestrator import CheckpointConfig, OrchestratorConfig
from prime_rl.orchestrator import shipper as shipper_module
from prime_rl.orchestrator.shipper import Shipper
from prime_rl.orchestrator.types import Batch
from tests.unit.orchestrator.fakes import RecordingHooks, RecordingMonitors, make_group


def make_batch(n: int = 2, *, policy=(0, 0)) -> Batch:
    return Batch(step=1, groups=[make_group(1, policy=policy) for _ in range(n)])


class FakeSender:
    def __init__(self):
        self.sent = []

    async def send(self, grid):
        self.sent.append(grid)

    def close(self):
        pass


class FakeTrainSource:
    env_names = ["env"]

    def __init__(self):
        self.loaded = None

    def metrics(self):
        return {}

    def state_dict(self):
        return {"envs": {"env": {}}}

    def load_state_dict(self, state):
        self.loaded = state


def make_shipper(tmp_path, monkeypatch, *, version=0, max_steps=None, max_off_policy_steps=1, resume_step=None):
    """A shipper over a checkpoint dir in ``tmp_path``, with the packer and the batch
    transport replaced: those need a model and a trainer."""
    monkeypatch.setattr(shipper_module, "BatchPacker", lambda config: type("P", (), {"pack": lambda self, s: [[s]]})())
    sender = FakeSender()
    monkeypatch.setattr(shipper_module, "setup_batch_sender", lambda *args: sender)
    config = OrchestratorConfig.model_construct(
        output_dir=tmp_path,
        max_steps=max_steps,
        max_off_policy_steps=max_off_policy_steps,
        ckpt=CheckpointConfig(interval=2),
        resume=None,
        heartbeat=None,
    )
    source = FakeTrainSource()
    shipper = Shipper(config, train_source=source, resume_step=resume_step)
    hooks, monitors = RecordingHooks(), RecordingMonitors()
    state = {"version": version}

    async def wait_for_version(v, *, reason=""):
        hooks.record("wait")(v)
        state["version"] = max(state["version"], v)
        return True

    shipper.bind(
        version=lambda: state["version"],
        wait_for_version=wait_for_version,
        gate=hooks.record("gate"),
        on_drain=hooks.record_async("drain"),
        monitors=monitors,
    )
    return shipper, hooks, monitors, sender, state


@pytest.mark.asyncio
async def test_ship_advances_the_step_and_reports_metrics(tmp_path, monkeypatch):
    shipper, _, monitors, sender, _ = make_shipper(tmp_path, monkeypatch)
    shipper.start_clock()
    await shipper.on_batch(make_batch())
    assert shipper.step() == 2
    assert len(sender.sent) == 1
    assert [(step, kind, subset) for _, step, kind, subset in monitors.episodes] == [(1, "train", "effective")]
    ((metrics, step),) = monitors.metrics
    assert step == 1 and metrics["progress/rollouts"] == 2 and metrics["step"] == 1
    assert metrics["train/agg/effective/agent/reward/mean"] == 1.0
    assert shipper.progress.total_samples == 2
    assert not (tmp_path / "checkpoints").exists()  # interval 2: step 1 is not a boundary
    await shipper.on_batch(make_batch())
    assert (tmp_path / "checkpoints" / "step_2" / "orchestrator" / "progress.pt").exists()


@pytest.mark.asyncio
async def test_lag_gate_closes_past_max_off_policy_steps_and_reopens_on_version(tmp_path, monkeypatch):
    shipper, hooks, _, _, state = make_shipper(tmp_path, monkeypatch, max_off_policy_steps=1)
    await shipper.on_batch(make_batch())  # step 1 shipped, lead = 1 - 0 = 1 -> open
    assert hooks["gate"][-1] == (True,)
    await shipper.on_batch(make_batch())  # step 2 shipped, lead 2 > 1 -> closed
    assert hooks["gate"][-1] == (False,)
    state["version"] = 1
    await shipper.on_version(1)
    assert hooks["gate"][-1] == (True,)
    assert shipper.wait_for_policy_time > 0


@pytest.mark.asyncio
async def test_batch_holds_for_the_required_version(tmp_path, monkeypatch):
    shipper, hooks, _, sender, _ = make_shipper(tmp_path, monkeypatch, max_off_policy_steps=1)
    shipper.progress.step = 5
    await shipper.on_batch(make_batch())
    # batch 5 needs v3 = 5 - 1 - max_off_policy_steps
    assert hooks["wait"] == [(3,)]
    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_final_step_drains_and_later_batches_are_dropped(tmp_path, monkeypatch):
    shipper, hooks, _, sender, _ = make_shipper(tmp_path, monkeypatch, max_steps=1)
    await shipper.on_batch(make_batch())
    assert shipper.draining.is_set()
    assert hooks["drain"] == [("Shipped the final batch",)]
    await shipper.on_batch(make_batch())
    assert len(sender.sent) == 1


def test_resume_restores_progress_and_the_final_checkpoint_lands_on_the_last_step(tmp_path, monkeypatch):
    shipper, _, _, _, _ = make_shipper(tmp_path, monkeypatch)
    shipper.progress.total_samples = 7
    shipper.save_ckpt(4)
    resumed, _, _, _, _ = make_shipper(tmp_path, monkeypatch, resume_step=4)
    assert resumed.step() == 5 and resumed.progress.total_samples == 7
    assert resumed.train_source.loaded == {"envs": {"env": {}}}
    resumed.save_final()
    assert resumed.step() == 4
    assert (tmp_path / "checkpoints" / "step_4" / "orchestrator" / "progress.pt").exists()
