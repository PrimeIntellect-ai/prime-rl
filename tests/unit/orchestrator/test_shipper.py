from types import SimpleNamespace

import pytest

from prime_rl.configs.orchestrator import CheckpointConfig, ShipperConfig
from prime_rl.orchestrator.metrics import TrainEpisodes
from prime_rl.orchestrator.shipper import Shipper
from prime_rl.orchestrator.types import TrainBatch
from tests.unit.orchestrator.fakes import RecordingHooks, RecordingMonitors, make_episode, make_sample


def make_batch(n: int = 2, *, policy=(0, 0)) -> TrainBatch:
    episodes = [make_episode(group_id=f"g{i}", policy=policy) for i in range(n)]
    ids = {ep.traces[0].id for ep in episodes}
    return TrainBatch(
        episodes=TrainEpisodes(list(episodes), sampled_trace_ids=ids),
        cohort=TrainEpisodes(list(episodes), sampled_trace_ids=ids),
        samples=[make_sample() for _ in episodes],
        failures=[],
        buffered_episode_ids=set(),
    )


class FakeSender:
    def __init__(self):
        self.sent = []

    async def send(self, grid):
        self.sent.append(grid)

    def close(self):
        pass


def make_shipper(tmp_path, *, version=0, **config):
    hooks, monitors = RecordingHooks(), RecordingMonitors()
    sender = FakeSender()
    saved = []
    ckpt_manager = SimpleNamespace(save=lambda step, progress, source: saved.append((step, progress.step)))
    train_source = SimpleNamespace(metrics=lambda: {}, env_names=["env"], state_dict=lambda: {})
    shipper = Shipper(
        ShipperConfig(**config),
        packer=SimpleNamespace(pack=lambda samples: [[samples]]),
        sender=sender,
        ckpt_manager=ckpt_manager,
        ckpt_config=CheckpointConfig(interval=2),
        train_source=train_source,
    )
    state = {"version": version}

    async def wait_for_version(v, *, timeout=None, reason=""):
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
    return shipper, hooks, monitors, sender, saved, state


@pytest.mark.asyncio
async def test_ship_advances_the_step_and_reports_metrics(tmp_path):
    shipper, hooks, monitors, sender, saved, _ = make_shipper(tmp_path)
    shipper.start_clock()
    await shipper.on_batch(make_batch())
    assert shipper.step() == 2
    assert len(sender.sent) == 1
    assert [(step, kind, subset) for _, step, kind, subset in monitors.episodes] == [(1, "train", "effective")]
    ((metrics, step),) = monitors.metrics
    assert step == 1 and metrics["progress/rollouts"] == 2 and metrics["step"] == 1
    assert shipper.progress.total_samples == 2
    assert saved == []  # interval 2: step 1 is not a boundary
    await shipper.on_batch(make_batch())
    assert saved == [(2, 3)]


@pytest.mark.asyncio
async def test_lag_gate_closes_past_target_lag_and_reopens_on_version(tmp_path):
    shipper, hooks, _, _, _, state = make_shipper(tmp_path, target_lag=1)
    await shipper.on_batch(make_batch())  # step 1 shipped, lead = 1 - 0 = 1 -> open
    assert hooks["gate"][-1] == (True,)
    await shipper.on_batch(make_batch())  # step 2 shipped, lead 2 > 1 -> closed
    assert hooks["gate"][-1] == (False,)
    state["version"] = 1
    await shipper.on_version(1)
    assert hooks["gate"][-1] == (True,)
    assert shipper.wait_for_policy_time > 0


@pytest.mark.asyncio
async def test_batch_holds_for_the_required_version(tmp_path):
    shipper, hooks, _, sender, _, state = make_shipper(tmp_path, target_lag=1)
    shipper.progress.step = 5
    await shipper.on_batch(make_batch())
    # batch 5 needs v3 = 5 - 1 - target_lag
    assert hooks["wait"] == [(3,)]
    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_final_step_drains_and_later_batches_are_dropped(tmp_path):
    shipper, hooks, _, sender, _, _ = make_shipper(tmp_path, max_steps=1)
    await shipper.on_batch(make_batch())
    assert shipper.draining.is_set()
    assert hooks["drain"] == [("Shipped the final batch",)]
    await shipper.on_batch(make_batch())
    assert len(sender.sent) == 1


@pytest.mark.asyncio
async def test_empty_batches_ship_nothing(tmp_path):
    shipper, _, _, sender, _, _ = make_shipper(tmp_path)
    batch = make_batch()
    batch.samples = []
    await shipper.on_batch(batch)
    assert sender.sent == [] and shipper.step() == 1


def test_resume_and_final_checkpoint(tmp_path):
    shipper, _, _, _, saved, _ = make_shipper(tmp_path)
    shipper.resume(4, None)
    assert shipper.step() == 5
    shipper.save_final()
    assert saved == [(4, 4)]
