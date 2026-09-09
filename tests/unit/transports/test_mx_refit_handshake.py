"""ModelExpress handshake behavior."""

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from modelexpress_rl import WeightVersionState

import prime_rl.transports.weights.mx_refit as mx_refit
from prime_rl.configs.trainer import MXRefitWeightBroadcastConfig
from prime_rl.transports.weights.base import SENDER_READY_MARKER
from prime_rl.transports.weights.mx_phases import PhaseTimer
from prime_rl.transports.weights.mx_refit import MXRefitWeightReceiver, MXRefitWeightSender
from prime_rl.utils.pathing import get_broadcast_dir


class ControlClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def get_weight_version(self, uid: str):
        return SimpleNamespace(state=WeightVersionState.READY)

    def delete_weight_version(self, uid: str) -> None:
        self.deleted.append(uid)


class FailingAdminPlane:
    async def update_weights(self, *args, **kwargs) -> None:
        raise RuntimeError("install failed")


def test_sender_does_not_wait_for_receiver(tmp_path):
    config = MXRefitWeightBroadcastConfig(run_uid="run", timeout=0)
    sender = MXRefitWeightSender(tmp_path, config, parallel_dims=None, model_name="model")

    sender._wait_for_receiver_ready(tmp_path)


def test_rendezvous_wait_is_bounded(tmp_path):
    config = MXRefitWeightBroadcastConfig(run_uid="run", timeout=0)
    sender = MXRefitWeightSender(tmp_path, config, parallel_dims=None, model_name="model")
    sender._control = ControlClient()

    with pytest.raises(TimeoutError, match="No generator pulled"):
        sender._wait_released("run.token:1")


def test_failed_install_retires_version(tmp_path):
    config = MXRefitWeightBroadcastConfig(run_uid="run", timeout=1)
    receiver = MXRefitWeightReceiver(
        get_broadcast_dir(tmp_path),
        config,
        admin_plane=FailingAdminPlane(),
        model_name="model",
    )
    control = ControlClient()
    receiver._control = control
    step_dir = receiver.step_dir(1)
    step_dir.mkdir(parents=True)
    (step_dir / SENDER_READY_MARKER).write_text("run.token\n")

    with pytest.raises(RuntimeError, match="install failed"):
        asyncio.run(receiver.receive(1))

    assert control.deleted == ["run.token:1"]


def test_offer_lag_is_a_mark(tmp_path):
    config = MXRefitWeightBroadcastConfig(run_uid="run", timeout=1)
    receiver = MXRefitWeightReceiver(
        get_broadcast_dir(tmp_path),
        config,
        admin_plane=None,
        model_name="model",
    )
    step_dir = receiver.step_dir(1)
    step_dir.mkdir(parents=True)
    (step_dir / SENDER_READY_MARKER).touch()
    timer = PhaseTimer("orchestrator", 1, "")

    receiver._mark_offer_lag(1, timer)

    assert timer.marks["offer_lag_s"] >= 0
    assert "offer_lag_s" not in timer.phases


def test_sender_records_trainer_client_metrics(tmp_path, monkeypatch):
    class TrainerClient:
        def publish_version(self, *, version) -> None:
            pass

        def pop_metrics(self):
            return {"manifest_cache_hit": 1, "manifest_generation_s": 0.002}

        def release_version(self, *, version) -> None:
            pass

    timer = PhaseTimer("trainer", 1, "")

    @contextmanager
    def use_timer(*args, **kwargs):
        yield timer

    config = MXRefitWeightBroadcastConfig(run_uid="run", timeout=1)
    sender = MXRefitWeightSender(tmp_path, config, parallel_dims=None, model_name="model")
    sender.world = SimpleNamespace(world_size=1, is_master=False)
    sender._initialized = True
    sender._client = TrainerClient()
    sender._offer_token = "run.token"
    monkeypatch.setattr(mx_refit, "timed_refit", use_timer)
    monkeypatch.setattr(mx_refit.dist, "barrier", lambda: None)

    sender._broadcast(model=None, step=1, step_dir=tmp_path)

    assert timer.marks["manifest_cache_hit"] == 1
    assert timer.marks["manifest_generation_s"] == 0.002
