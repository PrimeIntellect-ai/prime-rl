"""ModelExpress version naming across restarts."""

import asyncio
from pathlib import Path

from prime_rl.configs.trainer import MXRefitWeightBroadcastConfig
from prime_rl.transports.weights.base import SENDER_READY_MARKER
from prime_rl.transports.weights.mx_refit import (
    MXRefitWeightReceiver,
    MXRefitWeightSender,
    weight_version_uid,
)
from prime_rl.utils.pathing import get_broadcast_dir

RUN_UID = "testrun"


def make_sender(output_dir: Path, timeout: int = 5) -> MXRefitWeightSender:
    config = MXRefitWeightBroadcastConfig(run_uid=RUN_UID, timeout=timeout)
    return MXRefitWeightSender(output_dir, config, parallel_dims=None, model_name="model")


def make_receiver(output_dir: Path, timeout: int = 5) -> MXRefitWeightReceiver:
    config = MXRefitWeightBroadcastConfig(run_uid="unused-by-this-side", timeout=timeout)
    return MXRefitWeightReceiver(get_broadcast_dir(output_dir), config, admin_plane=None, model_name="model")


def offer(sender: MXRefitWeightSender, step: int) -> Path:
    step_dir = sender.step_dir(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    sender._offer(step_dir)
    return step_dir


def test_restart_uses_new_version_id(tmp_path):
    first = make_sender(tmp_path)
    offer(first, 3)
    restarted = make_sender(tmp_path)
    offer(restarted, 3)

    assert first._offer_token != restarted._offer_token
    assert weight_version_uid(first._offer_token, 3) != weight_version_uid(restarted._offer_token, 3)


def test_each_offer_uses_new_version_id(tmp_path):
    sender = make_sender(tmp_path)
    offer(sender, 3)
    first = sender._offer_token
    offer(sender, 3)

    assert first != sender._offer_token


def test_version_id_contains_run_and_step(tmp_path):
    sender = make_sender(tmp_path)
    offer(sender, 7)
    uid = weight_version_uid(sender._offer_token, 7)

    assert uid.startswith(f"{RUN_UID}.")
    assert uid.rpartition(":")[2] == "7"


def test_receiver_reads_offered_version_id(tmp_path):
    sender = make_sender(tmp_path)
    receiver = make_receiver(tmp_path)
    offer(sender, 4)

    token = asyncio.run(receiver._read_offer_token(4))

    assert token == sender._offer_token
    assert weight_version_uid(token, 4) == weight_version_uid(sender._offer_token, 4)


def test_receiver_follows_restarted_trainer(tmp_path):
    receiver = make_receiver(tmp_path)
    stale = make_sender(tmp_path)
    offer(stale, 5)
    stale_uid = weight_version_uid(stale._offer_token, 5)

    restarted = make_sender(tmp_path)
    offer(restarted, 5)

    resolved = weight_version_uid(asyncio.run(receiver._read_offer_token(5)), 5)
    assert resolved == weight_version_uid(restarted._offer_token, 5)
    assert resolved != stale_uid


def test_offer_marker_is_atomic(tmp_path):
    sender = make_sender(tmp_path)
    step_dir = offer(sender, 2)

    assert (step_dir / SENDER_READY_MARKER).read_text().strip() == sender._offer_token
    assert [path.name for path in step_dir.iterdir()] == [SENDER_READY_MARKER]


def test_receiver_run_uid_is_not_used(tmp_path):
    sender = make_sender(tmp_path)
    receiver = make_receiver(tmp_path)
    offer(sender, 8)

    assert receiver.config.run_uid != sender.config.run_uid
    assert asyncio.run(receiver._read_offer_token(8)) == sender._offer_token
