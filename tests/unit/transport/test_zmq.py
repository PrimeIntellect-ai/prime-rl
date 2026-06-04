import random
import socket
from pathlib import Path

import pytest

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transport.types import MicroBatch
from prime_rl.transport.zmq import ZMQMicroBatchReceiver, ZMQMicroBatchSender


def _free_base_port() -> int:
    for _ in range(100):
        base = random.randint(30_000, 60_000)
        sockets = []
        try:
            for port in (base + 1, base + 2):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(("0.0.0.0", port))
                sockets.append(sock)
        except OSError:
            pass
        else:
            return base
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("Could not find free ZMQ base port")


def _micro_batch(token: int = 1) -> MicroBatch:
    return MicroBatch(
        input_ids=[token],
        loss_mask=[True],
        advantages=[1.0],
        inference_logprobs=[0.0],
        position_ids=[0],
        temperatures=[1.0],
        env_names=["test"],
        lora_num_tokens=[1],
    )


def _transport(**overrides) -> ZMQTransportConfig:
    config = {"host": "127.0.0.1", "port": _free_base_port()}
    config.update(overrides)
    return ZMQTransportConfig(**config)


def test_zmq_micro_batch_routes_each_rank_topic(tmp_path: Path):
    transport = _transport(ready_timeout_seconds=2, recv_timeout_seconds=2)
    sender = ZMQMicroBatchSender(tmp_path, data_world_size=2, current_step=7, transport=transport)
    receiver_0 = ZMQMicroBatchReceiver(tmp_path, data_rank=0, current_step=7, transport=transport)
    receiver_1 = ZMQMicroBatchReceiver(tmp_path, data_rank=1, current_step=7, transport=transport)
    try:
        sender.send([[_micro_batch(10), _micro_batch(11)], [_micro_batch(20), _micro_batch(21)]])
        receiver_0.wait()
        receiver_1.wait()
        out_0 = receiver_0.receive()
        out_1 = receiver_1.receive()
    finally:
        receiver_0.close()
        receiver_1.close()
        sender.close()

    assert [micro_batch.input_ids for micro_batch in out_0] == [[10], [11]]
    assert [micro_batch.input_ids for micro_batch in out_1] == [[20], [21]]


def test_zmq_micro_batch_receive_timeout(tmp_path: Path):
    transport = _transport(recv_timeout_seconds=1)
    receiver = ZMQMicroBatchReceiver(tmp_path, data_rank=0, current_step=0, transport=transport)
    try:
        with pytest.raises(TimeoutError, match="Timed out waiting for ZMQ micro-batch"):
            receiver.wait()
    finally:
        receiver.close()


def test_zmq_micro_batch_ready_timeout(tmp_path: Path):
    transport = _transport(ready_timeout_seconds=1)
    sender = ZMQMicroBatchSender(tmp_path, data_world_size=1, current_step=0, transport=transport)
    try:
        with pytest.raises(TimeoutError, match="READY messages"):
            sender.send([[_micro_batch()]])
    finally:
        sender.close()


def test_zmq_micro_batch_step_mismatch_fails_fast(tmp_path: Path):
    transport = _transport(ready_timeout_seconds=2, recv_timeout_seconds=2)
    sender = ZMQMicroBatchSender(tmp_path, data_world_size=1, current_step=5, transport=transport)
    receiver = ZMQMicroBatchReceiver(tmp_path, data_rank=0, current_step=6, transport=transport)
    try:
        sender.send([[_micro_batch()]])
        receiver.wait()
        with pytest.raises(ValueError, match="expected 6"):
            receiver.receive()
    finally:
        receiver.close()
        sender.close()
