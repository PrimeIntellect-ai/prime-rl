"""Kill a batch-consuming process and restore its checkpoint against a live sender.

The scalar update stands in for an optimizer; this tests the journal/transport
contract without inference, weight publication or a process supervisor.
"""

import asyncio
import multiprocessing
import socket
from uuid import uuid4

import msgspec
import numpy as np
import pytest
import zmq

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.journal import BatchCheckpoint, BatchJournal, _write_atomic
from prime_rl.transports.batch.recovery import TrainerGroup
from prime_rl.transports.batch.types import MicroBatch
from prime_rl.transports.batch.zmq import ZMQBatchReceiver, ZMQBatchSender


def _trainer(output, port, group, crash_phase, events):
    journal = BatchJournal(output)
    if group.restored is None:
        state = {"value": 0, "batches": []}
    else:
        path = output / "checkpoints" / f"step_{group.restored.step}" / "trainer" / "state.json"
        state = msgspec.json.decode(path.read_bytes())
    receiver = ZMQBatchReceiver(output, 0, 0, 1, group.start_step, ZMQTransportConfig(port=port), group)
    receiver.socket.setsockopt(zmq.RCVTIMEO, 15000)

    def stop(phase, step):
        if phase == crash_phase and step == 2:
            events.send(("crash_ready", phase))
            events.recv()

    try:
        for step in range(group.start_step, 5):
            stop("before_receipt", step)
            receiver.wait()
            batch = receiver.receive()[0]
            stop("after_receipt", step)
            state["value"] += int(np.frombuffer(batch.input_ids, np.int32).sum())
            state["batches"].append(batch.journal_batch_id)
            stop("after_update", step)
            if step % 2 == 0:
                checkpoint = output / "checkpoints" / f"step_{step}" / "trainer"
                checkpoint.mkdir(parents=True, exist_ok=True)
                (checkpoint / "state.partial").write_bytes(b"incomplete")
                stop("during_checkpoint", step)
                _write_atomic(checkpoint / "state.json", msgspec.json.encode(state))
                journal.commit_checkpoint(checkpoint, step, batch.journal_batch_id)
                stop("after_checkpoint", step)
        events.send(("done", state))
    finally:
        receiver.close()
        events.close()


def _batch(step):
    return MicroBatch(
        input_ids=np.array([step + 1], np.int32),
        loss_mask=np.array([True]),
        advantages=np.array([1.0], np.float32),
        inference_logprobs=np.array([0.0], np.float32),
        position_ids=np.array([0], np.int32),
        sequence_lengths=[1],
        temperatures=np.array([1.0], np.float32),
        env_names=["fault-test"],
        seq_lens=[1],
    )


@pytest.mark.parametrize(
    "crash_phase", ["before_receipt", "after_receipt", "after_update", "during_checkpoint", "after_checkpoint", "twice"]
)
def test_killed_trainer_replays_without_restarting_sender(tmp_path, crash_phase):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    sender = ZMQBatchSender(tmp_path, 1, 0, ZMQTransportConfig(port=port))
    sender.journal = BatchJournal(tmp_path)
    context = multiprocessing.get_context("spawn")
    processes, connections = [], []

    def launch(group, phase):
        parent, child = context.Pipe()
        process = context.Process(target=_trainer, args=(tmp_path, port, group, phase, child))
        process.start()
        child.close()
        processes.append(process)
        connections.append(parent)
        return process, parent

    async def event(connection):
        assert await asyncio.to_thread(connection.poll, 20), "Trainer did not reach the expected phase"
        return connection.recv()

    async def produce():
        for step in range(5):
            await sender.send([[_batch(step)]])

    async def run():
        sending = asyncio.create_task(produce())
        group = TrainerGroup(uuid4().hex, 1, 0)
        process, connection = launch(group, "after_update" if crash_phase == "twice" else crash_phase)
        try:
            for crash in range(2 if crash_phase == "twice" else 1):
                assert (await event(connection))[0] == "crash_ready"
                process.kill()
                await asyncio.to_thread(process.join, 5)
                assert not process.is_alive() and process.exitcode != 0
                records = [
                    msgspec.json.decode(path.read_bytes(), type=BatchCheckpoint)
                    for path in (tmp_path / "checkpoints").glob("step_*/trainer/batch.json")
                ]
                restored = max(records, key=lambda record: record.step)
                assert restored.step == (2 if crash_phase == "after_checkpoint" else 0)
                replacement = TrainerGroup(uuid4().hex, 1, restored.step + 1, restored)
                sender.authorize_recovery(group.attempt_id, replacement.attempt_id)
                phase = "after_update" if crash_phase == "twice" and crash == 0 else None
                process, connection = launch(replacement, phase)
                group = replacement
            kind, state = await event(connection)
            assert kind == "done"
            await asyncio.wait_for(sending, 10)
            await asyncio.to_thread(process.join, 5)
            assert process.exitcode == 0
            assert state == {"value": 15, "batches": [f"{sender.journal_session}:{step}" for step in range(5)]}
            assert sender.current_step == 5 and sender.replay_batches_sent > 0
            assert not list(sender.journal.path.glob("*/step_*"))
        finally:
            if not sending.done():
                sending.cancel()
                await asyncio.gather(sending, return_exceptions=True)

    try:
        asyncio.run(run())
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=5)
        for connection in connections:
            connection.close()
        sender.close()
