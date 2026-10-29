import asyncio
import time
from pathlib import Path
from uuid import UUID

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.recovery import (
    BatchRequest,
    HandshakeReply,
    RecoveryRequired,
    ReplayPlan,
    TrainerGroup,
    TrainerHandshake,
    TrainerHello,
    plan_replay,
)
from prime_rl.transports.batch.types import MicroBatch


class ZMQBatchSender(BatchSender):
    def __init__(self, output_dir: Path, data_world_size: int, current_step: int, transport: ZMQTransportConfig):
        """ZMQ micro batch sender. One ROUTER socket in the orchestrator answers the DEALER socket of
        every trainer rank, replying to each rank on the connection it asked on."""
        super().__init__(output_dir, data_world_size)
        # Async context so ``send`` yields instead of blocking the orchestrator event loop
        self.context = zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.ROUTER)
        # Replying to a rank that went away raises instead of dropping silently
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")
        self.request_decoder = msgspec.msgpack.Decoder(type=TrainerHello | BatchRequest)
        self.handshake = TrainerHandshake(data_world_size)
        self.current_step = current_step
        self.delivery_step = current_step
        self.replay_plan: ReplayPlan | None = None
        self.recovery_authorization: tuple[str, str] | None = None
        self.failed_attempt: str | None = None
        self.replay_batches_sent = 0
        self.logger.debug(f"Initialized ZMQ micro batch sender (endpoint=tcp://{transport.host}:{transport.port})")

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers. Encoding happens before the wait, so only the
        hand-off sits between the last request and the batch."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        buffers = await asyncio.to_thread(self.encode_batch, micro_batch_grid, self.current_step)
        send_time = 0.0
        while True:
            identity_to_request = await self._wait_for_requests()
            replaying = self.replay_plan is not None
            delivery = await asyncio.to_thread(self._read_replay_batch) if replaying else buffers
            send_start = time.perf_counter()
            for identity, request in identity_to_request.items():
                if request.attempt_id == self.failed_attempt:
                    break
                data_rank = self.handshake.validate_request(identity, request, self.delivery_step)
                delivered = await self._reply(
                    identity,
                    [b"batch", request.attempt_id.encode(), str(self.delivery_step).encode(), delivery[data_rank]],
                )
                if not delivered:
                    if self.journal is None:
                        raise ConnectionError("Trainer disconnected during batch delivery without a recovery journal")
                    self.failed_attempt = request.attempt_id
                    self.logger.warning(
                        f"Trainer attempt {request.attempt_id} disconnected during delivery of step {self.delivery_step}; "
                        "retaining the batch and waiting for an authorized replacement"
                    )
                    break
            else:
                send_time += time.perf_counter() - send_start
                if replaying:
                    self.replay_batches_sent += 1
                    self.logger.info(f"Replayed batch {self.journal_session}:{self.delivery_step}")
                if self.delivery_step == self.current_step:
                    self.current_step += 1
                    self.delivery_step = self.current_step
                    self.replay_plan = None
                    return send_time
                del delivery
                self.delivery_step += 1
                continue
            send_time += time.perf_counter() - send_start
            del delivery

    def authorize_recovery(self, previous_attempt: str, replacement_attempt: str) -> None:
        """Authorize one replacement on the sender's event loop after the old group is stopped.

        This fences the batch channel only. The caller must also stop old checkpoint
        writers and reconcile the weight channel before allowing production training.
        """
        if self.journal is None:
            raise ValueError("Trainer recovery requires batch journaling")
        active = self.handshake.active
        if active is None or active.attempt_id != previous_attempt:
            raise ValueError("Recovery authorization does not match the active trainer attempt")
        if UUID(replacement_attempt).hex != replacement_attempt or replacement_attempt == previous_attempt:
            raise ValueError("Recovery requires a new trainer attempt ID")
        if replacement_attempt in self.handshake.retired:
            raise ValueError("Cannot authorize a retired trainer attempt")
        if self.recovery_authorization is not None:
            raise ValueError("A trainer replacement is already authorized")
        # Freeze old requests immediately, including requests already collected.
        self.failed_attempt = previous_attempt
        self.recovery_authorization = (previous_attempt, replacement_attempt)
        if self.handshake.candidate is not None and self.handshake.candidate.attempt_id != replacement_attempt:
            self.handshake.reject()

    def _read_replay_batch(self) -> list[bytes]:
        plan = self.replay_plan
        record = plan.batches[self.delivery_step - plan.restored.step - 1]
        if self.journal.record(record.batch_id) != record:
            raise ValueError(f"Replay manifest changed after validation: {record.batch_id}")
        return self.journal.read(record.batch_id)

    async def _reply(self, identity: bytes, frames: list[bytes]) -> bool:
        try:
            await self.socket.send_multipart([identity, *frames], copy=False)
        except zmq.ZMQError as error:
            if error.errno != zmq.EHOSTUNREACH:
                raise
            return False
        return True

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch sender")

    async def _hello(self, identity: bytes, hello: TrainerHello) -> None:
        try:
            complete = self.handshake.add(identity, hello)
        except ValueError as error:
            reply = HandshakeReply(hello.group.attempt_id, "rejected", error=str(error))
            recipients = {identity}
            candidate = self.handshake.candidate
            if candidate is not None and candidate.attempt_id == hello.group.attempt_id:
                recipients.update(connection for connection, _ in self.handshake.pending.values())
                self.handshake.reject()
            for connection in recipients:
                await self._reply(connection, [b"hello", msgspec.msgpack.encode(reply)])
            return
        if not complete:
            return
        group = hello.group
        recipients = [connection for connection, _ in self.handshake.pending.values()]
        if self.handshake.active is None and group.start_step == self.current_step:
            self.handshake.accept()
            reply = HandshakeReply(group.attempt_id, "accepted")
            restored = group.restored.batch.batch_id if group.restored is not None else None
            self.logger.info(
                f"Accepted trainer attempt {group.attempt_id}, next step {group.start_step}, restored={restored}"
            )
        else:
            try:
                if self.journal is None:
                    raise ValueError("Trainer recovery requires batch journaling")
                plan = await asyncio.to_thread(
                    plan_replay, self.journal, group, self.journal_session, self.current_step, self.data_world_size
                )
                active = self.handshake.active
                if active is not None and self.recovery_authorization == (active.attempt_id, group.attempt_id):
                    self.handshake.accept()
                    self.delivery_step = group.start_step
                    self.replay_plan = plan
                    self.recovery_authorization = None
                    self.failed_attempt = None
                    reply = HandshakeReply(group.attempt_id, "accepted", plan=plan)
                    self.logger.info(
                        f"Accepted replacement {group.attempt_id} after {active.attempt_id}; "
                        f"replaying {len(plan.batches)} batches after {plan.restored.batch.batch_id} "
                        f"through pending step {self.current_step}"
                    )
                else:
                    reply = HandshakeReply(group.attempt_id, "recovery_required", plan=plan)
                    self.logger.warning(
                        f"Trainer attempt {group.attempt_id} restored {plan.restored.batch.batch_id}; "
                        f"{len(plan.batches)} batches await replay through step {self.current_step}. "
                        "Replacement requires explicit authorization."
                    )
            except (ValueError, OSError) as error:
                reply = HandshakeReply(group.attempt_id, "rejected", error=str(error))
                self.logger.warning(f"Rejected trainer recovery attempt {group.attempt_id}: {error}")
            if self.handshake.candidate is not None:
                if self.recovery_authorization is not None and self.recovery_authorization[1] == group.attempt_id:
                    self.recovery_authorization = None
                self.handshake.reject()
        for connection in recipients:
            await self._reply(connection, [b"hello", msgspec.msgpack.encode(reply)])

    async def _wait_for_requests(self) -> dict[bytes, BatchRequest]:
        """Only distinct ranks of the accepted group can request the pending live batch."""
        identity_to_request = {}
        while (
            self.handshake.active is None
            or self.failed_attempt == self.handshake.active.attempt_id
            or len(identity_to_request) < self.handshake.active.world_size
        ):
            identity, payload = await self.socket.recv_multipart()
            message = self.request_decoder.decode(payload)
            if isinstance(message, TrainerHello):
                previous = self.handshake.active
                await self._hello(identity, message)
                if previous != self.handshake.active or (
                    self.handshake.active is not None and self.failed_attempt == self.handshake.active.attempt_id
                ):
                    identity_to_request.clear()
                continue
            try:
                if message.attempt_id == self.failed_attempt:
                    raise ValueError("Trainer attempt is fenced pending recovery")
                self.handshake.validate_request(identity, message, self.delivery_step)
            except ValueError as error:
                await self._reply(identity, [b"error", str(error).encode()])
                continue
            identity_to_request[identity] = message
        return identity_to_request


class ZMQBatchReceiver(BatchReceiver):
    def __init__(
        self,
        output_dir: Path,
        data_rank: int,
        rank: int,
        world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
        trainer_group: TrainerGroup,
    ):
        """ZMQ micro batch receiver. Every trainer rank owns one DEALER socket and requests each step
        before reading it."""
        super().__init__(output_dir, data_rank, rank, world_size)
        self.context = zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.DEALER)
        self.socket.connect(f"tcp://{transport.host}:{transport.port}")
        self.request_encoder = msgspec.msgpack.Encoder()
        self.current_step = current_step
        if trainer_group.world_size != world_size or trainer_group.start_step != current_step:
            raise ValueError("Trainer group does not match receiver world size or start step")
        self.trainer_group = trainer_group
        self.accepted = False
        self.replay_plan: ReplayPlan | None = None
        self.logger.debug(f"Initialized ZMQ micro batch receiver (endpoint=tcp://{transport.host}:{transport.port})")

    def wait(self) -> None:
        if not self.accepted:
            hello = TrainerHello(self.trainer_group, self.rank, self.data_rank)
            self.socket.send(self.request_encoder.encode(hello))
            frames = self.socket.recv_multipart()
            if len(frames) != 2 or frames[0] != b"hello":
                raise RuntimeError("Expected trainer handshake reply")
            reply = msgspec.msgpack.decode(frames[1], type=HandshakeReply)
            if reply.attempt_id != self.trainer_group.attempt_id:
                raise RuntimeError("Handshake reply belongs to a different trainer attempt")
            if reply.status == "recovery_required":
                if reply.plan is None:
                    raise RuntimeError("Recovery reply is missing the replay plan")
                raise RecoveryRequired(reply.plan)
            if reply.status != "accepted":
                raise RuntimeError(f"Trainer handshake rejected: {reply.error}")
            self.replay_plan = reply.plan
            self.accepted = True
        request = BatchRequest(
            attempt_id=self.trainer_group.attempt_id,
            rank=self.rank,
            step=self.current_step,
        )
        self.socket.send(self.request_encoder.encode(request))
        self.socket.poll(timeout=None)

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the orchestrator."""
        frames = self.socket.recv_multipart(copy=False)
        if bytes(frames[0]) == b"error":
            raise RuntimeError(bytes(frames[1]).decode())
        if (
            len(frames) != 4
            or bytes(frames[0]) != b"batch"
            or bytes(frames[1]).decode() != self.trainer_group.attempt_id
            or bytes(frames[2]) != str(self.current_step).encode()
        ):
            raise RuntimeError("Batch response does not match trainer attempt and step")
        micro_batches: list[MicroBatch] = self.decoder.decode(frames[3])
        self.logger.debug(f"Received {len(micro_batches)} micro batches for step {self.current_step}")
        self.current_step += 1
        return micro_batches

    def close(self) -> None:
        try:
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch receiver")
