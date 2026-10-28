import asyncio
import time
from pathlib import Path

import msgspec
import zmq
import zmq.asyncio

from prime_rl.configs.shared import ZMQTransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.recovery import (
    BatchRequest,
    HandshakeReply,
    RecoveryRequired,
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
        self.logger.debug(f"Initialized ZMQ micro batch sender (endpoint=tcp://{transport.host}:{transport.port})")

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers. Encoding happens before the wait, so only the
        hand-off sits between the last request and the batch."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        buffers = await asyncio.to_thread(self.encode_batch, micro_batch_grid, self.current_step)
        identity_to_request = await self._wait_for_requests()
        send_start = time.perf_counter()
        for identity, request in identity_to_request.items():
            data_rank = self.handshake.validate_request(identity, request, self.current_step)
            await self.socket.send_multipart(
                [identity, b"batch", request.attempt_id.encode(), str(self.current_step).encode(), buffers[data_rank]],
                copy=False,
            )
        send_end = time.perf_counter()

        self.logger.debug(f"Sent micro batch grid for step {self.current_step} to {len(identity_to_request)} ranks")
        self.current_step += 1
        return send_end - send_start

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
                await self.socket.send_multipart([connection, b"hello", msgspec.msgpack.encode(reply)])
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
                reply = HandshakeReply(group.attempt_id, "recovery_required", plan=plan)
                self.logger.warning(
                    f"Trainer attempt {group.attempt_id} restored {plan.restored.batch.batch_id}; "
                    f"{len(plan.batches)} batches await replay through step {self.current_step}. "
                    "Replay execution is not enabled; the attempt was not accepted."
                )
            except (ValueError, OSError) as error:
                reply = HandshakeReply(group.attempt_id, "rejected", error=str(error))
                self.logger.warning(f"Rejected trainer recovery attempt {group.attempt_id}: {error}")
            self.handshake.reject()
        for connection in recipients:
            await self.socket.send_multipart([connection, b"hello", msgspec.msgpack.encode(reply)])

    async def _wait_for_requests(self) -> dict[bytes, BatchRequest]:
        """Only distinct ranks of the accepted group can request the pending live batch."""
        identity_to_request = {}
        while self.handshake.active is None or len(identity_to_request) < self.handshake.active.world_size:
            identity, payload = await self.socket.recv_multipart()
            message = self.request_decoder.decode(payload)
            if isinstance(message, TrainerHello):
                await self._hello(identity, message)
                continue
            try:
                self.handshake.validate_request(identity, message, self.current_step)
            except ValueError as error:
                await self.socket.send_multipart([identity, b"error", str(error).encode()])
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
