import asyncio
import time
from collections.abc import Awaitable, Callable
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
    PolicyState,
    RecoveryRequired,
    ReplayPlan,
    SessionOrigin,
    TrainerGroup,
    TrainerHandshake,
    TrainerHello,
    plan_replay,
)
from prime_rl.transports.batch.types import MicroBatch

# Liveness probes for an attempt awaiting admission. Receivers ignore PING frames.
PING = b"ping"
PROBE_INTERVAL_S = 1.0
# The ROUTER closes connections of peers that stop answering ZMTP heartbeats (e.g. a lost host),
# so a vanished trainer cannot block a replacement forever. Killed processes disconnect at once.
HEARTBEAT_IVL_MS = 5_000
HEARTBEAT_TIMEOUT_MS = 30_000


class ZMQBatchSender(BatchSender):
    def __init__(
        self,
        output_dir: Path,
        data_world_size: int,
        current_step: int,
        transport: ZMQTransportConfig,
        auto_recover: bool = False,
    ):
        """ZMQ micro batch sender. One ROUTER socket in the orchestrator answers the DEALER socket of
        every trainer rank. A background task serves handshakes, journal replays and requests;
        ``send`` hands it the live batch and waits until every rank of the accepted trainer
        attempt has it.

        With ``auto_recover``, a complete replacement attempt is admitted once every connection of
        the active attempt is gone. A supervisor must stop the old trainer group (including its
        checkpoint writers) before starting the replacement."""
        super().__init__(output_dir, data_world_size)
        self.context = zmq.asyncio.Context.instance()
        self.socket: zmq.asyncio.Socket = self.context.socket(zmq.ROUTER)
        # Replying to a rank that went away raises instead of dropping silently
        self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.socket.setsockopt(zmq.HEARTBEAT_IVL, HEARTBEAT_IVL_MS)
        self.socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, HEARTBEAT_TIMEOUT_MS)
        self.socket.bind(f"tcp://{transport.host}:{transport.port}")
        self.request_decoder = msgspec.msgpack.Decoder(type=TrainerHello | BatchRequest)
        self.handshake = TrainerHandshake(data_world_size)
        self.auto_recover = auto_recover
        self.on_replacement: Callable[[TrainerGroup], Awaitable[PolicyState]] | None = None
        """Reconciles the weight channel before a replacement is admitted, returning the policy
        inference serves. Runs on the sender's event loop while the old attempt is fenced."""

        # ``current_step`` is the producer's live batch; ``delivery_step`` is what the accepted
        # trainer attempt receives next, behind ``current_step`` while it replays the journal.
        self.current_step = current_step
        self.delivery_step = current_step
        self.live: list[bytes] | None = None
        self.live_delivered: asyncio.Future[float] | None = None
        self.requests: dict[int, tuple[bytes, BatchRequest]] = {}
        self.origin: SessionOrigin | None = None
        self.replay_plan: ReplayPlan | None = None
        self.replay_batches_sent = 0
        self.recovery_authorization: tuple[str, str] | None = None
        self.failed_attempt: str | None = None
        self.awaiting_admission: str | None = None
        self.delivery_lock = asyncio.Lock()
        self.serve_task: asyncio.Task | None = None
        self.logger.debug(f"Initialized ZMQ micro batch sender (endpoint=tcp://{transport.host}:{transport.port})")

    def start(self) -> None:
        """Serve trainer handshakes and requests from now on, also while no batch is pending."""
        if self.serve_task is None:
            self.serve_task = asyncio.create_task(self._serve(), name="batch-sender")

    async def send(self, micro_batch_grid: list[list[MicroBatch]]) -> float:
        """Send grid of micro batches to the trainers once every trainer rank requested the step.
        Encoding + journaling run in a worker thread before the wait."""
        assert len(micro_batch_grid) == self.data_world_size, "Number of micro batch lists must match data world size"
        for micro_batch_list in micro_batch_grid:
            assert len(micro_batch_list) == len(micro_batch_grid[0]), "All micro batch lists must have the same length"

        self.start()
        self.live = await asyncio.to_thread(self.encode_batch, micro_batch_grid, self.current_step)
        self.live_delivered = asyncio.get_running_loop().create_future()
        await self._deliver()
        await asyncio.wait({self.live_delivered, self.serve_task}, return_when=asyncio.FIRST_COMPLETED)
        if not self.live_delivered.done():
            self.serve_task.result()
            raise RuntimeError("Batch sender stopped serving trainer requests")
        return self.live_delivered.result()

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
        self.requests.clear()
        if self.handshake.candidate is not None and self.handshake.candidate.attempt_id != replacement_attempt:
            self.handshake.reject()
            self.awaiting_admission = None

    def close(self) -> None:
        try:
            if self.serve_task is not None:
                self.serve_task.cancel()
            self.socket.close(linger=0)
        finally:
            self.logger.debug("Closed ZMQ micro batch sender")

    async def _serve(self) -> None:
        while True:
            if not await self.socket.poll(PROBE_INTERVAL_S * 1000, zmq.POLLIN):
                if self.awaiting_admission is not None:
                    await self._admit()
                continue
            identity, payload = await self.socket.recv_multipart()
            message = self.request_decoder.decode(payload)
            if isinstance(message, TrainerHello):
                await self._hello(identity, message)
            else:
                await self._request(identity, message)

    async def _request(self, identity: bytes, request: BatchRequest) -> None:
        # Validate after any delivery in progress: a rank that already received its batch may request
        # the next step while the other ranks are still being sent theirs.
        async with self.delivery_lock:
            try:
                if request.attempt_id == self.failed_attempt:
                    raise ValueError("Trainer attempt is fenced pending recovery")
                self.handshake.validate_request(identity, request, self.delivery_step)
            except ValueError as error:
                await self._reply(identity, [b"error", str(error).encode()])
                return
            self.requests[request.rank] = (identity, request)
            await self._deliver_locked()

    async def _deliver(self) -> None:
        async with self.delivery_lock:
            await self._deliver_locked()

    async def _deliver_locked(self) -> None:
        """Hand the next step to the accepted attempt once all its ranks requested it: a journaled
        batch while it replays, then the live batch."""
        active = self.handshake.active
        if active is None or active.attempt_id == self.failed_attempt or len(self.requests) < active.world_size:
            return
        step = self.delivery_step
        replaying = step < self.current_step
        if replaying:
            payloads = await asyncio.to_thread(self._read_replay_batch, step)
        elif self.live is not None:
            payloads = self.live
        else:
            return
        send_start = time.perf_counter()
        for rank, (identity, request) in sorted(self.requests.items()):
            data_rank = self.handshake.members[rank][1].data_rank
            delivered = await self._reply(
                identity, [b"batch", request.attempt_id.encode(), str(step).encode(), payloads[data_rank]]
            )
            if not delivered:
                if self.journal is None:
                    raise ConnectionError("Trainer disconnected during batch delivery without a recovery journal")
                self.failed_attempt = active.attempt_id
                self.requests.clear()
                self.logger.warning(
                    f"Trainer attempt {active.attempt_id} disconnected during delivery of step {step}; "
                    "retaining the batch and waiting for a replacement"
                )
                return
        send_time = time.perf_counter() - send_start
        self.requests.clear()
        self.delivery_step += 1
        if replaying:
            self.replay_batches_sent += 1
            self.logger.info(f"Replayed batch {self.journal_session}:{step}")
            return
        self.current_step += 1
        self.live = None
        self.replay_plan = None
        self.live_delivered.set_result(send_time)

    def _read_replay_batch(self, step: int) -> list[bytes]:
        plan = self.replay_plan
        record = plan.batches[step - plan.start_step]
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

    async def _connected(self, connections: list[bytes]) -> list[bytes]:
        """Connections that still reach a peer."""
        alive = []
        for identity in connections:
            try:
                await self.socket.send_multipart([identity, PING], flags=zmq.NOBLOCK)
            except zmq.ZMQError as error:
                if error.errno == zmq.EHOSTUNREACH:
                    continue
                if error.errno != zmq.EAGAIN:
                    raise
            alive.append(identity)
        return alive

    async def _reply_hello(self, connections, reply: HandshakeReply) -> None:
        for connection in connections:
            await self._reply(connection, [b"hello", msgspec.msgpack.encode(reply)])

    async def _discard_if_disconnected(self, candidate: TrainerGroup) -> bool:
        """Reject the candidate attempt if one of its ranks disconnected: it can never take a step,
        e.g. it failed again before admission and its supervisor started another attempt."""
        pending = [connection for connection, _ in self.handshake.pending.values()]
        if len(await self._connected(pending)) == len(pending):
            return False
        error = "A rank of this trainer attempt disconnected before admission"
        self.logger.warning(f"Discarding trainer attempt {candidate.attempt_id}: a rank disconnected")
        self.handshake.reject()
        self.awaiting_admission = None
        if self.recovery_authorization is not None and self.recovery_authorization[1] == candidate.attempt_id:
            self.recovery_authorization = None
        await self._reply_hello(pending, HandshakeReply(candidate.attempt_id, "rejected", error=error))
        return True

    async def _hello(self, identity: bytes, hello: TrainerHello) -> None:
        candidate = self.handshake.candidate
        if candidate is not None and candidate.attempt_id != hello.group.attempt_id:
            await self._discard_if_disconnected(candidate)
        try:
            complete = self.handshake.add(identity, hello)
        except ValueError as error:
            reply = HandshakeReply(hello.group.attempt_id, "rejected", error=str(error))
            recipients = {identity}
            candidate = self.handshake.candidate
            if candidate is not None and candidate.attempt_id == hello.group.attempt_id:
                recipients.update(connection for connection, _ in self.handshake.pending.values())
                self.handshake.reject()
                self.awaiting_admission = None
            self.logger.warning(f"Rejected trainer attempt {hello.group.attempt_id}: {error}")
            await self._reply_hello(recipients, reply)
            return
        if complete:
            await self._admit()

    async def _admit(self) -> None:
        """Decide on the complete candidate attempt: the first one starts the session; a later one
        replaces the active attempt and replays the journal from its restored checkpoint. Holds
        the delivery lock so no delivery to the old attempt interleaves with the switch."""
        async with self.delivery_lock:
            await self._admit_candidate()

    async def _admit_candidate(self) -> None:
        group = self.handshake.candidate
        if group is None:
            self.awaiting_admission = None
            return
        if await self._discard_if_disconnected(group):
            return
        recipients = [connection for connection, _ in self.handshake.pending.values()]
        active = self.handshake.active
        if active is None:
            if group.start_step != self.current_step:
                error = (
                    f"Trainer starts at step {group.start_step}, but the orchestrator is on step {self.current_step}"
                )
                self.handshake.reject()
                self.logger.warning(f"Rejected trainer attempt {group.attempt_id}: {error}")
                await self._reply_hello(recipients, HandshakeReply(group.attempt_id, "rejected", error=error))
                return
            self.handshake.accept()
            self.origin = SessionOrigin(group.start_step, group.restored)
            restored = group.restored.batch.batch_id if group.restored is not None else None
            self.logger.info(
                f"Accepted trainer attempt {group.attempt_id}, next step {group.start_step}, restored={restored}"
            )
            await self._reply_hello(recipients, HandshakeReply(group.attempt_id, "accepted"))
            return

        authorized = self.recovery_authorization == (active.attempt_id, group.attempt_id)
        if not authorized and self.auto_recover:
            alive = await self._connected([connection for connection, _ in self.handshake.members.values()])
            if alive:
                if self.awaiting_admission != group.attempt_id:
                    self.logger.warning(
                        f"Trainer attempt {group.attempt_id} awaits admission: {len(alive)} rank(s) of active "
                        f"attempt {active.attempt_id} are still connected"
                    )
                self.awaiting_admission = group.attempt_id
                return
            authorized = True
        self.awaiting_admission = None
        try:
            if self.journal is None:
                raise ValueError("Trainer recovery requires batch journaling")
            plan = await asyncio.to_thread(
                plan_replay,
                self.journal,
                group,
                self.journal_session,
                self.origin,
                self.current_step - 1,
                self.data_world_size,
            )
        except (ValueError, OSError) as error:
            self.handshake.reject()
            if self.recovery_authorization is not None and self.recovery_authorization[1] == group.attempt_id:
                self.recovery_authorization = None
            self.logger.warning(f"Rejected trainer recovery attempt {group.attempt_id}: {error}")
            await self._reply_hello(recipients, HandshakeReply(group.attempt_id, "rejected", error=str(error)))
            return
        if not authorized:
            self.handshake.reject()
            self.logger.warning(
                f"Trainer attempt {group.attempt_id} restored step {group.start_step - 1}; "
                f"{len(plan.batches)} batches await replay before step {self.current_step}. "
                "Replacement requires explicit authorization."
            )
            await self._reply_hello(recipients, HandshakeReply(group.attempt_id, "recovery_required", plan=plan))
            return

        # Fence the old attempt before reconciling anything else with the replacement.
        self.failed_attempt = active.attempt_id
        self.requests.clear()
        self.logger.warning(
            f"Admitting replacement trainer attempt {group.attempt_id} for {active.attempt_id}: "
            f"replaying {len(plan.batches)} journaled batches from step {group.start_step} "
            f"before live step {self.current_step}"
        )
        policy = await self.on_replacement(group) if self.on_replacement is not None else None
        self.handshake.accept()
        self.delivery_step = group.start_step
        self.replay_plan = plan
        self.recovery_authorization = None
        self.failed_attempt = None
        await self._reply_hello(recipients, HandshakeReply(group.attempt_id, "accepted", plan=plan, policy=policy))


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
        self.reply: HandshakeReply | None = None
        self.replay_plan: ReplayPlan | None = None
        self.response: list | None = None
        self.logger.debug(f"Initialized ZMQ micro batch receiver (endpoint=tcp://{transport.host}:{transport.port})")

    def _recv(self, copy: bool = True) -> list:
        while True:
            frames = self.socket.recv_multipart(copy=copy)
            if bytes(frames[0]) != PING:
                return frames

    def connect(self) -> HandshakeReply:
        """Announce this rank's attempt and wait until the orchestrator admits it."""
        if self.reply is not None:
            return self.reply
        hello = TrainerHello(self.trainer_group, self.rank, self.data_rank)
        self.socket.send(self.request_encoder.encode(hello))
        frames = self._recv()
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
        self.reply = reply
        return reply

    def wait(self) -> None:
        self.connect()
        request = BatchRequest(
            attempt_id=self.trainer_group.attempt_id,
            rank=self.rank,
            step=self.current_step,
        )
        self.socket.send(self.request_encoder.encode(request))
        # Read the response here: a liveness ping must not end the wait early.
        self.response = self._recv(copy=False)

    def receive(self) -> list[MicroBatch]:
        """Receive a micro batch from the orchestrator."""
        if self.response is None:
            raise RuntimeError("receive() must follow wait()")
        frames, self.response = self.response, None
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
