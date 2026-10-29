"""Trainer identity, restored positions and journal validation."""

import hashlib
from typing import Literal
from uuid import UUID

import msgspec

from prime_rl.transports.batch.journal import BatchCheckpoint, BatchJournal, BatchRecord, _position


class TrainerGroup(msgspec.Struct, frozen=True):
    attempt_id: str
    world_size: int
    start_step: int
    restored: BatchCheckpoint | None = None


class TrainerHello(msgspec.Struct, tag="hello", frozen=True):
    group: TrainerGroup
    rank: int
    data_rank: int


class BatchRequest(msgspec.Struct, tag="batch", frozen=True):
    attempt_id: str
    rank: int
    step: int


class SessionOrigin(msgspec.Struct, frozen=True):
    """Where the producer session's first trainer group started. A replacement that restores
    the same position replays the session's journal from its first batch."""

    start_step: int
    restored: BatchCheckpoint | None = None


class ReplayPlan(msgspec.Struct, frozen=True):
    start_step: int
    restored: BatchCheckpoint | None
    batches: list[BatchRecord]


class PolicyState(msgspec.Struct, frozen=True):
    """Which policy a replacement trainer publishes first.

    ``version`` is the newer of the policy inference last applied completely and the
    replacement's restored checkpoint; policy versions never move backwards. Without
    ``reload``, inference serves that version and the replacement publishes only later ones.
    With ``reload``, inference holds no complete policy (an interrupted in-place update) or an
    older one, and the replacement also publishes ``version`` itself once it recomputed it."""

    version: int
    reload: bool

    def publishes(self, step: int) -> bool:
        """Whether the replacement publishes the policy it holds after ``step``."""
        return step > self.version or (self.reload and step == self.version)


class HandshakeReply(msgspec.Struct):
    attempt_id: str
    status: Literal["accepted", "recovery_required", "rejected"]
    plan: ReplayPlan | None = None
    policy: PolicyState | None = None
    error: str | None = None


class RecoveryRequired(RuntimeError):
    def __init__(self, plan: ReplayPlan):
        self.plan = plan
        super().__init__(
            f"Trainer recovery requires replay of {len(plan.batches)} journaled batches "
            f"from step {plan.start_step}; replacement is not authorized"
        )


def validate_checkpoint(checkpoint: BatchCheckpoint) -> None:
    _, step = _position(checkpoint.batch.batch_id)
    if step != checkpoint.step:
        raise ValueError("Checkpoint step and batch identity disagree")
    if not checkpoint.batch.hashes or any(
        len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) for digest in checkpoint.batch.hashes
    ):
        raise ValueError("Checkpoint has invalid payload hashes")


def plan_replay(
    journal: BatchJournal,
    group: TrainerGroup,
    session: str,
    origin: SessionOrigin | None,
    through: int,
    data_world_size: int,
) -> ReplayPlan:
    """Validate every retained payload without loading whole batches into memory.

    ``through`` is the last batch the producer handed to the transport before the batch
    it is collecting now; the replacement then receives that live batch normally. A
    checkpoint from this producer session must match the journal's committed record; any
    other restored position must be the session origin. Entries from other producer
    sessions are never joined implicitly.
    """
    checkpoint = group.restored
    restored_session = _position(checkpoint.batch.batch_id)[0] if checkpoint is not None else None
    if restored_session == session:
        validate_checkpoint(checkpoint)
        if group.start_step != checkpoint.step + 1:
            raise ValueError("Restored position does not follow the restored checkpoint")
        if len(checkpoint.batch.hashes) != data_world_size:
            raise ValueError("Restored checkpoint has a different data world size")
        marker = journal.path / session / "checkpoint.json"
        if marker.exists():
            committed = msgspec.json.decode(marker.read_bytes(), type=BatchCheckpoint)
            validate_checkpoint(committed)
            if committed != checkpoint:
                raise ValueError("Restored checkpoint does not match the journal's committed checkpoint")
        elif journal.record(checkpoint.batch.batch_id) != checkpoint.batch:
            raise ValueError("Restored checkpoint hashes do not match the journal")
    elif origin is None or SessionOrigin(group.start_step, checkpoint) != origin:
        raise ValueError(
            "Restored checkpoint is neither committed in this producer session nor the position it started from"
        )
    if group.start_step > through + 1:
        raise ValueError("Restored position is ahead of the batches the producer has delivered")
    records = []
    for step in range(group.start_step, through + 1):
        record = journal.record(f"{session}:{step}")
        if len(record.hashes) != data_world_size:
            raise ValueError(f"Journal data world size mismatch: {record.batch_id}")
        for rank, expected in enumerate(record.hashes):
            with (journal.batch_path(record.batch_id) / f"rank_{rank}.bin").open("rb") as file:
                if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                    raise ValueError(f"Journal payload checksum mismatch: {record.batch_id}, rank {rank}")
        records.append(record)
    return ReplayPlan(group.start_step, checkpoint, records)


class TrainerHandshake:
    """Collect distinct ranks; only the caller can accept a complete candidate."""

    def __init__(self, data_world_size: int):
        self.data_world_size = data_world_size
        self.active: TrainerGroup | None = None
        self.members: dict[int, tuple[bytes, TrainerHello]] = {}
        self.candidate: TrainerGroup | None = None
        self.pending: dict[int, tuple[bytes, TrainerHello]] = {}
        self.retired: set[str] = set()

    def add(self, identity: bytes, hello: TrainerHello) -> bool:
        group = hello.group
        if UUID(group.attempt_id).hex != group.attempt_id:
            raise ValueError("Invalid trainer attempt ID")
        if group.attempt_id in self.retired or (self.active and group.attempt_id == self.active.attempt_id):
            raise ValueError("Trainer attempt already accepted or retired")
        if group.world_size < self.data_world_size or group.world_size % self.data_world_size:
            raise ValueError("Trainer world size must be divisible by data world size")
        if self.active and group.world_size != self.active.world_size:
            raise ValueError("Trainer recovery cannot change world size")
        if not 0 <= hello.rank < group.world_size:
            raise ValueError("Invalid trainer rank")
        if hello.data_rank != hello.rank // (group.world_size // self.data_world_size):
            raise ValueError("Invalid trainer rank to data rank mapping")
        if group.start_step < 0:
            raise ValueError("Invalid starting step")
        if group.restored is not None:
            validate_checkpoint(group.restored)
            if group.start_step != group.restored.step + 1:
                raise ValueError("Starting step does not follow restored checkpoint")
        if self.candidate is not None and group != self.candidate:
            raise ValueError("Trainer ranks disagree on attempt, world size or restored position")
        if hello.rank in self.pending and self.pending[hello.rank] != (identity, hello):
            raise ValueError("Duplicate trainer rank on a different connection")
        if any(connection == identity and rank != hello.rank for rank, (connection, _) in self.pending.items()):
            raise ValueError("Connection already belongs to another trainer rank")
        self.candidate = group
        self.pending[hello.rank] = (identity, hello)
        return len(self.pending) == group.world_size

    def accept(self) -> None:
        if self.candidate is None or len(self.pending) != self.candidate.world_size:
            raise ValueError("Cannot accept an incomplete trainer group")
        if self.active is not None:
            self.retired.add(self.active.attempt_id)
        self.active, self.members = self.candidate, self.pending
        self.candidate, self.pending = None, {}

    def reject(self) -> None:
        if self.candidate is not None:
            self.retired.add(self.candidate.attempt_id)
        self.candidate, self.pending = None, {}

    def validate_request(self, identity: bytes, request: BatchRequest, step: int) -> int:
        if self.active is None or request.attempt_id != self.active.attempt_id:
            raise ValueError("Request is not from the accepted trainer attempt")
        member = self.members.get(request.rank)
        if member is None or member[0] != identity:
            raise ValueError("Request does not match the accepted rank connection")
        if request.step != step:
            raise ValueError(f"Trainer requested step {request.step}, but the orchestrator is on step {step}")
        return member[1].data_rank
