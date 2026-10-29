import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from uuid import UUID

import msgspec


class BatchRecord(msgspec.Struct, frozen=True):
    batch_id: str
    hashes: list[str]


class BatchCheckpoint(msgspec.Struct, frozen=True):
    step: int
    batch: BatchRecord


def _position(batch_id: str) -> tuple[str, int]:
    session, step = batch_id.split(":")
    if UUID(session).hex != session or int(step) < 0 or str(int(step)) != step:
        raise ValueError(f"Invalid journal batch ID: {batch_id}")
    return session, int(step)


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())


def _write_atomic(path: Path, payload: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


class BatchJournal:
    """Exact transport payloads, retained until a trainer checkpoint covers them."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "batch_journal"

    def batch_path(self, batch_id: str) -> Path:
        session, step = _position(batch_id)
        return self.path / session / f"step_{step}"

    def write(self, batch_id: str, payloads: list[bytes]) -> BatchRecord:
        if not payloads:
            raise ValueError("Cannot journal an empty batch")
        destination = self.batch_path(batch_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _sync_dir(self.path.parent)
        _sync_dir(self.path)
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination.parent))
        try:
            hashes = []
            for rank, payload in enumerate(payloads):
                _write_file(temporary / f"rank_{rank}.bin", payload)
                hashes.append(hashlib.sha256(payload).hexdigest())
            record = BatchRecord(batch_id, hashes)
            _write_file(temporary / "manifest.json", msgspec.json.encode(record))
            _sync_dir(temporary)
            # A nonempty existing directory cannot be replaced: batch IDs are immutable.
            temporary.rename(destination)
            _sync_dir(destination.parent)
            return record
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def record(self, batch_id: str) -> BatchRecord:
        record = msgspec.json.decode((self.batch_path(batch_id) / "manifest.json").read_bytes(), type=BatchRecord)
        if record.batch_id != batch_id:
            raise ValueError(f"Journal batch identity mismatch: {batch_id}")
        return record

    def read(self, batch_id: str) -> list[bytes]:
        record = self.record(batch_id)
        payloads = []
        for rank, expected in enumerate(record.hashes):
            payload = (self.batch_path(batch_id) / f"rank_{rank}.bin").read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Journal payload checksum mismatch: {batch_id}, rank {rank}")
            payloads.append(payload)
        return payloads

    def commit_checkpoint(self, checkpoint_path: Path, step: int, batch_id: str) -> None:
        """Call only after every trainer rank has completed the checkpoint write."""
        if _position(batch_id)[1] != step:
            raise ValueError(f"Checkpoint step {step} does not match batch {batch_id}")
        record = self.record(batch_id)
        checkpoint = BatchCheckpoint(step, record)
        _write_atomic(checkpoint_path / "batch.json", msgspec.json.encode(checkpoint))
        _sync_dir(checkpoint_path.parent)
        _sync_dir(checkpoint_path.parent.parent)
        _sync_dir(checkpoint_path.parent.parent.parent)
        self.prune(checkpoint_path)

    def prune(self, checkpoint_path: Path) -> None:
        """Deletion is repeatable after interruption; other producer sessions are untouched."""
        checkpoint = msgspec.json.decode((checkpoint_path / "batch.json").read_bytes(), type=BatchCheckpoint)
        session, through = _position(checkpoint.batch.batch_id)
        if checkpoint.step != through:
            raise ValueError("Checkpoint and batch positions disagree")
        session_path = self.path / session
        if not session_path.exists():
            return
        # Persist the release boundary before deleting any payloads.
        _write_atomic(session_path / "checkpoint.json", msgspec.json.encode(checkpoint))
        for path in session_path.glob("step_*"):
            if int(path.name.removeprefix("step_")) <= through:
                shutil.rmtree(path)
        _sync_dir(session_path)
