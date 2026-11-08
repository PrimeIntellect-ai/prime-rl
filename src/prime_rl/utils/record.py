import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import orjson

from prime_rl.utils.logger import get_logger

# Bounded wait for a record lock: a blocking flock(LOCK_EX) would stall the
# async event-loop thread for as long as the other writer holds the lock.
RECORD_LOCK_TIMEOUT = 10.0
RECORD_LOCK_POLL = 0.05


def read_record(path: Path) -> dict[str, Any] | None:
    return orjson.loads(path.read_bytes()) if path.is_file() else None


def write_record(path: Path, record: dict[str, Any]) -> None:
    """Atomic replace: a reader never sees a torn file. The tmp file is pid-suffixed,
    because several processes of one run write the same record and two writers
    sharing one tmp path can tear each other's staged file mid-write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(orjson.dumps(record, option=orjson.OPT_INDENT_2))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def record_lock(path: Path) -> Iterator[None]:
    """Cross-process read-modify-write lock for a record that several processes of
    one run merge into: POSIX flock on a sibling .lock file. The lock file is NEVER
    unlinked: a waiting writer can still hold the unlinked inode's lock while a new
    writer creates and locks a fresh file - two simultaneous "exclusive" locks and a
    lost merge; a stable lock inode is the whole guarantee. Acquisition is BOUNDED
    (`RECORD_LOCK_TIMEOUT`), retried with non-blocking flock so the async event-loop
    thread never blocks; `TimeoutError` past the bound. Storage errors that make
    locking impossible (e.g. ENOLCK) raise immediately instead of retrying."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+b") as lock_file:
        deadline = time.monotonic() + RECORD_LOCK_TIMEOUT
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"record lock at {lock_path} still held after {RECORD_LOCK_TIMEOUT}s")
                time.sleep(RECORD_LOCK_POLL)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def update_record(path: Path, merge: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Best-effort locked read-modify-write: run `merge` on the current record under
    the record lock and atomically replace the file. The records only back links
    (dashboard, launcher banner), so a lock or storage failure (TimeoutError, ENOLCK,
    ...) is logged and SKIPPED, never fatal to the run."""
    try:
        with record_lock(path):
            write_record(path, merge(read_record(path) or {}))
    except OSError:
        get_logger().warning(f"Failed to persist {path} (lock or storage error); the run's links may be missing")
