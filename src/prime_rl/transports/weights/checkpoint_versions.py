"""What a checkpoint published to ModelExpress carries beyond its tensors.

ModelExpress versions hold floating-point tensors only. A checkpoint's integer tensors (DeepSeek-V4's hash-routing
tables) stay out of the version, and an engine that installs it keeps the tables it loaded at startup. That is right
only while both hold the same tables, so the publisher puts a digest of them into the version's uid and the engine
compares it with its own before it installs the version.

The digest covers the tables' values and shapes, not their names or integer widths: the engine names and stores them
its own way (vLLM keeps DeepSeek-V4's ``tid2eid`` as int32 under its module names; the checkpoint holds int64).
"""

import hashlib
import uuid
from collections.abc import Iterable

import torch

INTEGER_TABLES_MARK = ".int-"
"""Separates a checkpoint version's own id from the digest of its integer tables in its uid."""


def integer_tables_digest(tables: Iterable[torch.Tensor]) -> str:
    """Digest of integer tables by value and shape, independent of their order, names and integer widths."""
    digests = sorted(
        hashlib.sha256(
            repr(tuple(table.shape)).encode() + table.detach().to("cpu", torch.int64).contiguous().numpy().tobytes()
        ).hexdigest()
        for table in tables
    )
    return hashlib.sha256(",".join(digests).encode()).hexdigest()[:16]


def checkpoint_version_uid(step: int, integer_tables: str) -> str:
    """A fresh uid for a checkpoint version of ``step`` whose integer tables have digest ``integer_tables``."""
    return f"ckpt-{step}-{uuid.uuid4().hex[:12]}{INTEGER_TABLES_MARK}{integer_tables}"


def version_integer_tables(uid: str) -> str | None:
    """The integer tables digest a version uid carries, None for versions without one (a trainer's)."""
    if not uid.startswith("ckpt-") or INTEGER_TABLES_MARK not in uid:
        return None
    return uid.rsplit(INTEGER_TABLES_MARK, 1)[1]
