"""Storage contract — the reversible seam between producer, store, and viewer.

The viewer reads through ``StorageBackend``; the sync writes through it. Swapping
LocalFS → HuggingFace → R2 is implementing this interface, not a rewrite. This is
what makes the "migrate to R2 before heavy scale" path a config change.

Store layout (a backend maps these logical paths to its native namespace):

  runs.json                                   registry: [{run_id, steps:[...], updated}]
  index/{run_id}/episodes.parquet             one row per episode (queryable; ~1 KB/ep)
  index/{run_id}/setup.json                   run setup: configured protocol, from the tomls
  transcripts/{run_id}/step-{step:05d}.jsonl.gz   full per-step messages, gzipped

The INDEX (parquet) holds identity + flat metrics + per-episode/per-member
diagnostic rollups + a transcript pointer (``transcript_shard`` + ``transcript_line``).
It is always small enough to cache locally and query with DuckDB. The TRANSCRIPTS
(gzipped jsonl, one shard per training step) are fetched on drill-down and cached.

``episodes.parquet`` row schema is the index contract (see ``INDEX_COLUMNS``).
Backends store opaque bytes; serialization (parquet/gzip) lives in the sync/index
layer, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

# Logical layout (backends translate to their native namespace).
RUNS_REGISTRY = "runs.json"


def index_path(run_id: str) -> str:
    return f"index/{run_id}/episodes.parquet"


def setup_path(run_id: str) -> str:
    """Per-run run-setup blob: the configured protocol, distilled from the tomls."""
    return f"index/{run_id}/setup.json"


def transcript_path(run_id: str, step: int) -> str:
    return f"transcripts/{run_id}/step-{step:05d}.jsonl.gz"


# Index contract: columns every ``episodes.parquet`` row must carry. The sync
# layer writes these; DuckDB queries them for list/filter/sort/compare. Diagnostic
# rollups are flattened with stable names so they are sortable/filterable
# (e.g. "diag.mismatch_kl.mean", "diag.debater_a.importance_ratio.p99").
INDEX_COLUMNS: tuple[str, ...] = (
    "run_id",
    "step",
    "example_id",
    "trajectory_id",
    "rollout_id",
    "kind",
    "members",  # json-encoded list[str]
    "reward",
    "advantage",
    "is_filtered",
    "error",  # null when clean
    "winner",  # from mar.categorical, null when not multi-agent
    "answers",  # json-encoded dict[seat, final-answer letter] from mar.categorical; null when absent
    "metrics",  # json-encoded dict[str, float] (flat per-episode metrics)
    "diagnostics",  # json-encoded dict (per-episode + per-member rollups, with status)
    "tokens",  # json-encoded dict (per-episode + per-member prompt/completion token totals)
    "transcript_shard",  # logical transcript path
    "transcript_line",  # 0-based line within the shard
)

# Fail-loud guard for the HF-free 100 GB private ceiling. The sync warns at SOFT
# and refuses to write past HARD rather than silently dropping (no-card tier).
SOFT_LIMIT_BYTES = 80 * 1024**3
HARD_LIMIT_BYTES = 95 * 1024**3


@dataclass(frozen=True)
class RunRef:
    run_id: str
    steps: list[int] = field(default_factory=list)
    updated: str | None = None  # ISO timestamp, stamped by the writer


@dataclass(frozen=True)
class StorageStat:
    total_bytes: int
    object_count: int


class StorageBackend(ABC):
    """Read/write logical paths as opaque bytes. Implementations: LocalFS, HF, R2."""

    @abstractmethod
    def list_runs(self) -> list[RunRef]:
        """Return the run registry (reads ``runs.json``)."""

    @abstractmethod
    def read(self, path: str) -> bytes:
        """Read a logical path. Raise ``FileNotFoundError`` if absent — never
        return empty bytes as a stand-in (that would mask a missing shard)."""

    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write a logical path. Must raise if ``stat().total_bytes`` would cross
        ``HARD_LIMIT_BYTES`` (no-card ceiling); callers warn at ``SOFT_LIMIT_BYTES``."""

    @abstractmethod
    def exists(self, path: str) -> bool: ...

    @abstractmethod
    def stat(self) -> StorageStat:
        """Total bytes + object count, for the fail-loud ceiling check."""
