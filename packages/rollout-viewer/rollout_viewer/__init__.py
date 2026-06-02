"""rollout-viewer — online rollout visualization for prime-rl.

Phase 0 surface (the contracts). Implementations land in later workstreams:
  schema.py   Episode / Step / diagnostics (the normalized model)  [done]
  storage.py  StorageBackend + store layout                        [done]
  contracts.md  diagnostics sidecar + watertight join rules        [done]
  parser.py   raw jsonl ⊕ diagnostics → Episode (W2)
  sync.py     strip/gzip/parquet/retention → store (W1)
  visibility.py  per-seat prompt-delta verification (W3)
  index.py / app.py  DuckDB index + FastAPI + SSE (W4)
  web/        the SPA (W5)
"""

from rollout_viewer.schema import (
    Episode,
    EpisodeKind,
    MemberOutcome,
    Message,
    MultiAgentScore,
    Step,
    StepDiagnostics,
    ValueSummary,
    detect_kind,
)
from rollout_viewer.storage import (
    HARD_LIMIT_BYTES,
    INDEX_COLUMNS,
    SOFT_LIMIT_BYTES,
    RunRef,
    StorageBackend,
    StorageStat,
    index_path,
    transcript_path,
)

__all__ = [
    "Episode",
    "EpisodeKind",
    "Message",
    "MemberOutcome",
    "MultiAgentScore",
    "Step",
    "StepDiagnostics",
    "ValueSummary",
    "detect_kind",
    "StorageBackend",
    "StorageStat",
    "RunRef",
    "INDEX_COLUMNS",
    "SOFT_LIMIT_BYTES",
    "HARD_LIMIT_BYTES",
    "index_path",
    "transcript_path",
]
