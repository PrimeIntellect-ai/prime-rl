"""Concrete ``StorageBackend`` implementations — LocalFS and HuggingFace.

A backend maps the logical layout (``storage.py``) onto a native namespace and
reads/writes opaque bytes. Serialization (parquet/gzip) lives in ``index.py`` /
``sync.py``; here we only move bytes, enforce the fail-loud ceiling, and answer
``list_runs`` from ``runs.json``.

``LocalFSBackend`` is a directory tree. ``HuggingFaceBackend`` treats a PRIVATE
dataset repo as dumb blob storage: it BATCHES writes into a single commit
(``flush``) rather than one commit per step, dodging HF's commit-rate and
file-count limits (see ``contracts.md`` → "dumb parquet/blob").
"""

from __future__ import annotations

import json
from pathlib import Path

from rollout_viewer.storage import (
    HARD_LIMIT_BYTES,
    RUNS_REGISTRY,
    RunRef,
    StorageBackend,
    StorageStat,
)


def _parse_runs(data: bytes) -> list[RunRef]:
    """Decode ``runs.json`` bytes into ``RunRef`` s. Raises on malformed JSON."""
    records = json.loads(data.decode("utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"runs.json must be a JSON list, got {type(records).__name__}")
    runs: list[RunRef] = []
    for r in records:
        runs.append(
            RunRef(
                run_id=r["run_id"],
                steps=list(r.get("steps", [])),
                updated=r.get("updated"),
            )
        )
    return runs


class LocalFSBackend(StorageBackend):
    """Logical paths rooted at a local directory. Used for dev + HF-Space disk."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Seed running totals with ONE scan; ``write`` keeps them current so
        # ``stat`` is O(1) instead of an rglob per call.
        self._total_bytes = 0
        self._object_count = 0
        for f in self.root.rglob("*"):
            if f.is_file():
                self._total_bytes += f.stat().st_size
                self._object_count += 1

    def _full(self, path: str) -> Path:
        return self.root / path

    def list_runs(self) -> list[RunRef]:
        p = self._full(RUNS_REGISTRY)
        if not p.exists():
            return []
        return _parse_runs(p.read_bytes())

    def read(self, path: str) -> bytes:
        p = self._full(path)
        if not p.exists():
            raise FileNotFoundError(f"no object at logical path {path!r} (root={self.root})")
        return p.read_bytes()

    def write(self, path: str, data: bytes) -> None:
        p = self._full(path)
        old_size = p.stat().st_size if p.exists() else None
        projected = self._total_bytes - (old_size or 0) + len(data)
        if projected > HARD_LIMIT_BYTES:
            raise RuntimeError(
                f"write of {path!r} ({len(data)} bytes) would put the store at "
                f"{projected} bytes, past HARD_LIMIT_BYTES={HARD_LIMIT_BYTES}. "
                "Refusing to write — migrate to R2 (see contracts.md)."
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace so a concurrent reader never sees a half-written object.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
        # Keep running totals current: a replace swaps bytes (count unchanged), a
        # fresh write adds one object.
        self._total_bytes += len(data) - (old_size or 0)
        if old_size is None:
            self._object_count += 1

    def exists(self, path: str) -> bool:
        return self._full(path).exists()

    def stat(self) -> StorageStat:
        return StorageStat(total_bytes=self._total_bytes, object_count=self._object_count)


class HuggingFaceBackend(StorageBackend):
    """A PRIVATE HF dataset repo as dumb blob storage with BATCHED commits.

    Writes accumulate in an in-memory buffer and are committed together by
    ``flush`` (called once per sync at the end), so a multi-step sync produces a
    single commit instead of one-per-step. ``read``/``exists``/``stat`` see the
    pending buffer overlaid on the remote, so a sync can read what it just wrote
    before flushing.
    """

    def __init__(self, repo_id: str, *, token: str | None = None, revision: str = "main") -> None:
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.revision = revision
        self.api = HfApi(token=token)
        # Ensure the (private) dataset repo exists before any operation.
        self.api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True, token=token)
        self._pending: dict[str, bytes] = {}
        self._remote_files: set[str] | None = None

    def _remote_listing(self) -> set[str]:
        if self._remote_files is None:
            self._remote_files = set(
                self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset", revision=self.revision)
            )
        return self._remote_files

    def list_runs(self) -> list[RunRef]:
        if not self.exists(RUNS_REGISTRY):
            return []
        return _parse_runs(self.read(RUNS_REGISTRY))

    def read(self, path: str) -> bytes:
        if path in self._pending:
            return self._pending[path]
        if path not in self._remote_listing():
            raise FileNotFoundError(f"no object at logical path {path!r} in {self.repo_id}")
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=path,
            revision=self.revision,
            token=self.api.token,
        )
        return Path(local).read_bytes()

    def write(self, path: str, data: bytes) -> None:
        projected = self.stat().total_bytes
        # Replacing an already-pending object swaps its bytes, not adds.
        if path in self._pending:
            projected -= len(self._pending[path])
        projected += len(data)
        if projected > HARD_LIMIT_BYTES:
            raise RuntimeError(
                f"write of {path!r} ({len(data)} bytes) would put the store at "
                f"{projected} bytes, past HARD_LIMIT_BYTES={HARD_LIMIT_BYTES}. "
                "Refusing to write — migrate to R2 (see contracts.md)."
            )
        self._pending[path] = data

    def exists(self, path: str) -> bool:
        return path in self._pending or path in self._remote_listing()

    def stat(self) -> StorageStat:
        # Remote object sizes + pending buffer. Pending paths that already exist
        # remotely are counted once (the new bytes win).
        from huggingface_hub import HfApi  # noqa: F401  (api already constructed)

        infos = self.api.list_repo_tree(
            repo_id=self.repo_id, repo_type="dataset", revision=self.revision, recursive=True
        )
        remote_sizes: dict[str, int] = {}
        for entry in infos:
            size = getattr(entry, "size", None)
            if size is not None:
                remote_sizes[entry.path] = size
        for path, data in self._pending.items():
            remote_sizes[path] = len(data)
        return StorageStat(total_bytes=sum(remote_sizes.values()), object_count=len(remote_sizes))

    def flush(self, commit_message: str = "rollout-viewer sync") -> None:
        """Commit all buffered writes as ONE commit, then clear the buffer."""
        if not self._pending:
            return
        from huggingface_hub import CommitOperationAdd

        ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=data) for path, data in self._pending.items()]
        self.api.create_commit(
            repo_id=self.repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message=commit_message,
            revision=self.revision,
        )
        # Newly committed paths are now remote.
        if self._remote_files is not None:
            self._remote_files |= set(self._pending.keys())
        self._pending.clear()
