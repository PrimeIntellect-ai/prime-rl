"""Role-scoped synchronization over ModelExpress.

The policy session coordinates a complete weight update with the orchestrator.
A separate layer session uses the same READY/INITIALIZING handshake to keep a
bounded transfer arena live until every inference worker has acknowledged it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import grpc
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

Role = Literal["trainer", "inference", "orchestrator"]


@dataclass
class ModelExpressSession:
    client: MxClient
    role: Role
    rank: int
    session_id: str
    worker_id: str
    _source_id: str | None = field(init=False, default=None, repr=False)

    @property
    def source_id(self) -> str:
        if self._source_id is None:
            raise RuntimeError("ModelExpress metadata must be published before updating status")
        return self._source_id

    def identity(self, role: Role) -> p2p_pb2.SourceIdentity:
        return p2p_pb2.SourceIdentity(
            mx_version="0.3.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name="prime-rl-weights",
            extra_parameters={"role": role, "session_id": self.session_id},
        )

    def _rpc(self, call: Callable[[], Any]) -> Any:
        """Retry transient ModelExpress startup and transport failures."""
        deadline = time.monotonic() + 120.0
        while True:
            try:
                return call()
            except grpc.RpcError as exc:
                if (
                    exc.code()
                    not in (
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    )
                    or time.monotonic() >= deadline
                ):
                    raise
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))

    def publish(self, *, nixl_metadata: bytes = b"") -> str:
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.rank,
            nixl_metadata=nixl_metadata,
        )
        self._source_id = self._rpc(
            lambda: self.client.publish_metadata(self.identity(self.role), worker, self.worker_id)
        )
        return self._source_id

    def set_status(self, status: int) -> None:
        if not self._rpc(lambda: self.client.update_status(self.source_id, self.worker_id, self.rank, status)):
            raise RuntimeError(f"failed to set ModelExpress status {status} for {self.role} rank {self.rank}")

    def list_sources(self, role: Role, status: int | None) -> list[p2p_pb2.SourceInstanceRef]:
        response = self._rpc(lambda: self.client.list_sources(self.identity(role), status_filter=status))
        return list(response.instances)

    def exists_role_with_status(self, role: Role, status: int) -> bool:
        return bool(self.list_sources(role, status))

    def wait_for(
        self,
        role: Role,
        *,
        count: int,
        status: int | None,
        timeout: float,
        poll_interval: float = 0.05,
        cancelled: Callable[[], bool] | None = None,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        deadline = time.monotonic() + timeout
        expected_ranks = set(range(count))
        while True:
            if cancelled is not None and cancelled():
                raise RuntimeError(f"cancelled waiting for {count} {role} worker(s) in session {self.session_id!r}")
            refs = self.list_sources(role, status)
            by_rank = {ref.worker_rank: ref for ref in refs}
            if expected_ranks.issubset(by_rank):
                return [by_rank[rank] for rank in range(count)]
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {count} {role} worker(s) in session {self.session_id!r} "
                    f"with status={status}; found ranks={sorted(by_rank)}"
                )
            time.sleep(poll_interval)

    def fetch(self, ref: p2p_pb2.SourceInstanceRef) -> p2p_pb2.WorkerMetadata:
        response = self._rpc(lambda: self.client.get_metadata(ref.mx_source_id, ref.worker_id))
        if not response.found:
            raise LookupError(f"ModelExpress worker {ref.worker_id!r} disappeared")
        return response.worker
