"""Role-scoped synchronization over ModelExpress.

The protocol intentionally alternates ``INITIALIZING`` and ``READY``. The
trainer only publishes ``READY`` after every inference worker has entered the
current update (``INITIALIZING``), which prevents a stale READY from a prior
generation from being mistaken for an acknowledgement.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Literal, TypeVar

import grpc
from modelexpress import p2p_pb2
from modelexpress.client import MxClient

Role = Literal["trainer", "inference", "orchestrator"]
MX_MODEL_NAME = "prime-rl-weights"
_T = TypeVar("_T")
_RETRYABLE_RPC_CODES = frozenset((grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED))


@dataclass
class MxRendezvous:
    client: MxClient
    role: Role
    rank: int
    peer_world_size: int
    session_id: str
    worker_id: str = ""
    rpc_retry_timeout: float = 120.0
    rpc_retry_interval: float = 0.5

    def __post_init__(self) -> None:
        if not self.worker_id:
            self.worker_id = str(uuid.uuid4())
        self._mx_source_id: str | None = None

    @property
    def mx_source_id(self) -> str:
        if self._mx_source_id is None:
            raise RuntimeError("ModelExpress metadata must be published before updating status")
        return self._mx_source_id

    def identity(self, role: Role) -> p2p_pb2.SourceIdentity:
        return p2p_pb2.SourceIdentity(
            mx_version="0.3.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name=MX_MODEL_NAME,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            dtype="bfloat16",
            extra_parameters={"role": role, "session_id": self.session_id},
        )

    def _rpc(self, call: Callable[[], _T]) -> _T:
        """Retry transient ModelExpress startup and transport failures."""
        deadline = time.monotonic() + self.rpc_retry_timeout
        while True:
            try:
                return call()
            except grpc.RpcError as exc:
                if exc.code() not in _RETRYABLE_RPC_CODES or time.monotonic() >= deadline:
                    raise
                time.sleep(min(self.rpc_retry_interval, max(0.0, deadline - time.monotonic())))

    def publish(
        self,
        *,
        nixl_metadata: bytes = b"",
        tensors: Iterable[p2p_pb2.TensorDescriptor] = (),
    ) -> str:
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.rank,
            nixl_metadata=nixl_metadata,
            tensors=list(tensors),
        )
        self._mx_source_id = self._rpc(
            lambda: self.client.publish_metadata(self.identity(self.role), worker, self.worker_id)
        )
        return self._mx_source_id

    def set_status(self, status: int) -> None:
        if not self._rpc(lambda: self.client.update_status(self.mx_source_id, self.worker_id, self.rank, status)):
            raise RuntimeError(f"failed to set ModelExpress status {status} for {self.role} rank {self.rank}")

    def list(self, role: Role, status: int | None = None) -> list[p2p_pb2.SourceInstanceRef]:
        return list(self._rpc(lambda: self.client.list_sources(self.identity(role), status_filter=status)).instances)

    def has_status(self, role: Role, status: int) -> bool:
        return bool(self.list(role, status))

    def wait_for(
        self,
        role: Role,
        *,
        count: int,
        status: int | None = None,
        timeout: float = 1200,
        poll_interval: float = 0.05,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        deadline = time.monotonic() + timeout
        expected_ranks = set(range(count))
        while True:
            refs = self.list(role, status)
            by_rank = {ref.worker_rank: ref for ref in refs}
            if expected_ranks.issubset(by_rank):
                return [by_rank[rank] for rank in range(count)]
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out waiting for {count} {role} worker(s) in session {self.session_id!r} "
                    f"with status={status}; found ranks={sorted(by_rank)}"
                )
            time.sleep(poll_interval)

    def wait_for_peers(
        self, *, status: int | None = None, timeout: float = 1200, poll_interval: float = 0.05
    ) -> list[p2p_pb2.SourceInstanceRef]:
        peer_role: Role = "trainer" if self.role in ("inference", "orchestrator") else "inference"
        return self.wait_for(
            peer_role,
            count=self.peer_world_size,
            status=status,
            timeout=timeout,
            poll_interval=poll_interval,
        )

    def fetch(self, ref: p2p_pb2.SourceInstanceRef) -> p2p_pb2.WorkerMetadata:
        response = self._rpc(lambda: self.client.get_metadata(ref.mx_source_id, ref.worker_id))
        if not response.found:
            raise LookupError(f"ModelExpress worker {ref.worker_id!r} disappeared")
        return response.worker
