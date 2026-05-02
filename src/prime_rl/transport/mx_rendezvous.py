"""Per-rank rendezvous client over Model Express.

Each worker in a prime-rl run (trainer rank or inference vLLM worker)
constructs one :class:`MxRendezvous`, publishes its NIXL agent metadata
plus tensor descriptors, then blocks until the counterpart role is fully
visible. The class is intentionally thin: it owns identity construction
(role baked into ``SourceIdentity.extra_parameters`` so trainer/inference
hash to different ``mx_source_id``s) and the polling loop, and delegates
all gRPC to ``modelexpress.MxClient``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Literal

from modelexpress import p2p_pb2
from modelexpress.client import MxClient

Role = Literal["trainer", "inference"]


@dataclass
class MxRendezvous:
    """One rendezvous session per (role, rank).

    Attributes:
        client: A connected :class:`modelexpress.client.MxClient`.
        role: ``"trainer"`` or ``"inference"``. Recorded in
            ``SourceIdentity.extra_parameters["role"]`` so the two roles
            hash to different ``mx_source_id``s on the server.
        rank: This worker's rank within its role.
        peer_world_size: Number of workers expected on the counterpart
            role. :meth:`wait_for_peers` blocks until at least this many
            are visible.
        model_name: Inference model identifier (e.g.,
            ``"Qwen/Qwen3-235B-A22B-Thinking-2507-FP8"``).
        worker_id: Unique handle for this worker, defaulting to a fresh
            UUID. Two ranks must NOT share a ``worker_id``.
    """

    client: MxClient
    role: Role
    rank: int
    peer_world_size: int
    model_name: str
    worker_id: str = ""

    def __post_init__(self) -> None:
        if not self.worker_id:
            self.worker_id = str(uuid.uuid4())
        self._mx_source_id: str | None = None

    @property
    def peer_role(self) -> Role:
        return "inference" if self.role == "trainer" else "trainer"

    @property
    def mx_source_id(self) -> str:
        """The mx_source_id assigned by the server. Set after :meth:`publish`."""
        if self._mx_source_id is None:
            raise RuntimeError("publish() must be called before mx_source_id is available")
        return self._mx_source_id

    def _identity(self, role: Role) -> p2p_pb2.SourceIdentity:
        return p2p_pb2.SourceIdentity(
            mx_version="0.3.0",
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            model_name=self.model_name,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            dtype="bfloat16",
            extra_parameters={"role": role},
        )

    def publish(
        self,
        *,
        nixl_metadata: bytes,
        tensors: Iterable[p2p_pb2.TensorDescriptor],
    ) -> str:
        """Publish this worker's metadata. Returns the assigned ``mx_source_id``."""
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.rank,
            nixl_metadata=nixl_metadata,
            tensors=list(tensors),
        )
        self._mx_source_id = self.client.publish_metadata(self._identity(self.role), worker, self.worker_id)
        return self._mx_source_id

    def wait_for_peers(
        self,
        *,
        status: int | None = None,
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """Block until ``peer_world_size`` peers of the counterpart role are visible.

        Args:
            status: If set, only count peers in this :class:`p2p_pb2.SourceStatus`.
            timeout: Wall-clock seconds to wait before raising :class:`TimeoutError`.
            poll_interval: Seconds between ``ListSources`` polls.
        """
        deadline = time.monotonic() + timeout
        peer_id = self._identity(self.peer_role)
        while True:
            resp = self.client.list_sources(peer_id, status_filter=status)
            if len(resp.instances) >= self.peer_world_size:
                return list(resp.instances)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {self.peer_world_size} "
                    f"{self.peer_role!r} peers (saw {len(resp.instances)})"
                )
            time.sleep(poll_interval)

    def fetch_peer(self, ref: p2p_pb2.SourceInstanceRef) -> p2p_pb2.WorkerMetadata:
        """Fetch full :class:`WorkerMetadata` for one peer ref returned by
        :meth:`wait_for_peers`.
        """
        resp = self.client.get_metadata(ref.mx_source_id, ref.worker_id)
        if not resp.found:
            raise LookupError(f"peer worker {ref.worker_id!r} not found at {ref.mx_source_id}")
        return resp.worker

    def set_status(self, status: int) -> None:
        """Update this worker's lifecycle status. Requires :meth:`publish` first."""
        self.client.update_status(self.mx_source_id, self.worker_id, self.rank, status)
