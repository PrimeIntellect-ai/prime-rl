"""Per-rank rendezvous client over Model Express.

Model Express is the metadata store for the NIXL weight broadcast: each
participant (trainer rank, inference vLLM worker, orchestrator) constructs
one :class:`MxRendezvous`, publishes its payload (NIXL agent metadata,
baked-copy manifest), and polls for the counterpart role. The class is
intentionally thin: it owns identity construction (role baked into
``SourceIdentity.extra_parameters`` so roles hash to different
``mx_source_id``\\ s) and the polling loops, and delegates all gRPC to
``modelexpress.client.MxClient``.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Iterable, Literal

from modelexpress import p2p_pb2
from modelexpress.client import MxClient

Role = Literal["trainer", "inference", "orchestrator"]

# All participants publish under the same fixed model name: the MX server is
# launched per SLURM job, so there is exactly one rendezvous group per server
# and identities only need to differ by role and worker_id.
MX_MODEL_NAME = "prime-rl-weights"

logger = logging.getLogger("prime_rl.weight_transfer.mx")


@dataclass
class MxRendezvous:
    """One rendezvous session per (role, rank).

    Attributes:
        client: A connected :class:`modelexpress.client.MxClient`.
        role: Recorded in ``SourceIdentity.extra_parameters["role"]`` so the
            roles hash to different ``mx_source_id``s on the server.
        rank: This worker's rank within its role.
        peer_world_size: Number of workers expected on the counterpart role.
            :meth:`wait_for_peers` blocks until at least this many are visible.
        model_name: Inference model identifier (e.g. ``"Qwen/Qwen3-30B-A3B"``).
        worker_id: Unique handle for this worker, defaulting to a fresh UUID.
            Two ranks must NOT share a ``worker_id``.
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
        return "trainer" if self.role in ("inference", "orchestrator") else "inference"

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
        nixl_metadata: bytes = b"",
        tensors: Iterable[p2p_pb2.TensorDescriptor] = (),
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
        logged = False
        while True:
            resp = self.client.list_sources(peer_id, status_filter=status)
            if not logged:
                logger.info(
                    f"wait_for_peers: role={self.peer_role} need={self.peer_world_size} "
                    f"found_with_status={len(resp.instances)} status_filter={status} model={peer_id.model_name}"
                )
                logged = True
            if len(resp.instances) >= self.peer_world_size:
                return list(resp.instances)
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {self.peer_world_size} "
                    f"{self.peer_role!r} peers (saw {len(resp.instances)})"
                )
            time.sleep(poll_interval)

    def fetch_peer(self, ref: p2p_pb2.SourceInstanceRef) -> p2p_pb2.WorkerMetadata:
        """Fetch the full :class:`WorkerMetadata` for one peer ref."""
        resp = self.client.get_metadata(ref.mx_source_id, ref.worker_id)
        if not resp.found:
            raise LookupError(f"peer worker {ref.worker_id!r} not found at {ref.mx_source_id}")
        return resp.worker
