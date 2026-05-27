"""Per-rank rendezvous client over Model Express.

Each worker in a prime-rl run (trainer rank or inference vLLM worker)
constructs one :class:`MxRendezvous`, publishes its NIXL agent metadata
plus tensor descriptors, then blocks until the counterpart role is fully
visible. The class is intentionally thin: it owns identity construction
(role baked into ``SourceIdentity.extra_parameters`` so trainer/inference
hash to different ``mx_source_id``s) and the polling loop, and delegates
all gRPC to ``modelexpress.MxClient``.

Phase-2 fixes (post-#2389) baked in:

- **Heartbeat**: spawning :class:`MxRendezvous` starts a background
  :class:`HeartbeatThread` on ``publish()`` so the MX server's reaper can
  detect crashed workers and mark them ``STALE``. Crashed workers were
  leaving permanent ``READY`` rows that broke restarts on GB200.
- **Freshest-per-(role, rank) dedup**: when multiple entries for the same
  (role, rank) live in the catalog (e.g. after a partial pod restart),
  callers see only the most recently updated one. This is the second of
  the two GB200 runtime patches.
- **Same-rank-only filter**: optional ``same_rank_only=True`` on the wait
  methods restricts results to peers with ``worker_rank == self.rank``,
  closing the cross-subnet full-mesh path that fails on GCP multi-NIC
  RDMA fabrics. Off by default; the caller opts in.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterable, Literal

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.metadata.heartbeat import HeartbeatThread

Role = Literal["trainer", "inference", "orchestrator"]

_log = logging.getLogger("prime_rl.transport.mx_rendezvous")


def _freshest_per_rank(
    instances: Iterable[p2p_pb2.SourceInstanceRef],
    *,
    metas: dict[str, int],
) -> list[p2p_pb2.SourceInstanceRef]:
    """Dedup peers by ``worker_rank``, keeping the one with the largest
    ``updated_at`` from ``metas``.

    ``metas`` maps ``worker_id`` → ``updated_at`` (ms-epoch as reported by
    the MX server). Instances whose ``worker_id`` is missing from ``metas``
    are kept (we err on the side of "visible but not freshness-known").

    This is the Phase-2 codification of the runtime patch we applied on
    GB200: the prime-rl trainer's NIXL agent rotated ``mx_source_id`` on
    restart, leaving a stale ``READY`` entry at the same ``worker_rank``;
    receivers picked the stale one and got ``NIXL_ERR_NOT_ALLOWED`` when
    they tried to ``add_remote_agent``.
    """
    by_rank: dict[int, tuple[int, p2p_pb2.SourceInstanceRef]] = {}
    for inst in instances:
        ts = metas.get(inst.worker_id, 0)
        cur = by_rank.get(inst.worker_rank)
        if cur is None or ts > cur[0]:
            by_rank[inst.worker_rank] = (ts, inst)
    return [v[1] for _, v in sorted(by_rank.items())]


def _filter_same_rank(
    instances: Iterable[p2p_pb2.SourceInstanceRef], *, rank: int
) -> list[p2p_pb2.SourceInstanceRef]:
    """Keep only peers whose ``worker_rank == rank``.

    The cross-subnet full-mesh routing path failed on GCP GB200's multi-NIC
    fabric — each rank has its own IB subnet, so trainer rank N can only
    safely peer with inference rank N. Filtering at the rendezvous layer
    prevents the broken connections from ever being attempted.
    """
    return [inst for inst in instances if inst.worker_rank == rank]


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
    enable_heartbeat: bool = True

    def __post_init__(self) -> None:
        if not self.worker_id:
            self.worker_id = str(uuid.uuid4())
        self._mx_source_id: str | None = None
        self._heartbeat: HeartbeatThread | None = None

    @property
    def peer_role(self) -> Role:
        if self.role == "trainer":
            return "inference"
        if self.role == "orchestrator":
            return "trainer"
        return "trainer"

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
        nixl_metadata: bytes = b"",
        tensors: Iterable[p2p_pb2.TensorDescriptor] = (),
    ) -> str:
        """Publish this worker's metadata. Returns the assigned ``mx_source_id``.

        Side effect (Phase 2): if ``enable_heartbeat`` is True, a
        :class:`HeartbeatThread` is started after a successful publish so
        the MX server's reaper can detect liveness. Heartbeat is idempotent
        — calling ``publish()`` again on the same instance is a no-op for
        the heartbeat (the existing thread keeps running).
        """
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=self.rank,
            nixl_metadata=nixl_metadata,
            tensors=list(tensors),
        )
        self._mx_source_id = self.client.publish_metadata(self._identity(self.role), worker, self.worker_id)

        if self.enable_heartbeat and self._heartbeat is None:
            try:
                self._heartbeat = HeartbeatThread(
                    mx_client=self.client,
                    mx_source_id=self._mx_source_id,
                    worker_id=self.worker_id,
                    worker_rank=self.rank,
                    nixl_manager=None,  # prime-rl drives NIXL outside MX's manager
                )
                self._heartbeat.start()
            except Exception as e:  # noqa: BLE001
                _log.warning(
                    "MxRendezvous: failed to start HeartbeatThread (role=%s rank=%s): %s",
                    self.role,
                    self.rank,
                    e,
                )

        return self._mx_source_id

    def close(self) -> None:
        """Stop the heartbeat thread. Safe to call multiple times."""
        if self._heartbeat is not None:
            try:
                self._heartbeat.stop()
            except Exception as e:  # noqa: BLE001
                _log.warning("MxRendezvous: heartbeat.stop() failed: %s", e)
            self._heartbeat = None

    def wait_for_peers(
        self,
        *,
        status: int | None = None,
        timeout: float = 1200.0,
        poll_interval: float = 1.0,
        same_rank_only: bool = False,
        dedup_freshest_per_rank: bool = True,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """Block until ``peer_world_size`` peers of the counterpart role are visible.

        Args:
            status: If set, only count peers in this :class:`p2p_pb2.SourceStatus`.
            timeout: Wall-clock seconds to wait before raising :class:`TimeoutError`.
            poll_interval: Seconds between ``ListSources`` polls.
            same_rank_only: If True, only return peers whose ``worker_rank``
                equals this rendezvous's own rank. Required on GB200's
                multi-NIC fabric where cross-subnet routing fails. Off by
                default to preserve the pre-Phase-2 single-NIC behaviour.
            dedup_freshest_per_rank: If True (default), keep only the
                freshest ``SourceInstanceRef`` per ``worker_rank``. This
                neutralises the stale-READY-after-restart bug we caught on
                GB200. Pass ``False`` to keep all duplicates (e.g. debug).
        """
        deadline = time.monotonic() + timeout
        peer_id = self._identity(self.peer_role)
        _logged = False
        while True:
            resp = self.client.list_sources(peer_id, status_filter=status)
            kept = list(resp.instances)
            if same_rank_only:
                kept = _filter_same_rank(kept, rank=self.rank)
            if dedup_freshest_per_rank and kept:
                kept = _freshest_per_rank(
                    kept, metas=self._collect_updated_at(kept)
                )
            if not _logged:
                all_resp = self.client.list_sources(peer_id)
                _log.info(
                    "wait_for_peers: role=%s need=%s found_with_status=%s found_any=%s "
                    "post_filter=%s status_filter=%s model=%s same_rank_only=%s",
                    self.peer_role,
                    self.peer_world_size,
                    len(resp.instances),
                    len(all_resp.instances),
                    len(kept),
                    status,
                    peer_id.model_name,
                    same_rank_only,
                )
                _logged = True
            if len(kept) >= self.peer_world_size:
                return kept
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {self.peer_world_size} "
                    f"{self.peer_role!r} peers (saw {len(kept)} after filters; "
                    f"{len(resp.instances)} raw)"
                )
            time.sleep(poll_interval)

    def _collect_updated_at(
        self, instances: Iterable[p2p_pb2.SourceInstanceRef]
    ) -> dict[str, int]:
        """Fetch ``updated_at`` per peer in one round of GetMetadata calls.

        Used by the freshest-per-rank dedup. Failures (missing worker, RPC
        errors) are mapped to ``0`` so the stale entries lose to anything
        with a real timestamp.
        """
        out: dict[str, int] = {}
        for inst in instances:
            try:
                resp = self.client.get_metadata(inst.mx_source_id, inst.worker_id)
            except Exception:  # noqa: BLE001
                out[inst.worker_id] = 0
                continue
            if not getattr(resp, "found", False):
                out[inst.worker_id] = 0
                continue
            out[inst.worker_id] = int(getattr(resp.worker, "updated_at", 0) or 0)
        return out

    def wait_for_all_peers_ready(
        self,
        *,
        role: Role | None = None,
        status: int = p2p_pb2.SOURCE_STATUS_READY,
        timeout: float = 1200.0,
        poll_interval: float = 0.05,
        same_rank_only: bool = False,
        dedup_freshest_per_rank: bool = True,
    ) -> list[p2p_pb2.SourceInstanceRef]:
        """Discover peer count from MX, then block until ALL of them reach ``status``.

        Unlike :meth:`wait_for_peers` (which requires a pre-known
        ``peer_world_size``), this method first counts how many peer-role
        entries exist in MX (any status) and uses that count as the target.
        Each side publishes one entry per rank, so the count equals the
        peer's world size — no config plumbing needed.

        Phase-2 additions (``same_rank_only`` and ``dedup_freshest_per_rank``)
        behave identically to :meth:`wait_for_peers`.
        """
        target_role = role or self.peer_role
        peer_id = self._identity(target_role)
        deadline = time.monotonic() + timeout

        def _apply_filters(
            insts: list[p2p_pb2.SourceInstanceRef],
        ) -> list[p2p_pb2.SourceInstanceRef]:
            kept = insts
            if same_rank_only:
                kept = _filter_same_rank(kept, rank=self.rank)
            if dedup_freshest_per_rank and kept:
                kept = _freshest_per_rank(
                    kept, metas=self._collect_updated_at(kept)
                )
            return kept

        peer_count = 0
        while peer_count == 0:
            insts = list(self.client.list_sources(peer_id).instances)
            kept = _apply_filters(insts)
            peer_count = len(kept)
            if peer_count == 0:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for {target_role!r} peers to appear in MX")
                time.sleep(poll_interval)

        while True:
            insts = list(self.client.list_sources(peer_id, status_filter=status).instances)
            kept = _apply_filters(insts)
            if len(kept) >= peer_count:
                return kept
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout}s waiting for {peer_count} "
                    f"{target_role!r} peers to reach status {status} (saw {len(kept)})"
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
