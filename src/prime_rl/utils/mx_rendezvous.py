"""MX Server rendezvous helper — MX-mediated replacement for PI's
hard-coded SPG host/port.

When the environment variable ``PRIME_RL_MX_RENDEZVOUS`` is set to a MX
Server address (e.g. ``modelexpress-server.kavin.svc.cluster.local:8001``),
participants discover the SPG coordinator endpoint via MX Server instead of
the config's static ``host``/``port`` fields. This enables:

- **Dynamic topology** — trainer/rollout pods can start in any order, no
  need to pre-configure each other's IPs.
- **Pipeline replication** — after receive, a rollout publishes itself as
  an additional source for the same version; future newcomers discover
  it via ``list_sources``.
- **Peer recovery** — a restarting pod pulls from a surviving peer rather
  than the trainer.

Scope for v0.1: only the ``host``/``port`` rendezvous is replaced. Per-step
``spg.barrier()`` calls still use the SPG process group (which has a fixed
world size once established). This keeps the change minimally invasive to
PI's :class:`TransportPlan`. Future v0.2 extensions can replace the
per-step barrier with an MX-mediated barrier for full elastic-mid-run
support.

Toggle via environment variables (no config changes to PI's code):

- ``PRIME_RL_MX_RENDEZVOUS``: MX Server address. When unset or empty,
  this module is a no-op and the caller falls back to ``config.host`` /
  ``config.port``.
- ``PRIME_RL_MX_MODEL_NAME``: model name (e.g. HF ID). Used as the
  rendezvous cohort key on MX Server.
- ``PRIME_RL_MX_TRAINING_RUN_ID``: unique run ID. Optional; auto-generated
  from hostname if unset. Ensures multiple concurrent training runs on
  the same MX Server don't collide on rendezvous.
- ``PRIME_RL_MX_PIPELINE_REPLICATION``: ``1`` to enable rollout-as-source
  publishing after receive. Default ``0``.

Env-var surface rather than config plumbing for v0.1 — keeps the PR-on-PR
small; follow-up can promote these to first-class config fields.
"""

from __future__ import annotations

import logging
import os
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# How long to wait at rendezvous before giving up (seconds). Training
# startup already takes tens of seconds, so we can afford minutes here.
_DEFAULT_TIMEOUT_SECONDS = 300.0
_POLL_INTERVAL = 0.5

# MX Server's get_metadata() only preserves a subset of WorkerMetadata
# fields across the gRPC round-trip (empirically: nixl_metadata, status,
# updated_at survive; metadata_endpoint and agent_name come back empty).
# We smuggle the SPG coordinator host:port through the bytes-typed
# nixl_metadata field, tagged with this prefix so consumers can reliably
# distinguish a rendezvous blob from a real NIXL agent metadata blob.
_RENDEZVOUS_BLOB_PREFIX = "primerl-mx-rendezvous:"


def mx_rendezvous_enabled() -> bool:
    """Return True iff ``PRIME_RL_MX_RENDEZVOUS`` is set to a non-empty value."""
    return bool(os.environ.get("PRIME_RL_MX_RENDEZVOUS", "").strip())


def pipeline_replication_enabled() -> bool:
    """Return True iff ``PRIME_RL_MX_PIPELINE_REPLICATION=1``."""
    return os.environ.get("PRIME_RL_MX_PIPELINE_REPLICATION", "0") == "1"


def scratch_mode_enabled() -> bool:
    """Return True iff ``PRIME_RL_MX_TRANSFER_MODE=scratch``.

    When True, the receiver stages RDMA writes in isolated GPU tensors
    and applies via ``model.load_weights()`` instead of writing directly
    into live vLLM parameter memory. Diagnostic mode — useful for
    isolating direct-refit correctness issues (e.g. the KL drift PI's
    team investigated in #2326).
    """
    return os.environ.get("PRIME_RL_MX_TRANSFER_MODE", "direct") == "scratch"


@dataclass
class RendezvousEndpoint:
    """SPG coordinator endpoint discovered via MX Server."""

    host: str
    port: int
    source_id: str  # mx_source_id (16-char hex)
    trainer_worker_id: str  # publisher's worker_id


def _get_run_id() -> str:
    """Return a stable run ID for this training run.

    Prefers ``PRIME_RL_MX_TRAINING_RUN_ID`` (explicit, recommended). The
    hostname-derived fallback below is best-effort only — trainer and
    inference pods in the same run typically have *different* hostname
    prefixes (e.g. ``prime-rl-mx-trainer-0`` vs
    ``prime-rl-mx-inference-0``), so naive prefix-stripping yields
    different IDs and the rendezvous never converges. The fallback
    handles that by stripping a known role suffix (``-trainer-N``,
    ``-inference-N``, ``-rollout-N``, ``-orchestrator-...``) and
    returning the remaining cluster prefix shared by every pod in
    the run.

    Production deployments **must** set ``PRIME_RL_MX_TRAINING_RUN_ID``
    explicitly — depending on the fallback's role-suffix list is fragile
    if pods are renamed.
    """
    env_id = os.environ.get("PRIME_RL_MX_TRAINING_RUN_ID", "").strip()
    if env_id:
        return env_id

    hostname = socket.gethostname()
    # Try to strip a recognized role suffix so all pods in one run
    # converge on the same prefix. Order matters — match longest
    # role names first.
    for role in ("trainer", "inference", "rollout", "orchestrator"):
        token = f"-{role}-"
        idx = hostname.find(token)
        if idx > 0:
            return hostname[:idx]
    # Fallback to bare hostname-minus-ordinal. Will *not* match across
    # roles; logged as warning so the misconfiguration is visible.
    parts = hostname.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        derived = parts[0]
    else:
        derived = hostname
    logger.warning(
        "[mx-rendezvous] PRIME_RL_MX_TRAINING_RUN_ID not set; falling back "
        "to hostname-derived run_id=%r. This may differ across pod roles "
        "(trainer/inference) and prevent rendezvous. Set the env var "
        "explicitly in production.",
        derived,
    )
    return derived


def _get_model_name() -> str:
    """Return the model name used for the rendezvous cohort key."""
    return os.environ.get("PRIME_RL_MX_MODEL_NAME", "unknown").strip() or "unknown"


def _build_identity(run_id: str, model_name: str) -> Any:
    """Build a SourceIdentity keyed to (run_id, model_name) so concurrent
    runs on the same MX Server don't collide.
    """
    from modelexpress import p2p_pb2  # type: ignore

    identity = p2p_pb2.SourceIdentity(
        mx_version="0.3.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        dtype="bfloat16",
        quantization="",
        extra_parameters={
            "prime_rl_run_id": run_id,
            "rendezvous_version": "v0.1",
        },
    )
    return identity


def discover_spg_coordinator(
    role: str,
    rank: int,
    expected_trainer_ws: int,
    expected_inference_ws: int,
    fallback_host: str,
    fallback_port: int,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> RendezvousEndpoint:
    """Discover the SPG coordinator (host, port) via MX Server.

    Protocol:
    - Trainer rank 0 binds the SPG master on its local IP + ``fallback_port``
      and publishes the endpoint to MX Server.
    - All other participants (trainer ranks 1..N, inference ranks 0..M)
      poll MX Server for the published endpoint.
    - Once published+discovered, everyone returns the same endpoint and
      proceeds to create SPG against it.

    If ``mx_rendezvous_enabled()`` is False, this function is a no-op
    that returns ``RendezvousEndpoint(fallback_host, fallback_port, "", "")``
    — caller should check and skip the MX gRPC path in that case.

    Args:
        role: ``"trainer"`` or ``"inference"``.
        rank: Global rank within ``role``.
        expected_trainer_ws: Number of trainer ranks expected in cohort.
        expected_inference_ws: Number of inference ranks expected.
        fallback_host: Host to bind/use if ``rank == 0 and role == "trainer"``.
        fallback_port: Port to bind/use if ``rank == 0 and role == "trainer"``.
        timeout: Max seconds to wait for discovery.

    Returns:
        RendezvousEndpoint pointing to the SPG coordinator.

    Raises:
        RuntimeError: If rendezvous times out or fails.
    """
    if not mx_rendezvous_enabled():
        return RendezvousEndpoint(
            host=fallback_host,
            port=fallback_port,
            source_id="",
            trainer_worker_id="",
        )

    from modelexpress import MxClient, p2p_pb2  # type: ignore

    mx_url = os.environ["PRIME_RL_MX_RENDEZVOUS"]
    run_id = _get_run_id()
    model_name = _get_model_name()

    logger.info(
        "[mx-rendezvous] role=%s rank=%d mx_url=%s run_id=%s model=%s "
        "expected_trainer_ws=%d expected_inference_ws=%d",
        role,
        rank,
        mx_url,
        run_id,
        model_name,
        expected_trainer_ws,
        expected_inference_ws,
    )

    client = MxClient(server_url=mx_url)
    identity = _build_identity(run_id, model_name)
    is_coordinator = role == "trainer" and rank == 0

    # Each participant gets a unique worker_id so the server can track us
    # individually even if two replicas share a worker_rank.
    my_worker_id = f"{run_id}-{role}-{rank}-{uuid.uuid4().hex[:8]}"

    # Coordinator: publish the endpoint other participants will use.
    if is_coordinator:
        coordinator_host = _get_local_ip(fallback_host)
        coordinator_port = fallback_port

        # Empirically MX Server's get_metadata() only round-trips a subset
        # of WorkerMetadata fields (nixl_metadata, status, updated_at).
        # metadata_endpoint / agent_name come back empty. Stuff the SPG
        # coordinates into the nixl_metadata bytes field so peer lookups
        # actually survive the server round-trip. Format is a tiny self-
        # describing UTF-8 string prefixed with a magic tag so callers
        # know to parse it as a rendezvous endpoint rather than as a
        # real NIXL agent blob.
        coord_blob = (
            f"{_RENDEZVOUS_BLOB_PREFIX}{coordinator_host}:{coordinator_port}"
        ).encode("utf-8")

        worker = p2p_pb2.WorkerMetadata(
            worker_rank=rank,
            nixl_metadata=coord_blob,
            tensors=[],
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=int(time.time() * 1000),
            metadata_endpoint=f"{coordinator_host}:{coordinator_port}",
            agent_name=f"spg-coordinator-{run_id}",
        )
        source_id = client.publish_metadata(identity, worker, my_worker_id)
        logger.info(
            "[mx-rendezvous] coordinator published source_id=%s endpoint=%s:%d",
            source_id,
            coordinator_host,
            coordinator_port,
        )
        # Coordinator also "discovers" itself so the return value is
        # consistent across all ranks.
        return RendezvousEndpoint(
            host=coordinator_host,
            port=coordinator_port,
            source_id=source_id,
            trainer_worker_id=my_worker_id,
        )

    # Non-coordinator: poll until we see the coordinator's publish.
    #
    # Catalog state can contain *multiple* workers with worker_rank=0 —
    # the trainer coordinator (which encodes the SPG endpoint into
    # nixl_metadata with our magic prefix) PLUS any inference rank 0
    # / rollout-as-source rank 0 entries which carry empty
    # nixl_metadata. Picking the first worker_rank==0 ref unconditionally
    # may match a non-coordinator entry first, then loop forever waiting
    # for an endpoint that's never going to materialize on that entry.
    #
    # Solution: scan all worker_rank==0 candidates per poll, fetch each
    # one's metadata, and only accept the candidate whose nixl_metadata
    # starts with _RENDEZVOUS_BLOB_PREFIX (the coordinator marker).
    deadline = time.time() + timeout
    backoff = _POLL_INTERVAL
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        resp = client.list_sources(identity=identity)
        coordinator_candidates = [ref for ref in resp.instances if ref.worker_rank == 0]
        if not coordinator_candidates:
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 2.0)
            continue

        endpoint = ""
        coordinator_ref = None
        for ref in coordinator_candidates:
            meta_resp = client.get_metadata(ref.mx_source_id, ref.worker_id)
            if not meta_resp.found:
                continue
            # MX Server's get_metadata strips metadata_endpoint/agent_name, so
            # the coordinator encodes host:port into nixl_metadata prefixed
            # with _RENDEZVOUS_BLOB_PREFIX. Skip any worker_rank==0 entry
            # that doesn't carry that prefix — those are non-coordinator
            # rank-0 publishers (inference rank 0, rollout-as-source).
            blob = meta_resp.worker.nixl_metadata or b""
            try:
                decoded = blob.decode("utf-8", errors="replace")
                if decoded.startswith(_RENDEZVOUS_BLOB_PREFIX):
                    candidate_endpoint = decoded[len(_RENDEZVOUS_BLOB_PREFIX):]
                    if ":" in candidate_endpoint:
                        endpoint = candidate_endpoint
                        coordinator_ref = ref
                        break
            except Exception:  # noqa: BLE001 — try next candidate
                continue
            # Forward-compat fallback: if a future MX Server version
            # persists metadata_endpoint, accept that too — but only if
            # it's non-empty (empty endpoint = non-coordinator entry).
            legacy_endpoint = meta_resp.worker.metadata_endpoint
            if legacy_endpoint and ":" in legacy_endpoint:
                endpoint = legacy_endpoint
                coordinator_ref = ref
                break

        if not endpoint or coordinator_ref is None:
            time.sleep(backoff)
            backoff = min(backoff * 1.5, 2.0)
            continue
        host_str, port_str = endpoint.rsplit(":", 1)
        logger.info(
            "[mx-rendezvous] role=%s rank=%d discovered coordinator=%s after %d polls",
            role,
            rank,
            endpoint,
            attempts,
        )

        # Also publish our own metadata so pipeline replication / peer
        # recovery works — downstream code can call
        # ``publish_rollout_source`` after receive finalizes.
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=rank,
            nixl_metadata=b"",
            tensors=[],
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            updated_at=int(time.time() * 1000),
            metadata_endpoint="",
            agent_name=f"{role}-{run_id}-rank-{rank}",
        )
        try:
            client.publish_metadata(identity, worker, my_worker_id)
        except Exception as exc:  # noqa: BLE001 — cosmetic only
            logger.warning("[mx-rendezvous] non-coordinator publish failed (non-fatal): %s", exc)

        return RendezvousEndpoint(
            host=host_str,
            port=int(port_str),
            source_id=coordinator_ref.mx_source_id,
            trainer_worker_id=coordinator_ref.worker_id,
        )

    raise RuntimeError(
        f"[mx-rendezvous] role={role} rank={rank} timed out after {timeout}s "
        f"waiting for coordinator in model={model_name} run={run_id}"
    )


def _get_local_ip(fallback: str) -> str:
    """Best-effort local IP discovery. Returns ``fallback`` (the configured
    host, typically ``localhost`` or a service name) on failure.
    """
    if fallback not in ("localhost", "0.0.0.0", "127.0.0.1", ""):
        # Caller already configured a specific host — trust it.
        return fallback
    try:
        # Discover the primary outbound IP (doesn't actually send traffic).
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostbyname(socket.gethostname())


def publish_as_rollout_source(
    source_id: str,
    rank: int,
    run_id: str | None = None,
    model_name: str | None = None,
) -> None:
    """Publish the local rollout as an additional source for pipeline
    replication.

    Called from the inference worker after finalize() completes. Future
    pollers for the same (model, rank) may discover this rollout as an
    alternate source, reducing trainer NIC pressure.

    No-op if pipeline replication is disabled.
    """
    if not pipeline_replication_enabled():
        return
    if not mx_rendezvous_enabled():
        logger.warning(
            "[mx-rendezvous] pipeline_replication requested but PRIME_RL_MX_RENDEZVOUS "
            "is not set; skipping rollout-source publish"
        )
        return

    from modelexpress import MxClient, p2p_pb2  # type: ignore

    mx_url = os.environ["PRIME_RL_MX_RENDEZVOUS"]
    run_id = run_id or _get_run_id()
    model_name = model_name or _get_model_name()

    client = MxClient(server_url=mx_url)
    identity = _build_identity(run_id, model_name)

    # Secondary source uses a fresh worker_id and marks itself READY.
    # Future pollers will see multiple workers for this rank and can
    # pick any of them.
    my_worker_id = f"{run_id}-rollout-source-{rank}-{uuid.uuid4().hex[:8]}"
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=rank,
        nixl_metadata=b"",  # stub; real impl would pass our agent_meta
        tensors=[],
        status=p2p_pb2.SOURCE_STATUS_READY,
        updated_at=int(time.time() * 1000),
        metadata_endpoint="",
        agent_name=f"rollout-source-{run_id}-rank-{rank}",
    )
    try:
        client.publish_metadata(identity, worker, my_worker_id)
        logger.info(
            "[mx-rendezvous] published rollout-as-source rank=%d worker_id=%s",
            rank,
            my_worker_id,
        )
    except Exception as exc:  # noqa: BLE001 — non-fatal
        logger.warning("[mx-rendezvous] rollout-source publish failed (non-fatal): %s", exc)
