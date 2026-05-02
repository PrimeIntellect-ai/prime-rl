"""Wire-format types exchanged between the trainer publisher, inference
receiver, and transport plan.

* :class:`LayoutEntry` — a trainer-side registered buffer that the inference
  side needs to narrow into its destination tensor and chunk for RDMA.
  Published as part of the trainer's rendezvous payload.
* :class:`PeerInfo` — one inference peer's response: its NIXL agent name,
  its serialized chunked xfer descriptors keyed by buffer name, and the
  ``expert_map`` for MoE routing.
* :class:`WriteEntry` — one RDMA WRITE description, produced by a slot
  given a peer list and resolved by the transport plan into NIXL prep
  handles + ``post_write`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutEntry:
    slot_key: str
    inference_name: str
    offset_rows: int
    rows: int
    num_chunks: int  # trainer_ws for sharded buffers, 1 for gathered


@dataclass(frozen=True)
class PeerInfo:
    """One inference peer's payload.

    ``descriptors`` maps :attr:`LayoutEntry.slot_key` (or an expert
    destination name) to a list of serialized xfer dlists, one per chunk.
    ``expert_map`` maps a MoE prefix to the list of global expert IDs
    the peer owns.
    """

    agent_name: str
    descriptors: dict[str, list[bytes]]
    expert_map: dict[str, list[int]]


@dataclass(frozen=True)
class WriteEntry:
    """One RDMA WRITE description, resolved later by the transport plan."""

    local_buffer_key: str
    local_chunk_idx: int
    peer_name: str
    remote_buffer_key: str
    remote_chunk_idx: int
    tag: str  # diagnostics
