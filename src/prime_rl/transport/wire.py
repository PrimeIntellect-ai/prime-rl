"""Wire-format types exchanged between the trainer publisher, inference
receiver, and transport plan, all riding on Model Express.

* :class:`LayoutEntry` — a trainer-side registered buffer that the inference
  side needs to narrow into its destination tensor and chunk for RDMA.
  Published as part of the trainer's :class:`RendezvousPayload`.
* :class:`PeerInfo` — the trainer's view of one inference peer after both
  publishes have landed: NIXL agent name, serialized chunked xfer
  descriptors keyed by buffer name, and the ``expert_map``.
* :class:`WriteEntry` — one RDMA WRITE description, produced by a slot
  given a peer list and resolved by the transport plan into NIXL prep
  handles + ``post_write`` calls.
* :class:`RendezvousPayload` — what gets msgpack-encoded into
  :attr:`p2p_pb2.WorkerMetadata.nixl_metadata` so MX can carry both the
  raw NIXL agent metadata blob *and* our auxiliary structured fields on
  one channel.
"""

from __future__ import annotations

import msgspec


class LayoutEntry(msgspec.Struct, frozen=True):
    slot_key: str
    inference_name: str
    offset_rows: int
    rows: int
    num_chunks: int  # trainer_ws for sharded buffers, 1 for gathered


class PeerInfo(msgspec.Struct):
    """One peer's payload after fetching and unpacking via MX.

    ``tensor_addrs`` maps tensor name → ``(base_addr, total_bytes,
    device_id)`` for every NIXL-registered buffer the peer published; the
    trainer combines this with its own :class:`LayoutEntry` list at
    RDMA-prep time to build per-chunk dlists locally — no need for the
    peer to round-trip serialized descriptors. ``expert_map`` maps a MoE
    prefix to the list of global expert IDs the peer owns.
    """

    agent_name: str
    agent_metadata: bytes
    tensor_addrs: dict[str, tuple[int, int, int]]
    expert_map: dict[str, list[int]]


class WriteEntry(msgspec.Struct, frozen=True):
    """One RDMA WRITE description, resolved later by the transport plan."""

    local_buffer_key: str
    local_chunk_idx: int
    peer_name: str
    remote_buffer_key: str
    remote_chunk_idx: int
    tag: str  # diagnostics


class RendezvousPayload(msgspec.Struct):
    """Packed blob carried in :attr:`p2p_pb2.WorkerMetadata.nixl_metadata`.

    Trainer publishes ``agent_metadata`` + ``agent_name`` + ``layout``.
    Inference publishes ``agent_metadata`` + ``agent_name`` + ``expert_map``.
    Each side publishes once; the trainer chunks remote dlists locally at
    RDMA-prep time using inference's tensor base addresses (from
    :attr:`p2p_pb2.WorkerMetadata.tensors`) plus its own ``layout``.
    """

    agent_metadata: bytes
    agent_name: str = ""
    layout: list[LayoutEntry] = msgspec.field(default_factory=list)
    expert_map: dict[str, list[int]] = msgspec.field(default_factory=dict)
