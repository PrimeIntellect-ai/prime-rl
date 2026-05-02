"""Trainer-side orchestrator over MX.

After :class:`TrainerPublisher` and the inference ranks have each done
their single MX publish, :class:`TransportPlan`:

1. Waits for inference peers to flip to ``READY`` status (their setup
   is complete and their NIXL agents accept remote writes).
2. Fetches each peer's :class:`RendezvousPayload` plus the
   :class:`p2p_pb2.WorkerMetadata.tensors` table; assembles a
   :class:`PeerInfo` per peer.
3. Calls :meth:`Slot.build_writes` on each trainer slot to accumulate
   the complete RDMA WRITE table for this trainer rank.

Per-chunk dlists aren't sent over the wire: the trainer constructs them
locally at RDMA-prep time (next step) using each peer's tensor base
addresses + its own :class:`LayoutEntry` list.
"""

from __future__ import annotations

import msgspec
from modelexpress import p2p_pb2

from prime_rl.transport.trainer_publisher import TrainerPublisher
from prime_rl.transport.wire import PeerInfo, RendezvousPayload, WriteEntry


class TransportPlan:
    def __init__(self, publisher: TrainerPublisher) -> None:
        self.publisher = publisher
        self.peers: list[PeerInfo] = []
        self.writes: list[WriteEntry] = []

    def negotiate(self, *, timeout: float = 1200.0, poll_interval: float = 1.0) -> None:
        """Wait for ``READY`` inference peers, build :class:`PeerInfo`s + write table."""
        rendezvous = self.publisher.rendezvous
        peer_refs = rendezvous.wait_for_peers(
            status=p2p_pb2.SOURCE_STATUS_READY,
            timeout=timeout,
            poll_interval=poll_interval,
        )

        peers: list[PeerInfo] = []
        for ref in peer_refs:
            meta = rendezvous.fetch_peer(ref)
            payload = msgspec.msgpack.decode(meta.nixl_metadata, type=RendezvousPayload)
            tensor_addrs = {td.name: (td.addr, td.size, td.device_id) for td in meta.tensors}
            peers.append(
                PeerInfo(
                    agent_name=payload.agent_name,
                    agent_metadata=payload.agent_metadata,
                    tensor_addrs=tensor_addrs,
                    expert_map=payload.expert_map,
                )
            )

        self.peers = peers
        self.writes = []
        for slot in self.publisher.slots:
            self.writes.extend(slot.build_writes(peers))
