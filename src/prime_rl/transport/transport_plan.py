from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import msgspec
import torch
from modelexpress import p2p_pb2
from torch import Tensor

from prime_rl.trainer.models.slots import Slot
from prime_rl.transport.nixl_agent import NixlAgentWrapper
from prime_rl.transport.wire import PeerInfo, RendezvousPayload, WriteEntry


class TransportPlan:
    def __init__(self, agent: NixlAgentWrapper, peer_metadata: list[p2p_pb2.WorkerMetadata], slots: list[Slot]) -> None:
        """Initialize the transport plan with the given agent, peer metadata, and slots.
        This method will:
            1. Decode the peer_metadata into PeerInfo objects to create the peers list
            2. Builds the writes list by calling build_writes on each slot
            3. Prepares the local and remote preps by calling prep_local and prep_remote on the agent for each slot and peer
        """
        peers: list[PeerInfo] = []
        for meta in peer_metadata:
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

        self.peers: list[PeerInfo] = peers
        self.writes: list[WriteEntry] = []
        self.local_preps: dict[str, Any] = {}
        self.remote_preps: dict[tuple[str, str], Any] = {}

        for slot in slots:
            self.writes.extend(slot.build_writes(peers))

        for peer in self.peers:
            name = agent.add_remote_agent(peer.agent_metadata)
            agent.make_connection(name)

        for slot in slots:
            for buf_key, tensor, num_chunks in slot.buffers:
                total_bytes = tensor.numel() * tensor.element_size()
                chunk_bytes = total_bytes // num_chunks
                base_ptr = tensor.data_ptr()
                dev = tensor.get_device()
                local_descs = [(base_ptr + i * chunk_bytes, chunk_bytes, dev) for i in range(num_chunks)]
                self.local_preps[buf_key] = agent.prep_local(local_descs)

        for peer in self.peers:
            for slot in slots:
                for buf_key, descs in slot.peer_chunk_descs(peer).items():
                    if not descs:
                        continue  # peer owns no chunks for this slot (e.g. unowned experts)
                    self.remote_preps[(peer.agent_name, buf_key)] = agent.prep_remote(peer.agent_name, descs)

    def prepare_writes(self, state_dict: dict[str, Tensor], slots: list[Slot]) -> Iterator[tuple[Any, Any, WriteEntry]]:
        """Iterator yielding (local_prep, remote_prep, write_entry) tuples for each write. To be posted by the caller"""

        for slot in slots:
            slot.convert(state_dict)

        torch.cuda.synchronize()

        for entry in self.writes:
            local_prep = self.local_preps[entry.local_buffer_key]
            remote_prep = self.remote_preps[(entry.peer_name, entry.remote_buffer_key)]
            yield (local_prep, remote_prep, entry)
