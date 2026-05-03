"""Trainer-side orchestrator for the MX rendezvous + RDMA push.

After :class:`TrainerPublisher` and the inference ranks have published
their metadata once each, :class:`TransportPlan`:

1. :meth:`negotiate` — waits for inference peers to flip to ``READY``,
   fetches their full payloads (agent_metadata + tensor base addresses +
   ``expert_map``) and builds a :class:`PeerInfo` per peer.
2. :meth:`setup_remote_agents` — imports each peer's NIXL agent metadata
   so this trainer rank can prep dlists against them.
3. :meth:`prepare` — for each slot buffer, builds local + per-peer
   prepped dlists. Local chunks come from
   :attr:`Slot.buffers`; remote chunks come from
   :meth:`Slot.peer_chunk_descs`.
4. :meth:`push_once` — per training step: ``slot.convert(state_dict)``
   to fill slot buffers, then walk the :class:`WriteEntry` table posting
   ``WRITE`` transfers and waiting for completion.

Ordering: ``negotiate`` → ``setup_remote_agents`` → ``prepare`` once at
trainer startup; ``push_once`` per step; final ``set_status(READY)``
after each push so inference workers can resume request processing.
"""

from __future__ import annotations

from typing import Any

import msgspec
import torch
from modelexpress import p2p_pb2
from torch import Tensor

from prime_rl.transport.trainer_publisher import TrainerPublisher
from prime_rl.transport.wire import PeerInfo, RendezvousPayload, WriteEntry


class TransportPlan:
    def __init__(self, publisher: TrainerPublisher) -> None:
        self.publisher = publisher
        self.peers: list[PeerInfo] = []
        self.writes: list[WriteEntry] = []
        # Filled by prepare(): keyed by slot buffer key (local) and
        # ``(peer_agent_name, buffer_key)`` (remote).
        self.local_preps: dict[str, Any] = {}
        self.remote_preps: dict[tuple[str, str], Any] = {}

    def negotiate(self, *, timeout: float = 1200.0, poll_interval: float = 1.0) -> None:
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

    def setup_remote_agents(self) -> None:
        """Import every peer's NIXL agent metadata."""
        for peer in self.peers:
            self.publisher.agent.add_remote_agent(peer.agent_metadata)

    def prepare(self) -> None:
        """Build local + remote prepped dlists for every slot × peer × buffer."""
        for slot in self.publisher.slots:
            for buf_key, tensor, num_chunks in slot.buffers:
                total_bytes = tensor.numel() * tensor.element_size()
                chunk_bytes = total_bytes // num_chunks
                base_ptr = tensor.data_ptr()
                dev = tensor.get_device()
                local_descs = [(base_ptr + i * chunk_bytes, chunk_bytes, dev) for i in range(num_chunks)]
                self.local_preps[buf_key] = self.publisher.agent.prep_local(local_descs)

        for peer in self.peers:
            for slot in self.publisher.slots:
                for buf_key, descs in slot.peer_chunk_descs(peer).items():
                    if not descs:
                        continue  # peer owns no chunks for this slot (e.g. unowned experts)
                    self.remote_preps[(peer.agent_name, buf_key)] = self.publisher.agent.prep_remote(
                        peer.agent_name, descs
                    )

    def push_once(self, state_dict: dict[str, Tensor]) -> None:
        """One end-to-end push: convert sources into slots, post all WRITEs, wait.

        The state_dict values are cast to bfloat16 before conversion — the
        optimizer may keep parameters in float32 (optimization_dtype), but
        the FP8 block quantization must match the precision the checkpoint
        was originally quantized from (bfloat16). Without this cast, the
        per-block scales differ from what vLLM's FP8 kernels expect and KL
        drifts monotonically.
        """
        from torch.distributed.tensor import DTensor

        bf16_state: dict[str, Tensor] = {}
        for k, v in state_dict.items():
            if isinstance(v, DTensor):
                bf16_state[k] = v.to(torch.bfloat16)
            else:
                bf16_state[k] = v.to(torch.bfloat16) if v.is_floating_point() else v

        for slot in self.publisher.slots:
            slot.convert(bf16_state)
        # Ensure conversions are visible before remote agents pick up the bytes
        # over GPUDirect RDMA (writes bypass CUDA stream ordering).
        torch.cuda.synchronize()

        handles: list[tuple[Any, str]] = []
        for entry in self.writes:
            local_prep = self.local_preps[entry.local_buffer_key]
            remote_prep = self.remote_preps[(entry.peer_name, entry.remote_buffer_key)]
            handle = self.publisher.agent.post_write(
                local_prep=local_prep,
                local_idx=entry.local_chunk_idx,
                remote_prep=remote_prep,
                remote_idx=entry.remote_chunk_idx,
            )
            handles.append((handle, entry.tag))

        for handle, tag in handles:
            self.publisher.agent.wait(handle, context=tag)
