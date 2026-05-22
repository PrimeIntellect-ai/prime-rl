from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
import logging
from typing import Any

import msgspec
import torch
from modelexpress import p2p_pb2
from torch import Tensor

from prime_rl.trainer.models.slots import Slot
from prime_rl.transport.nixl_agent import NixlAgentWrapper
from prime_rl.transport.wire import PeerInfo, RendezvousPayload, WriteEntry

logger = logging.getLogger(__name__)


class TransportPlan:
    def __init__(self, agent: NixlAgentWrapper, peer_metadata: list[p2p_pb2.WorkerMetadata], slots: list[Slot]) -> None:
        """Initialize the transport plan with the given agent, peer metadata, and slots.
        This method will:
            1. Decode the peer_metadata into PeerInfo objects to create the peers list
            2. Builds the writes list by calling build_writes on each slot
            3. Prepares the local and remote preps by calling prep_local and prep_remote on the agent for each slot and peer
        """
        peers: list[PeerInfo] = []
        peer_tensor_sizes: dict[str, dict[str, int]] = {}
        peer_tensor_dtypes: dict[str, dict[str, str]] = {}
        for meta in peer_metadata:
            payload = msgspec.msgpack.decode(meta.nixl_metadata, type=RendezvousPayload)
            tensor_addrs = {td.name: (td.addr, td.size, td.device_id) for td in meta.tensors}
            peer_tensor_sizes[payload.agent_name] = {td.name: td.size for td in meta.tensors}
            peer_tensor_dtypes[payload.agent_name] = {td.name: td.dtype for td in meta.tensors}
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

        self._validate_descriptors(slots, peers, peer_tensor_sizes, peer_tensor_dtypes)

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

    def _validate_descriptors(
        self,
        slots: list[Slot],
        peers: list[PeerInfo],
        peer_tensor_sizes: dict[str, dict[str, int]],
        peer_tensor_dtypes: dict[str, dict[str, str]],
    ) -> None:
        buffer_tensors: dict[str, Tensor] = {}
        errors: list[str] = []
        covered_remote_names: set[str] = set()

        for slot in slots:
            for buf_key, tensor, num_chunks in slot.buffers:
                if buf_key in buffer_tensors:
                    errors.append(f"duplicate local NIXL buffer key {buf_key!r}")
                buffer_tensors[buf_key] = tensor
                total_bytes = tensor.numel() * tensor.element_size()
                if total_bytes % num_chunks != 0:
                    errors.append(
                        f"local buffer {buf_key!r} bytes={total_bytes} is not divisible by chunks={num_chunks}"
                    )

        entries_by_remote: dict[str, list[Any]] = defaultdict(list)
        for slot in slots:
            for entry in slot.layout_payload():
                entries_by_remote[entry.inference_name].append(entry)
                covered_remote_names.add(entry.inference_name)

        expert_slots = [slot for slot in slots if not slot.layout_payload() and hasattr(slot, "moe_prefix")]
        for slot in expert_slots:
            for buf_key, _, _ in slot.buffers:
                covered_remote_names.add(buf_key)

        for remote_name, entries in entries_by_remote.items():
            intervals = sorted((entry.offset_rows, entry.offset_rows + entry.rows, entry.slot_key) for entry in entries)
            expected_start = 0
            for start, stop, slot_key in intervals:
                if start != expected_start:
                    errors.append(
                        f"non-contiguous layout for {remote_name!r}: expected row {expected_start}, "
                        f"got {start} from {slot_key!r}"
                    )
                expected_start = stop

        for peer in peers:
            tensor_sizes = peer_tensor_sizes[peer.agent_name]
            tensor_dtypes = peer_tensor_dtypes[peer.agent_name]
            for remote_name, entries in entries_by_remote.items():
                if remote_name not in tensor_sizes:
                    errors.append(f"{peer.agent_name}: missing inference tensor {remote_name!r}")
                    continue

                remote_rows = max(entry.offset_rows + entry.rows for entry in entries)
                remote_size = tensor_sizes[remote_name]
                if remote_rows == 0 or remote_size % remote_rows != 0:
                    errors.append(
                        f"{peer.agent_name}: {remote_name!r} size={remote_size} is not divisible by rows={remote_rows}"
                    )
                    continue
                remote_row_bytes = remote_size // remote_rows

                for entry in entries:
                    tensor = buffer_tensors[entry.slot_key]
                    local_row_bytes = _row_bytes(entry.slot_key, tensor)
                    expected_local_rows = entry.rows // entry.num_chunks
                    if entry.rows % entry.num_chunks != 0:
                        errors.append(
                            f"{peer.agent_name}: {entry.slot_key!r} rows={entry.rows} is not divisible by "
                            f"chunks={entry.num_chunks}"
                        )
                    if tensor.shape[0] != expected_local_rows:
                        errors.append(
                            f"{peer.agent_name}: {entry.slot_key!r} local rows={tensor.shape[0]} "
                            f"expected={expected_local_rows} from layout"
                        )
                    if local_row_bytes != remote_row_bytes:
                        errors.append(
                            f"{peer.agent_name}: row-byte mismatch for {entry.slot_key!r} -> {remote_name!r}: "
                            f"local dtype={_dtype_name(tensor)} shape={tuple(tensor.shape)} row_bytes={local_row_bytes}; "
                            f"remote dtype={tensor_dtypes.get(remote_name)} size={remote_size} "
                            f"rows={remote_rows} row_bytes={remote_row_bytes}"
                        )

            for slot in expert_slots:
                moe_prefix = getattr(slot, "moe_prefix")
                peer_local_experts = len(peer.expert_map.get(moe_prefix, []))
                for buf_key, tensor, num_chunks in slot.buffers:
                    if buf_key not in tensor_sizes:
                        errors.append(f"{peer.agent_name}: missing expert tensor {buf_key!r}")
                        continue
                    if peer_local_experts == 0:
                        continue
                    remote_size = tensor_sizes[buf_key]
                    if remote_size % peer_local_experts != 0:
                        errors.append(
                            f"{peer.agent_name}: expert tensor {buf_key!r} size={remote_size} is not divisible by "
                            f"peer_local_experts={peer_local_experts}"
                        )
                        continue
                    remote_chunk_bytes = remote_size // peer_local_experts
                    local_total_bytes = tensor.numel() * tensor.element_size()
                    if local_total_bytes % num_chunks != 0:
                        errors.append(
                            f"{peer.agent_name}: expert local buffer {buf_key!r} bytes={local_total_bytes} "
                            f"is not divisible by chunks={num_chunks}"
                        )
                        continue
                    local_chunk_bytes = local_total_bytes // num_chunks
                    if local_chunk_bytes != remote_chunk_bytes:
                        errors.append(
                            f"{peer.agent_name}: expert chunk-byte mismatch for {buf_key!r}: "
                            f"local dtype={_dtype_name(tensor)} shape={tuple(tensor.shape)} "
                            f"chunk_bytes={local_chunk_bytes}; remote dtype={tensor_dtypes.get(buf_key)} "
                            f"size={remote_size} peer_local_experts={peer_local_experts} "
                            f"chunk_bytes={remote_chunk_bytes}"
                        )

            uncovered = sorted(set(tensor_sizes) - covered_remote_names)
            if uncovered:
                logger.warning(
                    "NIXL descriptor validation: peer %s has %d inference tensors with no trainer slot; first 20: %s",
                    peer.agent_name,
                    len(uncovered),
                    uncovered[:20],
                )

        if errors:
            raise RuntimeError("NIXL descriptor/slot validation failed:\n" + "\n".join(errors[:80]))

        logger.info(
            "NIXL descriptor validation passed for %d peers, %d slots, %d covered inference tensors",
            len(peers),
            len(slots),
            len(covered_remote_names),
        )

    def prepare_writes(self, state_dict: dict[str, Tensor], slots: list[Slot]) -> Iterator[tuple[Any, Any, WriteEntry]]:
        """Iterator yielding (local_prep, remote_prep, write_entry) tuples for each write. To be posted by the caller"""

        for slot in slots:
            slot.convert(state_dict)

        torch.cuda.synchronize()

        for entry in self.writes:
            local_prep = self.local_preps[entry.local_buffer_key]
            remote_prep = self.remote_preps[(entry.peer_name, entry.remote_buffer_key)]
            yield (local_prep, remote_prep, entry)


def _dtype_name(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _row_bytes(name: str, tensor: Tensor) -> int:
    if tensor.ndim == 0 or tensor.shape[0] == 0:
        raise ValueError(f"cannot compute row bytes for {name!r} with shape={tuple(tensor.shape)}")
    return tensor.numel() * tensor.element_size() // tensor.shape[0]
