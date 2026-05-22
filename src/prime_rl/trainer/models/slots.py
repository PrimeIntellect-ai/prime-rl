"""Trainer-side destination slots for NIXL weight transfer.

Three slot types share a uniform protocol:

* :class:`ShardedSlot` — non-expert param whose dim-0 is FSDP-shardable and
  large enough to shard. The slot holds *this rank's* shard. Writes to
  ``chunk[my_rank]`` on every inference peer.
* :class:`GatheredSlot` — non-expert param too small to shard or whose
  shape doesn't divide. The slot holds the full tensor. Written once per
  peer, round-robin across trainer ranks (``i % trainer_ws == my_rank``).
* :class:`ExpertSlot` — MoE expert param, fused per-rank into a 3D buffer
  along ``cat_dim``. Each local expert is one chunk; writes target peers
  that own each global expert (via vLLM's ``expert_map``).

Each slot captures everything it needs at construction (``my_rank``,
``trainer_ws``, ``owned_global_experts``) so runtime methods only take
dynamic inputs (peers, source state_dict). Wire types
(:class:`~prime_rl.transport.wire.LayoutEntry`,
:class:`~prime_rl.transport.wire.PeerInfo`,
:class:`~prime_rl.transport.wire.WriteEntry`) live in the transport layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.conversions import ConversionEntry, resolve
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.parallel_dims import ParallelDims
from prime_rl.transport.wire import LayoutEntry, PeerInfo, WriteEntry

# Source tensors smaller than this fall out of the per-shard NIXL path and
# are gathered instead — below ~2 MiB the per-shard fragment drops under
# 32 KiB per peer and the RDMA handle overhead eats any parallelism gain.
SMALL_NON_EXPERT_BYTES = 2 * 1024 * 1024


_CAST_SLOT_DTYPES = {
    "bf16_cast": torch.bfloat16,
    "fp32_cast": torch.float32,
}


def _source_for_transfer(src: Tensor) -> Tensor:
    if not src.is_floating_point():
        return src
    return src.to(torch.bfloat16)


def _slot_dtype(conversion_type: str, conversion: ConversionEntry, base_dtype: torch.dtype) -> torch.dtype:
    if conversion.requires_scale:
        return torch.float8_e4m3fn
    return _CAST_SLOT_DTYPES.get(conversion_type, base_dtype)


# --- Slot protocol --------------------------------------------------------- #


class Slot(Protocol):
    weight: Tensor
    scale: Optional[Tensor]
    spec: ConversionSpec
    conversion: ConversionEntry

    @property
    def buffers(self) -> list[tuple[str, Tensor, int]]:
        """``(buffer_key, tensor, num_chunks)`` triples for NIXL registration."""
        ...

    def convert(self, state_dict: dict[str, Tensor]) -> None:
        """Pull this slot's source tensor(s) from ``state_dict`` and write into the buffers."""
        ...

    def layout_payload(self) -> list[LayoutEntry]:
        """Layout entries to publish so inference can chunk its destination."""
        ...

    def build_writes(self, peers: list[PeerInfo]) -> list[WriteEntry]:
        """One RDMA WRITE per ``(buffer, peer-chunk)`` this slot owns."""
        ...

    def peer_chunk_descs(self, peer: PeerInfo) -> dict[str, list[tuple[int, int, int]]]:
        """Per-buffer ``(addr, size, device_id)`` tuples for ``peer``'s side.

        Returns one entry per :attr:`buffers` key. The list length matches
        the peer's chunk count for that buffer (``trainer_ws`` for sharded,
        ``1`` for gathered, ``len(peer.expert_map[moe_prefix])`` for
        experts). Used by the transport plan to ``prep_remote`` per
        (peer, buffer).
        """
        ...


# --- Non-expert slots ------------------------------------------------------ #


@dataclass
class ShardedSlot:
    """Non-expert slot holding one FSDP shard. Writes to every peer at chunk[my_rank]."""

    weight: Tensor
    scale: Optional[Tensor]
    spec: ConversionSpec
    conversion: ConversionEntry
    source_name: str  # full source name in state_dict
    slot_key: str  # local + remote buffer key for the weight
    scale_key: Optional[str]  # local scale buffer key (per-source naming)
    inference_name: str  # destination name on inference side (fused dst)
    inference_scale_name: Optional[str]
    offset_rows: int  # this source's row offset in the fused inference dst
    scale_offset_rows: Optional[int]
    rows: int  # source's full dim-0
    scale_rows: Optional[int]
    my_rank: int
    trainer_ws: int

    @classmethod
    def from_spec(
        cls,
        spec: ConversionSpec,
        conversion: ConversionEntry,
        prefix: str,
        src_name: str,
        src: Tensor,
        parallel_dims: ParallelDims,
        slot_dtype: torch.dtype,
        offset_rows: int,
        scale_offset_rows: int,
    ) -> "ShardedSlot":
        fsdp_total = parallel_dims.dp_shard * parallel_dims.cp
        if dist.is_initialized():
            mesh = parallel_dims.get_mesh("dp_shard_cp")
            my_rank, trainer_ws = mesh.get_local_rank(), mesh.size()
        else:
            my_rank, trainer_ws = 0, 1
        src_rows = src.shape[0]
        rows_per_shard = src_rows // fsdp_total
        weight = torch.empty(
            (rows_per_shard,) + tuple(src.shape[1:]),
            dtype=slot_dtype,
            device=src.device,
        )
        slot_key = f"{prefix}.{src_name}" if prefix else src_name
        scale: Optional[Tensor] = None
        scale_key: Optional[str] = None
        inference_scale_name: Optional[str] = None
        scale_rows: Optional[int] = None
        if conversion.requires_scale:
            scale = torch.empty(
                (ceil_div(weight.shape[0], BLOCK_SIZE), ceil_div(weight.shape[1], BLOCK_SIZE)),
                dtype=torch.float32,
                device=weight.device,
            )
            scale_key = ConversionSpec.scale_name(slot_key, allow_direct_parameter=True)
            inference_scale_name = ConversionSpec.scale_name(f"{prefix}.{spec.dst}" if prefix else spec.dst)
            scale_rows = ceil_div(src_rows, BLOCK_SIZE)
        return cls(
            weight=weight,
            scale=scale,
            spec=spec,
            conversion=conversion,
            source_name=slot_key,
            slot_key=slot_key,
            scale_key=scale_key,
            inference_name=f"{prefix}.{spec.dst}" if prefix else spec.dst,
            inference_scale_name=inference_scale_name,
            offset_rows=offset_rows,
            scale_offset_rows=scale_offset_rows if conversion.requires_scale else None,
            rows=src_rows,
            scale_rows=scale_rows,
            my_rank=my_rank,
            trainer_ws=trainer_ws,
        )

    @property
    def buffers(self) -> list[tuple[str, Tensor, int]]:
        out: list[tuple[str, Tensor, int]] = [(self.slot_key, self.weight, 1)]
        if self.scale is not None:
            assert self.scale_key is not None
            out.append((self.scale_key, self.scale, 1))
        return out

    def convert(self, state_dict: dict[str, Tensor]) -> None:
        src = _source_for_transfer(state_dict[self.source_name])
        if isinstance(src, DTensor):
            src = src.full_tensor() if self.weight.shape[0] == src.shape[0] else src.to_local()
        self.conversion.fn(src, self.weight, self.scale)

    def layout_payload(self) -> list[LayoutEntry]:
        entries = [
            LayoutEntry(
                slot_key=self.slot_key,
                inference_name=self.inference_name,
                offset_rows=self.offset_rows,
                rows=self.rows,
                num_chunks=self.trainer_ws,
            )
        ]
        if self.scale is not None:
            assert self.scale_key is not None and self.scale_rows is not None
            assert self.inference_scale_name is not None and self.scale_offset_rows is not None
            entries.append(
                LayoutEntry(
                    slot_key=self.scale_key,
                    inference_name=self.inference_scale_name,
                    offset_rows=self.scale_offset_rows,
                    rows=self.scale_rows,
                    num_chunks=self.trainer_ws,
                )
            )
        return entries

    def build_writes(self, peers: list[PeerInfo]) -> list[WriteEntry]:
        out: list[WriteEntry] = []
        for peer in peers:
            for buf_key, _, _ in self.buffers:
                out.append(
                    WriteEntry(
                        local_buffer_key=buf_key,
                        local_chunk_idx=0,
                        peer_name=peer.agent_name,
                        remote_buffer_key=buf_key,
                        remote_chunk_idx=self.my_rank,
                        tag=f"per_shard:{buf_key}",
                    )
                )
        return out

    def peer_chunk_descs(self, peer: PeerInfo) -> dict[str, list[tuple[int, int, int]]]:
        out: dict[str, list[tuple[int, int, int]]] = {}
        # Weight: ``trainer_ws`` chunks along inference dst dim 0, each
        # ``rows/trainer_ws`` rows wide, starting at ``offset_rows``.
        weight_row_bytes = self.weight.numel() * self.weight.element_size() // self.weight.shape[0]
        weight_chunk_rows = self.rows // self.trainer_ws
        weight_base, _, weight_dev = peer.tensor_addrs[self.inference_name]
        out[self.slot_key] = [
            (
                weight_base + (self.offset_rows + i * weight_chunk_rows) * weight_row_bytes,
                weight_chunk_rows * weight_row_bytes,
                weight_dev,
            )
            for i in range(self.trainer_ws)
        ]
        if self.scale is not None:
            assert self.scale_key is not None and self.scale_rows is not None
            assert self.inference_scale_name is not None and self.scale_offset_rows is not None
            scale_row_bytes = self.scale.numel() * self.scale.element_size() // self.scale.shape[0]
            scale_chunk_rows = self.scale_rows // self.trainer_ws
            scale_base, _, scale_dev = peer.tensor_addrs[self.inference_scale_name]
            out[self.scale_key] = [
                (
                    scale_base + (self.scale_offset_rows + i * scale_chunk_rows) * scale_row_bytes,
                    scale_chunk_rows * scale_row_bytes,
                    scale_dev,
                )
                for i in range(self.trainer_ws)
            ]
        return out


@dataclass
class GatheredSlot:
    """Non-expert slot holding the full tensor. Written once per peer, round-robin across trainer ranks."""

    weight: Tensor
    scale: Optional[Tensor]
    spec: ConversionSpec
    conversion: ConversionEntry
    source_name: str
    slot_key: str
    scale_key: Optional[str]
    inference_name: str
    inference_scale_name: Optional[str]
    offset_rows: int
    scale_offset_rows: Optional[int]
    rows: int
    scale_rows: Optional[int]
    my_rank: int
    trainer_ws: int

    @classmethod
    def from_spec(
        cls,
        spec: ConversionSpec,
        conversion: ConversionEntry,
        prefix: str,
        src_name: str,
        src: Tensor,
        parallel_dims: ParallelDims,
        slot_dtype: torch.dtype,
        offset_rows: int,
        scale_offset_rows: int,
    ) -> "GatheredSlot":
        if dist.is_initialized():
            mesh = parallel_dims.get_mesh("dp_shard_cp")
            my_rank, trainer_ws = mesh.get_local_rank(), mesh.size()
        else:
            my_rank, trainer_ws = 0, 1
        weight = torch.empty(tuple(src.shape), dtype=slot_dtype, device=src.device)
        slot_key = f"{prefix}.{src_name}" if prefix else src_name
        scale: Optional[Tensor] = None
        scale_key: Optional[str] = None
        inference_scale_name: Optional[str] = None
        scale_rows: Optional[int] = None
        src_rows = src.shape[0]
        if conversion.requires_scale:
            scale = torch.empty(
                (ceil_div(weight.shape[0], BLOCK_SIZE), ceil_div(weight.shape[1], BLOCK_SIZE)),
                dtype=torch.float32,
                device=weight.device,
            )
            scale_key = ConversionSpec.scale_name(slot_key, allow_direct_parameter=True)
            inference_scale_name = ConversionSpec.scale_name(f"{prefix}.{spec.dst}" if prefix else spec.dst)
            scale_rows = ceil_div(src_rows, BLOCK_SIZE)
        return cls(
            weight=weight,
            scale=scale,
            spec=spec,
            conversion=conversion,
            source_name=slot_key,
            slot_key=slot_key,
            scale_key=scale_key,
            inference_name=f"{prefix}.{spec.dst}" if prefix else spec.dst,
            inference_scale_name=inference_scale_name,
            offset_rows=offset_rows,
            scale_offset_rows=scale_offset_rows if conversion.requires_scale else None,
            rows=src_rows,
            scale_rows=scale_rows,
            my_rank=my_rank,
            trainer_ws=trainer_ws,
        )

    @property
    def buffers(self) -> list[tuple[str, Tensor, int]]:
        out: list[tuple[str, Tensor, int]] = [(self.slot_key, self.weight, 1)]
        if self.scale is not None:
            assert self.scale_key is not None
            out.append((self.scale_key, self.scale, 1))
        return out

    def convert(self, state_dict: dict[str, Tensor]) -> None:
        src = _source_for_transfer(state_dict[self.source_name])
        if isinstance(src, DTensor):
            src = src.full_tensor() if self.weight.shape[0] == src.shape[0] else src.to_local()
        self.conversion.fn(src, self.weight, self.scale)

    def layout_payload(self) -> list[LayoutEntry]:
        entries = [
            LayoutEntry(
                slot_key=self.slot_key,
                inference_name=self.inference_name,
                offset_rows=self.offset_rows,
                rows=self.rows,
                num_chunks=1,
            )
        ]
        if self.scale is not None:
            assert self.scale_key is not None and self.scale_rows is not None
            assert self.inference_scale_name is not None and self.scale_offset_rows is not None
            entries.append(
                LayoutEntry(
                    slot_key=self.scale_key,
                    inference_name=self.inference_scale_name,
                    offset_rows=self.scale_offset_rows,
                    rows=self.scale_rows,
                    num_chunks=1,
                )
            )
        return entries

    def build_writes(self, peers: list[PeerInfo]) -> list[WriteEntry]:
        out: list[WriteEntry] = []
        for i, peer in enumerate(peers):
            if i % self.trainer_ws != self.my_rank:
                continue
            for buf_key, _, _ in self.buffers:
                out.append(
                    WriteEntry(
                        local_buffer_key=buf_key,
                        local_chunk_idx=0,
                        peer_name=peer.agent_name,
                        remote_buffer_key=buf_key,
                        remote_chunk_idx=0,
                        tag=f"gather:{buf_key}",
                    )
                )
        return out

    def peer_chunk_descs(self, peer: PeerInfo) -> dict[str, list[tuple[int, int, int]]]:
        out: dict[str, list[tuple[int, int, int]]] = {}
        # Single chunk on the peer side, covering this slot's full row range
        # (``rows`` rows starting at ``offset_rows``).
        weight_row_bytes = self.weight.numel() * self.weight.element_size() // self.weight.shape[0]
        weight_base, _, weight_dev = peer.tensor_addrs[self.inference_name]
        out[self.slot_key] = [
            (
                weight_base + self.offset_rows * weight_row_bytes,
                self.rows * weight_row_bytes,
                weight_dev,
            )
        ]
        if self.scale is not None:
            assert self.scale_key is not None and self.scale_rows is not None
            assert self.inference_scale_name is not None and self.scale_offset_rows is not None
            scale_row_bytes = self.scale.numel() * self.scale.element_size() // self.scale.shape[0]
            scale_base, _, scale_dev = peer.tensor_addrs[self.inference_scale_name]
            out[self.scale_key] = [
                (
                    scale_base + self.scale_offset_rows * scale_row_bytes,
                    self.scale_rows * scale_row_bytes,
                    scale_dev,
                )
            ]
        return out


# --- Expert slot ----------------------------------------------------------- #


@dataclass
class ExpertSlot:
    """Fused stacked-expert slot. One 3D buffer holding ``num_local`` experts.

    Each local expert is one chunk; writes go per-(local, peer) pair filtered
    by the peer's ``expert_map`` (only peers that own a global expert receive
    a WRITE for it).
    """

    weight: Tensor  # (num_local, cat_dim_size, hidden)
    scale: Optional[Tensor]  # (num_local, ceil/128, ceil/128)
    spec: ConversionSpec
    conversion: ConversionEntry
    source_names: tuple[str, ...]
    slot_key: str  # also serves as the inference destination name
    scale_key: Optional[str]
    moe_prefix: str  # e.g. "model.layers.0.mlp.experts" — keys peer.expert_map
    owned_global_experts: list[int]  # index i is this weight's chunk i
    cat_dim: int

    @classmethod
    def from_spec(
        cls,
        spec: ConversionSpec,
        conversion: ConversionEntry,
        prefix: str,
        state_dict: dict[str, Tensor],
        parallel_dims: ParallelDims,
        slot_dtype: torch.dtype,
    ) -> "ExpertSlot":
        if parallel_dims.ep_enabled:
            ep_mesh = parallel_dims.get_mesh("ep")
            fsdp_mesh = parallel_dims.get_mesh("dp_shard_mod_ep")
            ep_size, ep_rank = ep_mesh.size(), ep_mesh.get_local_rank()
            fsdp_size, fsdp_rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()
        else:
            ep_size, ep_rank, fsdp_size, fsdp_rank = 1, 0, 1, 0

        sample = state_dict[f"{prefix}.{spec.sources[0]}"]
        local_sample = sample.to_local() if isinstance(sample, DTensor) else sample
        num_local_experts = local_sample.shape[0]
        if isinstance(sample, DTensor):
            total_experts = sample.shape[0]
            assert num_local_experts * fsdp_size * ep_size == total_experts, (
                f"EP partition mismatch for {spec.dst!r} at {prefix}: "
                f"local={num_local_experts} * fsdp={fsdp_size} * ep={ep_size} "
                f"!= total={total_experts}"
            )
        num_experts_per_ep = num_local_experts * fsdp_size
        base = ep_rank * num_experts_per_ep + fsdp_rank * num_local_experts
        owned_global_experts = list(range(base, base + num_local_experts))

        src_local_shapes = []
        for name in spec.sources:
            t = state_dict[f"{prefix}.{name}"]
            src_local_shapes.append((t.to_local() if isinstance(t, DTensor) else t).shape)
        dst_shape = list(src_local_shapes[0])
        dst_shape[spec.cat_dim] = sum(sh[spec.cat_dim] for sh in src_local_shapes)
        device = local_sample.device
        weight = torch.empty(tuple(dst_shape), dtype=slot_dtype, device=device)
        slot_key = f"{prefix}.{spec.dst}" if prefix else spec.dst
        moe_prefix = slot_key.rsplit(".", 1)[0]
        scale: Optional[Tensor] = None
        scale_key: Optional[str] = None
        if conversion.requires_scale:
            scale = torch.empty(
                (
                    weight.shape[0],
                    ceil_div(weight.shape[1], BLOCK_SIZE),
                    ceil_div(weight.shape[2], BLOCK_SIZE),
                ),
                dtype=torch.float32,
                device=device,
            )
            scale_key = ConversionSpec.scale_name(slot_key)
        return cls(
            weight=weight,
            scale=scale,
            spec=spec,
            conversion=conversion,
            source_names=tuple(f"{prefix}.{name}" for name in spec.sources),
            slot_key=slot_key,
            scale_key=scale_key,
            moe_prefix=moe_prefix,
            owned_global_experts=owned_global_experts,
            cat_dim=spec.cat_dim,
        )

    @property
    def num_local_experts(self) -> int:
        return self.weight.shape[0]

    @property
    def buffers(self) -> list[tuple[str, Tensor, int]]:
        n = self.num_local_experts
        out: list[tuple[str, Tensor, int]] = [(self.slot_key, self.weight, n)]
        if self.scale is not None:
            assert self.scale_key is not None
            out.append((self.scale_key, self.scale, n))
        return out

    def convert(self, state_dict: dict[str, Tensor]) -> None:
        srcs = []
        for name in self.source_names:
            t = _source_for_transfer(state_dict[name])
            srcs.append(t.to_local() if isinstance(t, DTensor) else t)
        tensor = srcs[0] if len(srcs) == 1 else torch.cat(srcs, dim=self.cat_dim)
        self.conversion.fn(tensor, self.weight, self.scale)

    def layout_payload(self) -> list[LayoutEntry]:
        # Expert slots route via peer.expert_map; no LayoutEntry needed.
        return []

    def build_writes(self, peers: list[PeerInfo]) -> list[WriteEntry]:
        out: list[WriteEntry] = []
        for local_idx, global_id in enumerate(self.owned_global_experts):
            for peer in peers:
                peer_experts = peer.expert_map.get(self.moe_prefix, [])
                if global_id not in peer_experts:
                    continue
                remote_idx = peer_experts.index(global_id)
                for buf_key, _, _ in self.buffers:
                    out.append(
                        WriteEntry(
                            local_buffer_key=buf_key,
                            local_chunk_idx=local_idx,
                            peer_name=peer.agent_name,
                            remote_buffer_key=buf_key,
                            remote_chunk_idx=remote_idx,
                            tag=f"expert:{buf_key}:E{global_id}",
                        )
                    )
        return out

    def peer_chunk_descs(self, peer: PeerInfo) -> dict[str, list[tuple[int, int, int]]]:
        # Peer's chunks = peer's local experts (one per ``expert_map[moe_prefix]`` entry),
        # each one global-expert-sized row of the 3D buffer.
        peer_local_experts = len(peer.expert_map.get(self.moe_prefix, []))
        out: dict[str, list[tuple[int, int, int]]] = {}

        per_expert_bytes_w = self.weight.numel() * self.weight.element_size() // self.weight.shape[0]
        weight_base, _, weight_dev = peer.tensor_addrs[self.slot_key]
        out[self.slot_key] = [
            (weight_base + i * per_expert_bytes_w, per_expert_bytes_w, weight_dev) for i in range(peer_local_experts)
        ]
        if self.scale is not None:
            assert self.scale_key is not None
            per_expert_bytes_s = self.scale.numel() * self.scale.element_size() // self.scale.shape[0]
            scale_base, _, scale_dev = peer.tensor_addrs[self.scale_key]
            out[self.scale_key] = [
                (scale_base + i * per_expert_bytes_s, per_expert_bytes_s, scale_dev) for i in range(peer_local_experts)
            ]
        return out


# --- Builders -------------------------------------------------------------- #


def build_slots_for_conversion_spec(
    spec: ConversionSpec,
    *,
    prefix: str,
    state_dict: dict[str, Tensor],
    parallel_dims: ParallelDims,
    default_conversion: str,
    base_dtype: torch.dtype,
) -> list[Slot]:
    """Instantiate every slot this spec produces at ``prefix``.

    Expert specs always yield one :class:`ExpertSlot`. Non-expert specs
    yield one slot per source: :class:`ShardedSlot` when dim 0 divides
    ``dp_shard*cp`` (and the shard size is FP8-block-aligned for quantized
    sources) and the source is large enough to amortize RDMA overhead;
    otherwise :class:`GatheredSlot`. Fused destinations may have one
    source land sharded and another gathered.
    """
    conversion_type = spec.conversion.conversion_type or default_conversion
    conversion = resolve(conversion_type, default_conversion)
    slot_dtype = _slot_dtype(conversion_type, conversion, base_dtype)
    if spec.is_expert_spec:
        return [
            ExpertSlot.from_spec(
                spec,
                conversion,
                prefix=prefix,
                state_dict=state_dict,
                parallel_dims=parallel_dims,
                slot_dtype=slot_dtype,
            )
        ]

    fsdp_total = parallel_dims.dp_shard * parallel_dims.cp
    slots: list[Slot] = []
    row_off = 0
    scale_row_off = 0
    for src_name in spec.sources:
        full_src = f"{prefix}.{src_name}" if prefix else src_name
        raw = state_dict[full_src]
        # Pass the RAW tensor (possibly DTensor) to from_spec — it needs the
        # global shape[0] to compute rows_per_shard = global_rows // fsdp_total.
        # Dispatch uses the global shape too: divisibility, FP8-block alignment,
        # and size threshold are all properties of the full (unfragmented) tensor.
        src_rows = raw.shape[0]
        per_shard = (
            src_rows % fsdp_total == 0
            and (not conversion.requires_scale or (src_rows // fsdp_total) % BLOCK_SIZE == 0)
            and raw.numel() * raw.element_size() >= SMALL_NON_EXPERT_BYTES
        )
        cls = ShardedSlot if per_shard else GatheredSlot
        slots.append(
            cls.from_spec(
                spec,
                conversion,
                prefix=prefix,
                src_name=src_name,
                src=raw,
                parallel_dims=parallel_dims,
                slot_dtype=slot_dtype,
                offset_rows=row_off,
                scale_offset_rows=scale_row_off,
            )
        )
        row_off += raw.shape[0]
        if conversion.requires_scale:
            scale_row_off += ceil_div(raw.shape[0], BLOCK_SIZE)
    return slots
