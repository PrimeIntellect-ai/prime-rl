"""Trainer-side destination slots for NIXL weight transfer.

Three slot types, sharing a uniform protocol:

* :class:`ShardedSlot` — non-expert param whose dim-0 is FSDP-shardable and
  large enough to shard. Slot holds this rank's shard. Writes to
  ``chunk[my_rank]`` on every inference peer.
* :class:`GatheredSlot` — non-expert param too small to shard or whose
  shape doesn't divide. Slot holds the full tensor. Written once per peer,
  round-robin across trainer ranks by ``i % trainer_ws == my_rank``.
* :class:`ExpertSlot` — MoE expert param, fused across local experts into a
  3D buffer. Writes to ``chunk[remote_idx]`` on peers that own each global
  expert (via vLLM's ``expert_map``).

Each slot captures everything it needs at construction — my_rank, trainer_ws,
and owned_global_experts are baked in via :meth:`from_spec` so runtime
methods take only dynamic inputs (peers, source state_dict, the NIXL agent).

Supporting types:

* :class:`WriteEntry` — one RDMA WRITE description. TransportPlan
  materializes these into ``agent.post_write`` calls by resolving
  ``local_buffer_key``/``remote_buffer_key`` into prep handles.
* :class:`LayoutEntry` — what a slot publishes to inference in SPG round 1
  so inference can narrow its vLLM tensor and build chunk descriptors.
  Experts don't use LayoutEntries (they use ``expert_map`` instead).
* :class:`PeerInfo` — one inference peer's round-2 payload, consumed by
  :meth:`Slot.build_writes` to decide where to route each transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor

from prime_rl.trainer.models.conversion_spec import ConversionSpec
from prime_rl.trainer.models.fp8 import BLOCK_SIZE, ceil_div
from prime_rl.trainer.parallel_dims import ParallelDims

# Source tensors smaller than this fall out of the per-shard NIXL path and are
# gathered instead — a single trainer rank writes the full tensor to each
# inference peer. Below ~2 MiB the per-shard shard size drops under 32 KiB per
# peer, at which point the RDMA handle overhead eats any parallelism gain.
SMALL_NON_EXPERT_BYTES = 2 * 1024 * 1024


# --- Wire / transfer types -------------------------------------------------- #


@dataclass
class LayoutEntry:
    """A single registered buffer the inference side needs to chunk up.

    Corresponds to one of the trainer-side slot's buffers (weight or scale).
    ``slot_key`` is the name under which the inference side will publish
    its serialized chunk descriptors in round 2; trainer-side lookups use
    the same key.
    """

    slot_key: str
    inference_name: str
    offset_rows: int
    rows: int
    num_chunks: int  # trainer_ws for per_shard, 1 for gather


@dataclass
class PeerInfo:
    """One inference peer's round-2 payload, in the shape trainer code needs.

    ``descriptors`` is keyed by ``LayoutEntry.slot_key`` (or the expert
    destination name) and maps to a list of serialized xfer dlists (one per
    chunk). ``expert_map`` maps MoE prefix → list of global expert IDs this
    peer owns.
    """

    agent_name: str
    descriptors: dict[str, list[bytes]]
    expert_map: dict[str, list[int]]


@dataclass
class WriteEntry:
    """One RDMA WRITE description. Resolved to NIXL prep handles by TransportPlan."""

    local_buffer_key: str  # key into TransportPlan._local_preps
    local_chunk_idx: int
    peer_name: str
    remote_buffer_key: str  # key into peer.descriptors
    remote_chunk_idx: int
    tag: str  # for diagnostics


# --- Slot protocol ---------------------------------------------------------- #


class Slot(Protocol):
    """Runtime interface every slot type implements.

    All slots own a ``weight`` tensor and an optional ``scale`` tensor.
    They expose:
      * :attr:`buffers` — iterable of ``(key, tensor, num_chunks)`` tuples
        for NIXL registration.
      * :meth:`convert` — pull sources from the live ``state_dict`` and
        write them into the buffers (quantize or dtype-cast).
      * :meth:`layout_payload` — what to send to inference in SPG round 1.
        Experts return an empty list (they route via ``expert_map``).
      * :meth:`build_writes` — given the list of inference peers, emit
        every RDMA WRITE this slot will post.
    """

    weight: Tensor
    scale: Optional[Tensor]
    spec: ConversionSpec

    @property
    def buffers(self) -> list[tuple[str, Tensor, int]]: ...
    def convert(self, state_dict: dict[str, Tensor]) -> None: ...
    def layout_payload(self) -> list[LayoutEntry]: ...
    def build_writes(self, peers: list[PeerInfo]) -> list[WriteEntry]: ...


# --- Helpers --------------------------------------------------------------- #


def _resolve_source(src: Tensor, slot_rows: int) -> Tensor:
    """Pick the rank-local slab or the full tensor to feed into quantization.

    If the slot's dim 0 equals the source's global dim 0 we want the full
    tensor (gather); otherwise we want this rank's FSDP shard (per_shard).
    Params are DTensors after ``fully_shard``; registered buffers (e.g.
    ``mlp.expert_bias``) stay plain and are used as-is.
    """
    if isinstance(src, DTensor):
        if slot_rows == src.shape[0]:
            src = src.full_tensor()
        else:
            src = src.to_local()
    return src


def _shard_rank_and_size(parallel_dims: ParallelDims) -> tuple[int, int]:
    """Rank + size along the FSDP shard axis (``dp_shard_cp``).

    With HSDP (``dp_replicate > 1``), the full params are replicated across
    replicas. Only the primary replica participates in NIXL, so slot
    chunking / write routing is indexed by this per-replica rank, not the
    global process rank.
    """
    if not dist.is_initialized():
        return 0, 1
    mesh = parallel_dims.get_mesh("dp_shard_cp")
    return mesh.get_local_rank(), mesh.size()


def _alloc_scale_2d(weight_shape: tuple[int, ...], device: torch.device) -> Tensor:
    """2D non-expert scale: both dims tile 128×128."""
    return torch.empty(
        (ceil_div(weight_shape[0], BLOCK_SIZE), ceil_div(weight_shape[1], BLOCK_SIZE)),
        dtype=torch.float32,
        device=device,
    )


def _alloc_scale_3d(weight_shape: tuple[int, ...], device: torch.device) -> Tensor:
    """3D expert scale: leading expert dim un-blocked; trailing two tile 128×128."""
    return torch.empty(
        (weight_shape[0], ceil_div(weight_shape[1], BLOCK_SIZE), ceil_div(weight_shape[2], BLOCK_SIZE)),
        dtype=torch.float32,
        device=device,
    )


# --- Non-expert slots ------------------------------------------------------ #


@dataclass
class ShardedSlot:
    """Non-expert slot holding one FSDP shard. Writes to every peer at chunk[my_rank]."""

    weight: Tensor
    scale: Optional[Tensor]
    spec: ConversionSpec
    source_name: str  # full name, e.g. "model.layers.0.self_attn.q_a_proj.weight"
    slot_key: str  # used as both local buffer key and remote descriptor key
    scale_key: Optional[str]
    inference_name: str
    inference_scale_name: Optional[str]
    offset_rows: int  # row offset of this source inside the fused vLLM dst
    scale_offset_rows: Optional[int]
    rows: int  # source's full dim-0
    scale_rows: Optional[int]
    my_rank: int
    trainer_ws: int

    @classmethod
    def from_spec(
        cls,
        spec: ConversionSpec,
        prefix: str,
        src_name: str,
        src: Tensor,
        parallel_dims: ParallelDims,
        offset_rows: int,
        scale_offset_rows: int,
    ) -> "ShardedSlot":
        fsdp_total = parallel_dims.dp_shard * parallel_dims.cp
        my_rank, trainer_ws = _shard_rank_and_size(parallel_dims)
        src_rows = src.shape[0]
        rows_per_shard = src_rows // fsdp_total
        weight = torch.empty(
            (rows_per_shard,) + tuple(src.shape[1:]),
            dtype=spec.slot_dtype,
            device=src.device,
        )
        slot_key = f"{prefix}.{src_name}"
        scale: Optional[Tensor] = None
        scale_key: Optional[str] = None
        inference_scale_name: Optional[str] = None
        scale_rows: Optional[int] = None
        if spec.quantized:
            scale = _alloc_scale_2d(weight.shape, weight.device)
            scale_key = spec.per_source_scale_key(slot_key)
            inference_scale_name = spec.scale_name(prefix)
            scale_rows = ceil_div(src_rows, BLOCK_SIZE)
        return cls(
            weight=weight,
            scale=scale,
            spec=spec,
            source_name=slot_key,
            slot_key=slot_key,
            scale_key=scale_key,
            inference_name=f"{prefix}.{spec.dst}",
            inference_scale_name=inference_scale_name,
            offset_rows=offset_rows,
            scale_offset_rows=scale_offset_rows if spec.quantized else None,
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
        value = _resolve_source(state_dict[self.source_name], self.weight.shape[0])
        self.spec.quantization.apply(value, self.weight, self.scale)

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


@dataclass
class GatheredSlot:
    """Non-expert slot holding the full tensor. Written once per peer, round-robin by trainer rank."""

    weight: Tensor
    scale: Optional[Tensor]
    spec: ConversionSpec
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
        prefix: str,
        src_name: str,
        src: Tensor,
        parallel_dims: ParallelDims,
        offset_rows: int,
        scale_offset_rows: int,
    ) -> "GatheredSlot":
        my_rank, trainer_ws = _shard_rank_and_size(parallel_dims)
        weight = torch.empty(tuple(src.shape), dtype=spec.slot_dtype, device=src.device)
        slot_key = f"{prefix}.{src_name}"
        scale: Optional[Tensor] = None
        scale_key: Optional[str] = None
        inference_scale_name: Optional[str] = None
        scale_rows: Optional[int] = None
        src_rows = src.shape[0]
        if spec.quantized:
            scale = _alloc_scale_2d(weight.shape, weight.device)
            scale_key = spec.per_source_scale_key(slot_key)
            inference_scale_name = spec.scale_name(prefix)
            scale_rows = ceil_div(src_rows, BLOCK_SIZE)
        return cls(
            weight=weight,
            scale=scale,
            spec=spec,
            source_name=slot_key,
            slot_key=slot_key,
            scale_key=scale_key,
            inference_name=f"{prefix}.{spec.dst}",
            inference_scale_name=inference_scale_name,
            offset_rows=offset_rows,
            scale_offset_rows=scale_offset_rows if spec.quantized else None,
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
        value = _resolve_source(state_dict[self.source_name], self.weight.shape[0])
        self.spec.quantization.apply(value, self.weight, self.scale)

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


# --- Expert slot ----------------------------------------------------------- #


@dataclass
class ExpertSlot:
    """Fused stacked-expert slot. One 3D buffer holding ``num_local`` experts.

    Each local expert is one chunk; writes go per-(local, peer) pair filtered
    by the peer's ``expert_map`` (only peers that own that global expert
    receive a WRITE for it).
    """

    weight: Tensor  # (num_local, cat_dim_size, hidden)
    scale: Optional[Tensor]  # (num_local, ceil/128, ceil/128)
    spec: ConversionSpec
    source_names: tuple[str, ...]  # full names, e.g. ("model.layers.0.mlp.experts.w1", ...)
    slot_key: str  # == inference_name
    scale_key: Optional[str]
    moe_prefix: str  # e.g. "model.layers.0.mlp.experts"
    owned_global_experts: list[int]  # index i of this list is the weight's chunk i
    cat_dim: int  # concat axis for the sources before quantize

    @classmethod
    def from_spec(
        cls,
        spec: ConversionSpec,
        prefix: str,
        state_dict: dict[str, Tensor],
        parallel_dims: ParallelDims,
    ) -> "ExpertSlot":
        if parallel_dims.ep_enabled:
            ep_mesh = parallel_dims.get_mesh("ep")
            fsdp_mesh = parallel_dims.get_mesh("dp_shard_mod_ep")
            ep_size, ep_rank = ep_mesh.size(), ep_mesh.get_local_rank()
            fsdp_size, fsdp_rank = fsdp_mesh.size(), fsdp_mesh.get_local_rank()
        else:
            ep_size, ep_rank, fsdp_size, fsdp_rank = 1, 0, 1, 0

        # The config is reached via any model param; we pull n_routed_experts from
        # the first source tensor's shape instead — its leading dim is the global
        # expert count (for stacked-expert sources like mlp.experts.w1).
        sample: DTensor = state_dict[f"{prefix}.{spec.sources[0]}"]
        num_local_experts = sample.to_local().shape[0]
        # Sanity check: when sample is a DTensor, its global shape carries
        # the full expert count and must factor cleanly into (ep × fsdp × local).
        # Catches EP>fsdp configs where the mesh wiring isn't what we assume.
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

        # Build the fused destination shape by summing each source's cat_dim.
        src_local_shapes = [state_dict[f"{prefix}.{name}"].to_local().shape for name in spec.sources]
        dst_shape = list(src_local_shapes[0])
        dst_shape[spec.cat_dim] = sum(sh[spec.cat_dim] for sh in src_local_shapes)
        device = sample.device
        weight = torch.empty(tuple(dst_shape), dtype=spec.slot_dtype, device=device)
        scale: Optional[Tensor] = None
        scale_key: Optional[str] = None
        if spec.quantized:
            scale = _alloc_scale_3d(tuple(dst_shape), device)
            scale_key = spec.scale_name(prefix)
        slot_key = f"{prefix}.{spec.dst}"
        # moe_prefix strips the trailing ".w13_weight" / ".w2_weight" etc. so
        # inference's expert_map lookup can key on e.g. "model.layers.0.mlp.experts".
        moe_prefix = slot_key.rsplit(".", 1)[0]
        return cls(
            weight=weight,
            scale=scale,
            spec=spec,
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
        srcs = [state_dict[name].to_local() for name in self.source_names]
        tensor = srcs[0] if len(srcs) == 1 else torch.cat(srcs, dim=self.cat_dim)
        self.spec.quantization.apply(tensor, self.weight, self.scale)

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
