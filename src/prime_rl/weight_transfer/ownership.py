"""Select one serving owner for replicated trainer tensor shards."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import torch

from prime_rl.weight_transfer.publication import SourceTensor
from prime_rl.weight_transfer.sharding import SourceShard, validate_dim0_shards


@dataclass(frozen=True)
class ShardCandidate:
    """A local trainer tensor region that a rank can expose through NIXL."""

    rank: int
    name: str
    dtype: torch.dtype
    full_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    shape: tuple[int, ...]
    address: int
    device_id: int

    @property
    def region(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.global_offset, self.shape


def _choose_owner(name: str, region: tuple[tuple[int, ...], tuple[int, ...]], candidates: list[ShardCandidate]):
    ordered = sorted(candidates, key=lambda candidate: candidate.rank)
    digest = sha256(f"{name}:{region}".encode()).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def select_shard_owners(candidates: tuple[ShardCandidate, ...]) -> tuple[ShardCandidate, ...]:
    """Deduplicate replicas and select a deterministic serving rank per region.

    HSDP, CP, and replicated parameters can produce several candidates for the
    same logical region. Exactly one owner is selected deterministically, with
    different regions spread across available ranks. Trainer and inference
    topology are intentionally absent from this decision.
    """

    by_tensor: dict[str, list[ShardCandidate]] = {}
    for candidate in candidates:
        by_tensor.setdefault(candidate.name, []).append(candidate)

    selected_all: list[ShardCandidate] = []
    for name, tensor_candidates in sorted(by_tensor.items()):
        first = tensor_candidates[0]
        for candidate in tensor_candidates[1:]:
            if candidate.dtype != first.dtype or candidate.full_shape != first.full_shape:
                raise ValueError(
                    f"inconsistent source metadata for {name!r}: "
                    f"{first.dtype}/{first.full_shape} != {candidate.dtype}/{candidate.full_shape}"
                )

        by_region: dict[tuple[tuple[int, ...], tuple[int, ...]], list[ShardCandidate]] = {}
        for candidate in tensor_candidates:
            by_region.setdefault(candidate.region, []).append(candidate)
        selected = [_choose_owner(name, region, replicas) for region, replicas in sorted(by_region.items())]

        validation_shards = tuple(
            SourceShard(
                agent=candidate.rank,
                global_offset=candidate.global_offset,
                shape=candidate.shape,
                address=candidate.address,
                device_id=candidate.device_id,
            )
            for candidate in selected
        )
        validate_dim0_shards(first.full_shape, validation_shards)
        selected_all.extend(selected)
    return tuple(selected_all)


def select_source_tensors(
    candidates: tuple[ShardCandidate, ...], rank_to_agent: dict[int, int]
) -> tuple[SourceTensor, ...]:
    """Build logical source tensors from the selected registered candidates."""

    selected = select_shard_owners(candidates)
    by_tensor: dict[str, list[ShardCandidate]] = {}
    for candidate in selected:
        by_tensor.setdefault(candidate.name, []).append(candidate)

    sources: list[SourceTensor] = []
    for name, tensor_candidates in sorted(by_tensor.items()):
        first = tensor_candidates[0]

        shards: list[SourceShard] = []
        for candidate in tensor_candidates:
            if candidate.rank not in rank_to_agent:
                raise KeyError(f"selected trainer rank {candidate.rank} has no published NIXL agent")
            shards.append(
                SourceShard(
                    agent=rank_to_agent[candidate.rank],
                    global_offset=candidate.global_offset,
                    shape=candidate.shape,
                    address=candidate.address,
                    device_id=candidate.device_id,
                )
            )
        sources.append(SourceTensor(name, first.dtype, first.full_shape, tuple(shards)))
    return tuple(sources)
