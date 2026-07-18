"""Metadata published by the trainer for one-sided NIXL weight pulls."""

from __future__ import annotations

import msgspec


class TrainerAgent(msgspec.Struct, frozen=True):
    """One trainer rank's NIXL agent."""

    name: str
    metadata: bytes


class TrainerShard(msgspec.Struct, frozen=True):
    """A contiguous dim-0 range of one logical trainer tensor."""

    agent: int
    row_start: int
    num_rows: int
    addr: int
    row_bytes: int
    device_id: int


class TrainerGatheredShard(msgspec.Struct, frozen=True):
    """A dim-0 shard at an offset relative to a gathered group replica."""

    row_start: int
    num_rows: int
    offset_bytes: int
    row_bytes: int


class TrainerReplica(msgspec.Struct, frozen=True):
    """One complete replica of a gathered transfer group's packed tensors."""

    agent: int
    addr: int
    device_id: int


class TrainerGatheredGroup(msgspec.Struct, frozen=True):
    """Registered bases containing the same rank-major gathered group."""

    group: int
    replicas: list[TrainerReplica]


class TrainerTensor(msgspec.Struct):
    """One trainer-format tensor served as BF16 shards.

    ``master_dtype`` documents the optimizer-owned source precision. ``dtype``
    is the wire precision and is deliberately independent of it. Shard
    addresses are valid for this tensor's transfer group and may overlap with
    addresses used by other groups. Gathered shard offsets are relative to any
    replica base published for the group.
    """

    name: str
    master_dtype: str
    dtype: str
    shape: tuple[int, ...]
    group: int
    shards: list[TrainerShard]
    gathered_shards: list[TrainerGatheredShard]


class TrainerTable(msgspec.Struct):
    agents: list[TrainerAgent]
    groups: list[str]
    buffer_count: int
    tensors: list[TrainerTensor]
    gathered_groups: list[TrainerGatheredGroup]


def encode_table(table: TrainerTable) -> bytes:
    return msgspec.msgpack.encode(table)


def decode_table(data: bytes) -> TrainerTable:
    return msgspec.msgpack.decode(data, type=TrainerTable)
