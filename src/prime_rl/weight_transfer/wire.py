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


class TrainerTensor(msgspec.Struct):
    """One trainer-format tensor served as persistent BF16 shards.

    ``master_dtype`` documents the optimizer-owned source precision. ``dtype``
    is the wire precision and is deliberately independent of it.
    """

    name: str
    master_dtype: str
    dtype: str
    shape: tuple[int, ...]
    shards: list[TrainerShard]


class TrainerTable(msgspec.Struct):
    agents: list[TrainerAgent]
    tensors: list[TrainerTensor]


def encode_table(table: TrainerTable) -> bytes:
    return msgspec.msgpack.encode(table)


def decode_table(data: bytes) -> TrainerTable:
    return msgspec.msgpack.decode(data, type=TrainerTable)
