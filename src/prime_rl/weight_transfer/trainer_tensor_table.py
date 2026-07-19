"""Trainer tensor metadata published for one-sided NIXL weight pulls."""

from __future__ import annotations

import msgspec


class TrainerAgent(msgspec.Struct, frozen=True):
    """One trainer rank's NIXL agent."""

    name: str
    metadata: bytes
    device_id: int


class TrainerShard(msgspec.Struct, frozen=True):
    """One contiguous flat range of a logical trainer tensor.

    ``offset`` and ``numel`` describe the logical element range. ``addr`` is
    the remote address of its first wire-dtype element.
    """

    agent: int
    offset: int
    numel: int
    addr: int


class TrainerTensor(msgspec.Struct):
    """One trainer-format tensor served as typed wire shards.

    ``wire_dtype`` is BF16 by default and may be FP32 for model-declared
    precision-sensitive tensors. Shard addresses may overlap with addresses
    used by tensors in other transfer groups.
    """

    name: str
    wire_dtype: str
    shape: tuple[int, ...]
    shards: list[TrainerShard]


class TrainerGroup(msgspec.Struct):
    """One independently staged and acknowledged transfer unit."""

    name: str
    tensors: list[TrainerTensor]


class TrainerTensorTable(msgspec.Struct):
    agents: list[TrainerAgent]
    source_ring_size: int
    groups: list[TrainerGroup]

    def encode(self) -> bytes:
        return msgspec.msgpack.encode(self)

    @classmethod
    def decode(cls, data: bytes) -> TrainerTensorTable:
        return msgspec.msgpack.decode(data, type=cls)
