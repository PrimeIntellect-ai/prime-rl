"""Wire types the trainer publishes to Model Express for the NIXL pull.

The trainer master publishes a single :class:`TrainerTable`: every NIXL agent
in the trainer's serving group, plus, for every state-dict tensor, the set of
dim-0 *shards* that make up the full logical tensor and which agent (rank)
serves each one. Parameters are sharded only along dim 0 (FSDP shards the
output/vocab dim, expert-parallel shards the expert dim), so a shard is a
contiguous dim-0 range served from one agent's registered buffer.

Workers are pure consumers — they publish nothing, bake their own pull plans
from the table, and RDMA-READ only the slices they need straight from the
owning trainer ranks. No gather, no conversion, no per-consumer state on the
trainer.
"""

from __future__ import annotations

import msgspec

# Filesystem marker created by the trainer master in a run's broadcast step
# directory once every rank's shard buffers hold this step's weights. Each
# inference worker's update RPC blocks on it before pulling — step-scoped by
# construction, so a marker from a previous sync can never be observed as fresh.
NIXL_DONE_MARKER = "NIXL_DONE"

# Per-worker ack marker (suffixed with the worker's global rank), created once
# that worker's pull completed. The trainer blocks on the full set before
# returning from a broadcast, so no rank tears down its NIXL agent (the trainer
# exits right after the final sync) or refreshes its shard buffers under an
# in-flight pull.
NIXL_PULLED_MARKER = "NIXL_PULLED"


class TrainerAgent(msgspec.Struct, frozen=True):
    """One trainer rank's NIXL agent serving a partition of the weights."""

    name: str
    metadata: bytes


class TrainerShard(msgspec.Struct, frozen=True):
    """One contiguous dim-0 run of a logical tensor, served by one agent.

    Global dim-0 indices ``[row_start, row_start + num_rows)`` of the full
    logical tensor live on ``agent`` (index into :attr:`TrainerTable.agents`)
    at device address ``addr`` (the byte address of local row 0), laid out
    C-contiguously with ``row_bytes`` per dim-0 row.
    """

    agent: int
    row_start: int
    num_rows: int
    addr: int
    row_bytes: int
    device_id: int


class TrainerTensor(msgspec.Struct):
    """One logical (prime-format) tensor and the shards that tile it on dim 0."""

    name: str
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
