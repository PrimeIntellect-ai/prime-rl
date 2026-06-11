"""Wire types the trainer publishes to Model Express for the NIXL pull.

The trainer master publishes a single :class:`TrainerTable` (msgpack-encoded
into ``p2p_pb2.WorkerMetadata.nixl_metadata``): every NIXL agent in the
trainer's serving group plus one :class:`TrainerTensor` row per state-dict
entry, addressed inside the owning agent's registered weight store. Workers
are pure consumers — they publish nothing, so inference can scale out or
restart without the trainer ever rebuilding a plan.
"""

from __future__ import annotations

import msgspec

# Filesystem marker created by the trainer master in a run's broadcast step
# directory once the weight store holds this step's weights. Each inference
# worker's update RPC blocks on it before pulling — step-scoped by
# construction, so a marker from a previous sync can never be observed as
# fresh.
NIXL_DONE_MARKER = "NIXL_DONE"


class TrainerAgent(msgspec.Struct, frozen=True):
    """One trainer rank's NIXL agent serving a partition of the store."""

    name: str
    metadata: bytes


class TrainerTensor(msgspec.Struct, frozen=True):
    """One state-dict tensor in the trainer's weight store.

    ``name``/``dtype``/``shape`` describe the tensor exactly as the trainer
    holds it (native naming, full/unsharded shape, C-contiguous at ``addr``).
    ``agent`` indexes into :attr:`TrainerTable.agents`.
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    addr: int
    device_id: int
    agent: int


class TrainerTable(msgspec.Struct):
    agents: list[TrainerAgent]
    tensors: list[TrainerTensor]


def encode_table(table: TrainerTable) -> bytes:
    return msgspec.msgpack.encode(table)


def decode_table(data: bytes) -> TrainerTable:
    return msgspec.msgpack.decode(data, type=TrainerTable)
