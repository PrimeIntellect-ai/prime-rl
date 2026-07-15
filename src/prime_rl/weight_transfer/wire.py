"""Versioned metadata exchanged for pull-based weight transfer."""

from typing import Literal

import msgspec

PROTOCOL_VERSION = 1


class AgentDescriptor(msgspec.Struct, frozen=True):
    """A trainer-side NIXL agent that owns registered source memory."""

    name: str
    metadata: bytes


class TensorSegment(msgspec.Struct, frozen=True):
    """A contiguous portion of a logical HF tensor served by one agent.

    ``logical_offset`` and ``numel`` describe the segment in the tensor's
    contiguous logical element stream. ``address`` points at the same elements
    in registered trainer memory. Segments need not follow the trainer's
    parallel layout; together they must tile the logical tensor exactly once.
    """

    agent: int
    logical_offset: int
    numel: int
    address: int
    device_id: int


class PublishedTensor(msgspec.Struct, frozen=True):
    """One HF-named tensor exposed to inference consumers."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    segments: tuple[TensorSegment, ...]


class WeightManifest(msgspec.Struct, frozen=True):
    """Static, session-scoped description of all pullable model weights."""

    session_id: str
    epoch: int
    model: str
    fingerprint: str
    agents: tuple[AgentDescriptor, ...]
    tensors: tuple[PublishedTensor, ...]
    protocol_version: int = PROTOCOL_VERSION


class SyncSignal(msgspec.Struct, frozen=True):
    """Generation-aware control message for one synchronized update."""

    session_id: str
    epoch: int
    step: int
    phase: Literal["trainer_ready", "inference_applied"]
    rank: int
    fingerprint: str
    protocol_version: int = PROTOCOL_VERSION


def encode_manifest(manifest: WeightManifest) -> bytes:
    return msgspec.msgpack.encode(manifest)


def decode_manifest(data: bytes) -> WeightManifest:
    manifest = msgspec.msgpack.decode(data, type=WeightManifest)
    if manifest.protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported weight-transfer protocol {manifest.protocol_version}; expected {PROTOCOL_VERSION}"
        )
    return manifest


def encode_signal(signal: SyncSignal) -> bytes:
    return msgspec.msgpack.encode(signal)


def decode_signal(data: bytes) -> SyncSignal:
    signal = msgspec.msgpack.decode(data, type=SyncSignal)
    if signal.protocol_version != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported weight-transfer protocol {signal.protocol_version}; expected {PROTOCOL_VERSION}"
        )
    return signal
