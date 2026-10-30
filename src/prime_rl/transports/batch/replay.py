"""Packed micro batches recorded on disk, so the trainer can train without an orchestrator.

A replay directory holds ``meta.json`` and one ``step_<n>/rank_<r>.bin`` per step and data rank,
each the msgpack-encoded ``list[MicroBatch]`` the orchestrator would have sent that rank.
"""

from pathlib import Path

import msgspec

from prime_rl.transports.batch.types import MicroBatch, encode_array
from prime_rl.utils.pathing import get_step_path


class ReplayMeta(msgspec.Struct):
    """The trainer topology a replay directory was packed for."""

    num_steps: int
    seq_len: int
    num_train_workers: int
    pad_to_multiple_of: int


def get_replay_meta_path(directory: Path) -> Path:
    return directory / "meta.json"


def get_replay_step_path(directory: Path, step: int, data_rank: int) -> Path:
    return get_step_path(directory, step) / f"rank_{data_rank}.bin"


def write_replay_meta(directory: Path, meta: ReplayMeta) -> None:
    get_replay_meta_path(directory).write_bytes(msgspec.json.encode(meta))


def read_replay_meta(directory: Path) -> ReplayMeta:
    return msgspec.json.decode(get_replay_meta_path(directory).read_bytes(), type=ReplayMeta)


def write_replay_step(directory: Path, step: int, micro_batch_grid: list[list[MicroBatch]]) -> None:
    """Write one step's micro batches, one file per data rank."""
    encoder = msgspec.msgpack.Encoder(enc_hook=encode_array)
    get_step_path(directory, step).mkdir(parents=True, exist_ok=True)
    for data_rank, micro_batches in enumerate(micro_batch_grid):
        get_replay_step_path(directory, step, data_rank).write_bytes(encoder.encode(micro_batches))


def read_replay_step(directory: Path, step: int, data_rank: int) -> list[MicroBatch]:
    decoder = msgspec.msgpack.Decoder(type=list[MicroBatch])
    return decoder.decode(get_replay_step_path(directory, step, data_rank).read_bytes())
