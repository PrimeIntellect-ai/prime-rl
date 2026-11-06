from pathlib import Path

from prime_rl.configs.shared import TransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.filesystem import (
    FileSystemBatchReceiver,
    FileSystemBatchSender,
)
from prime_rl.transports.batch.journal import BatchJournal
from prime_rl.transports.batch.recovery import TrainerGroup
from prime_rl.transports.batch.types import (
    MicroBatch,
    MMImageRef,
    MMRefs,
    RoutedExperts,
    SamplingMask,
    TrainingSample,
)
from prime_rl.transports.batch.zmq import (
    ZMQBatchReceiver,
    ZMQBatchSender,
)


def setup_batch_sender(
    output_dir: Path,
    data_world_size: int,
    current_step: int,
    transport: TransportConfig,
    journal: bool = False,
    auto_recover: bool = False,
) -> BatchSender:
    if transport.type == "filesystem":
        if auto_recover:
            raise ValueError("Trainer recovery requires the ZMQ batch transport")
        sender = FileSystemBatchSender(output_dir, data_world_size, current_step)
    elif transport.type == "zmq":
        sender = ZMQBatchSender(output_dir, data_world_size, current_step, transport, auto_recover=auto_recover)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")
    if journal:
        sender.journal = BatchJournal(output_dir)
    return sender


def setup_batch_receiver(
    output_dir: Path,
    data_rank: int,
    rank: int,
    world_size: int,
    current_step: int,
    transport: TransportConfig,
    trainer_group: TrainerGroup | None = None,
) -> BatchReceiver:
    if transport.type == "filesystem":
        return FileSystemBatchReceiver(output_dir, data_rank, rank, world_size, current_step)
    elif transport.type == "zmq":
        if trainer_group is None:
            raise ValueError("ZMQ batch transport requires a shared trainer group identity")
        return ZMQBatchReceiver(output_dir, data_rank, rank, world_size, current_step, transport, trainer_group)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemBatchSender",
    "FileSystemBatchReceiver",
    "ZMQBatchSender",
    "ZMQBatchReceiver",
    "BatchReceiver",
    "BatchSender",
    "MMImageRef",
    "MMRefs",
    "TrainingSample",
    "MicroBatch",
    "SamplingMask",
    "RoutedExperts",
    "setup_batch_sender",
    "setup_batch_receiver",
]
