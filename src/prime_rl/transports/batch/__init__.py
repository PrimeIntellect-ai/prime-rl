from pathlib import Path

from prime_rl.configs.shared import TransportConfig
from prime_rl.transports.batch.base import BatchReceiver, BatchSender
from prime_rl.transports.batch.filesystem import (
    FileSystemBatchReceiver,
    FileSystemBatchSender,
)
from prime_rl.transports.batch.types import (
    MicroBatch,
    RoutedExperts,
    TrainingSample,
)
from prime_rl.transports.batch.zmq import (
    ZMQBatchReceiver,
    ZMQBatchSender,
)


def setup_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfig
) -> BatchSender:
    if transport.type == "filesystem":
        return FileSystemBatchSender(output_dir, data_world_size, current_step)
    elif transport.type == "zmq":
        return ZMQBatchSender(output_dir, data_world_size, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfig
) -> BatchReceiver:
    if transport.type == "filesystem":
        return FileSystemBatchReceiver(output_dir, data_rank, current_step)
    elif transport.type == "zmq":
        return ZMQBatchReceiver(output_dir, data_rank, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemBatchSender",
    "FileSystemBatchReceiver",
    "ZMQBatchSender",
    "ZMQBatchReceiver",
    "BatchReceiver",
    "BatchSender",
    "TrainingSample",
    "MicroBatch",
    "RoutedExperts",
    "setup_batch_sender",
    "setup_batch_receiver",
]
