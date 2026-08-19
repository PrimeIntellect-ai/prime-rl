from pathlib import Path

from prime_rl.configs.shared import TransportConfig
from prime_rl.transports.rollouts.base import MicroBatchReceiver, MicroBatchSender
from prime_rl.transports.rollouts.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
)
from prime_rl.transports.rollouts.types import (
    MicroBatch,
    RoutedExperts,
    TrainingSample,
)
from prime_rl.transports.rollouts.zmq import (
    ZMQMicroBatchReceiver,
    ZMQMicroBatchSender,
)


def setup_micro_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfig
) -> MicroBatchSender:
    if transport.type == "filesystem":
        return FileSystemMicroBatchSender(output_dir, data_world_size, current_step)
    elif transport.type == "zmq":
        return ZMQMicroBatchSender(output_dir, data_world_size, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfig
) -> MicroBatchReceiver:
    if transport.type == "filesystem":
        return FileSystemMicroBatchReceiver(output_dir, data_rank, current_step)
    elif transport.type == "zmq":
        return ZMQMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "ZMQMicroBatchSender",
    "ZMQMicroBatchReceiver",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TrainingSample",
    "MicroBatch",
    "RoutedExperts",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
