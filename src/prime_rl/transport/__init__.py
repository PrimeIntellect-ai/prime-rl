from pathlib import Path

from prime_rl.configs.shared import TransportConfig
from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
    MultiRunFileSystemTrainingBatchSender,
)
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample
from prime_rl.transport.zmq import (
    ZMQMicroBatchReceiver,
    ZMQMicroBatchSender,
    ZMQTrainingBatchReceiver,
    ZMQTrainingBatchSender,
)


def setup_training_batch_sender(output_dir: Path, transport: TransportConfig) -> TrainingBatchSender:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)
    elif transport.type == "zmq":
        return ZMQTrainingBatchSender(output_dir, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_multi_run_training_batch_sender(
    output_dir: Path, run_names: list[str], transport: TransportConfig
) -> MultiRunFileSystemTrainingBatchSender:
    if transport.type != "filesystem":
        raise ValueError(f"Multi-run sender only supports filesystem transport, got: {transport.type}")
    return MultiRunFileSystemTrainingBatchSender(output_dir, run_names)


def setup_training_batch_receiver(transport: TransportConfig) -> TrainingBatchReceiver:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchReceiver()
    elif transport.type == "zmq":
        return ZMQTrainingBatchReceiver(transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


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
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "MultiRunFileSystemTrainingBatchSender",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_multi_run_training_batch_sender",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
