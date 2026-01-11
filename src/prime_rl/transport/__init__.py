from pathlib import Path

from prime_rl.transport.base import MicroBatchReceiver, MicroBatchSender, TrainingBatchReceiver, TrainingBatchSender
from prime_rl.transport.config import TransportConfigType
from prime_rl.transport.filesystem import (
    FileSystemMicroBatchReceiver,
    FileSystemMicroBatchSender,
    FileSystemTrainingBatchReceiver,
    FileSystemTrainingBatchSender,
)
from prime_rl.transport.tcpstore import TCPStoreMicroBatchReceiver, TCPStoreMicroBatchSender
from prime_rl.transport.types import MicroBatch, TrainingBatch, TrainingSample
from prime_rl.transport.zmq import (
    ZMQMicroBatchReceiver,
    ZMQMicroBatchSender,
    ZMQTrainingBatchReceiver,
    ZMQTrainingBatchSender,
)


def setup_training_batch_sender(output_dir: Path, transport: TransportConfigType) -> TrainingBatchSender:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchSender(output_dir)
    elif transport.type == "zmq":
        return ZMQTrainingBatchSender(output_dir, transport)
    elif transport.type == "tcpstore":
        raise ValueError("TCPStore transport is only supported for micro batches, not training batches")
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_training_batch_receiver(transport: TransportConfigType) -> TrainingBatchReceiver:
    if transport.type == "filesystem":
        return FileSystemTrainingBatchReceiver()
    elif transport.type == "zmq":
        return ZMQTrainingBatchReceiver(transport)
    elif transport.type == "tcpstore":
        raise ValueError("TCPStore transport is only supported for micro batches, not training batches")
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_sender(
    output_dir: Path, data_world_size: int, current_step: int, transport: TransportConfigType
) -> MicroBatchSender:
    if transport.type == "filesystem":
        return FileSystemMicroBatchSender(output_dir, data_world_size, current_step)
    elif transport.type == "zmq":
        return ZMQMicroBatchSender(output_dir, data_world_size, current_step, transport)
    elif transport.type == "tcpstore":
        return TCPStoreMicroBatchSender(output_dir, data_world_size, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


def setup_micro_batch_receiver(
    output_dir: Path, data_rank: int, current_step: int, transport: TransportConfigType
) -> MicroBatchReceiver:
    if transport.type == "filesystem":
        return FileSystemMicroBatchReceiver(output_dir, data_rank, current_step)
    elif transport.type == "zmq":
        return ZMQMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    elif transport.type == "tcpstore":
        return TCPStoreMicroBatchReceiver(output_dir, data_rank, current_step, transport)
    else:
        raise ValueError(f"Invalid transport type: {transport.type}")


__all__ = [
    "FileSystemTrainingBatchSender",
    "FileSystemTrainingBatchReceiver",
    "FileSystemMicroBatchSender",
    "FileSystemMicroBatchReceiver",
    "MicroBatchReceiver",
    "MicroBatchSender",
    "TCPStoreMicroBatchSender",
    "TCPStoreMicroBatchReceiver",
    "TrainingSample",
    "TrainingBatch",
    "MicroBatch",
    "setup_training_batch_sender",
    "setup_training_batch_receiver",
    "setup_micro_batch_sender",
    "setup_micro_batch_receiver",
]
